"""
com-DBLP Reconstruction with Edge Provenance

Rebuilds the SNAP com-DBLP overlapping-community benchmark from the raw DBLP
XML, following Yang & Leskovec, "Defining and Evaluating Network Communities
based on Ground-truth" (ICDM 2012):

  * nodes are authors, and two authors are linked if they co-authored a paper;
  * every publication venue (journal or conference) defines a group of the
    authors who published there. By default each venue is one ground-truth
    community, as in the paper's Table I; with --community-mode component,
    each connected component of the venue's induced subgraph is a separate
    community, as the SNAP release documents (size >= 3 either way);
  * communities are ranked by their average rank over six quality scores
    (conductance, Flake-ODF, FOMD, TPR, cut ratio, modularity), and the
    top 5,000 form the benchmark.

Unlike the SNAP release, which only ships integer IDs, this keeps author names
and, for every edge, the papers behind it, so edge features can be computed
from paper content later.

Usage:
    python src/com_dblp.py --xml data/dblp.xml --out data/com_dblp_ef
"""

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

try:
    from .dblp_parser import DBLPStreamParser
except ImportError:  # Fallback for direct script execution
    from dblp_parser import DBLPStreamParser

logger = logging.getLogger(__name__)

# Statistics of the SNAP release (https://snap.stanford.edu/data/com-DBLP.html),
# kept for comparison. The DBLP snapshot SNAP used is not documented, so a
# current dump will not match exactly.
SNAP_REFERENCE = {
    "nodes": 317080,
    "edges": 1049866,
    "communities_all": 13477,
    "communities_top": 5000,
}

# DBLP row of Table I in Yang & Leskovec (2012), one community per venue
PAPER_REFERENCE = {
    "nodes": 420000,
    "edges": 1340000,
    "communities_all": 2547,
    "all_mean_size": 429.79,
    "all_memberships_per_node": 2.56,
}

# Score name -> True if a higher value means a better community
QUALITY_SCORES = {
    "conductance": False,
    "flake_odf": False,
    "fomd": True,
    "tpr": True,
    "cut_ratio": False,
    "modularity": True,
}


def venue_id_from_key(key: str) -> str:
    """Map a DBLP key such as 'conf/kdd/YangL12' to its venue 'conf/kdd'."""
    parts = key.split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else ""


def strip_volume(booktitle: str) -> str:
    """'ECCV (2)' -> 'ECCV': volumes of one proceedings share a track."""
    return re.sub(r"\s*\(\d+\)$", "", booktitle)


class PaperTable:
    """Compact store of the papers kept after filtering."""

    def __init__(self):
        self.keys: List[str] = []
        self.titles: List[str] = []
        self.dois: List[str] = []
        self.booktitles: List[str] = []
        self.crossrefs: List[str] = []
        self.years: List[int] = []
        self.venue_ids: List[int] = []
        self.author_ids: List[np.ndarray] = []

        self.author_name_to_id: Dict[str, int] = {}
        self.author_names: List[str] = []
        self.venue_key_to_id: Dict[str, int] = {}
        self.venue_keys: List[str] = []
        self.venue_names: List[str] = []

    def _author_id(self, name: str) -> int:
        aid = self.author_name_to_id.get(name)
        if aid is None:
            aid = len(self.author_names)
            self.author_name_to_id[name] = aid
            self.author_names.append(name)
        return aid

    def _venue_id(self, venue_key: str) -> int:
        vid = self.venue_key_to_id.get(venue_key)
        if vid is None:
            vid = len(self.venue_keys)
            self.venue_key_to_id[venue_key] = vid
            self.venue_keys.append(venue_key)
        return vid

    def add(self, paper: Dict, venue_key: str):
        # A name listed twice on one paper is still one author
        authors = sorted({self._author_id(a) for a in paper["authors"]})
        self.keys.append(paper["key"])
        self.titles.append(paper["title"])
        self.dois.append(paper.get("doi", ""))
        self.booktitles.append(paper["venue"])
        self.crossrefs.append(paper.get("crossref", ""))
        self.years.append(paper["year"] if paper["year"] is not None else -1)
        self.venue_ids.append(self._venue_id(venue_key))
        self.author_ids.append(np.asarray(authors, dtype=np.int64))

    def keep(self, mask: np.ndarray):
        """Drop the papers where mask is False."""
        for name in ("keys", "titles", "dois", "booktitles", "crossrefs",
                     "years", "venue_ids", "author_ids"):
            values = getattr(self, name)
            setattr(self, name, [x for x, k in zip(values, mask) if k])

    def name_venues(self):
        """Label each venue with its most frequent journal/booktitle."""
        counts = defaultdict(Counter)
        for vid, title in zip(self.venue_ids, self.booktitles):
            counts[vid][strip_volume(title)] += 1
        self.venue_names = [counts[vid].most_common(1)[0][0] if counts[vid] else ""
                            for vid in range(len(self.venue_keys))]

    def __len__(self):
        return len(self.keys)


class ComDBLPBuilder:
    """
    Build the co-authorship graph, its edge-to-paper provenance and the
    venue-based overlapping ground-truth communities.
    """

    def __init__(self,
                 pub_types: Iterable[str] = ("article", "inproceedings"),
                 include_informal: bool = False,
                 min_year: Optional[int] = None,
                 max_year: Optional[int] = None,
                 max_authors: int = 50,
                 min_community_size: int = 3,
                 top_k: int = 5000,
                 largest_component: bool = True,
                 community_mode: str = "venue",
                 main_volume_only: bool = False):
        if community_mode not in ("venue", "component"):
            raise ValueError(f"Unknown community_mode: {community_mode}")
        self.pub_types = set(pub_types)
        self.include_informal = include_informal
        self.min_year = min_year
        self.max_year = max_year
        self.max_authors = max_authors
        self.min_community_size = min_community_size
        self.top_k = top_k
        self.largest_component = largest_component
        self.community_mode = community_mode
        self.main_volume_only = main_volume_only

        self.papers = PaperTable()
        self.skipped = defaultdict(int)

    # ------------------------------------------------------------------
    # Step 1: collect papers
    # ------------------------------------------------------------------
    def collect(self, papers: Iterable[Dict], limit: Optional[int] = None):
        for paper in papers:
            if limit is not None and len(self.papers) >= limit:
                break
            reason = self._reject_reason(paper)
            if reason:
                self.skipped[reason] += 1
                continue
            self.papers.add(paper, venue_id_from_key(paper["key"]))

        if self.main_volume_only:
            self._keep_main_volumes()
        self.papers.name_venues()
        logger.info(f"Kept {len(self.papers):,} papers by "
                    f"{len(self.papers.author_names):,} authors in "
                    f"{len(self.papers.venue_keys):,} venues")
        logger.info(f"Skipped: {dict(self.skipped)}")

    def _reject_reason(self, paper: Dict) -> Optional[str]:
        if paper["pub_type"] not in self.pub_types:
            return "pub_type"
        if not self.include_informal and paper.get("publtype") == "informal":
            return "informal"
        if not venue_id_from_key(paper["key"]):
            return "no_venue"
        year = paper["year"]
        if self.min_year is not None and (year is None or year < self.min_year):
            return "year"
        if self.max_year is not None and (year is None or year > self.max_year):
            return "year"
        if self.max_authors and len(set(paper["authors"])) > self.max_authors:
            return "too_many_authors"
        return None

    def _keep_main_volumes(self):
        """
        Keep only each venue-year's main track, which drops satellite
        workshops that DBLP files under the conference key (e.g. "ICDE
        Workshops" under conf/icde). The main track is the booktitle with the
        most papers that year, ignoring volume numbers, so multi-volume
        proceedings such as "ECCV (1)" ... "ECCV (4)" stay together.
        """
        papers = self.papers
        tracks = [strip_volume(title) for title in papers.booktitles]
        counts = defaultdict(Counter)
        for vid, year, track in zip(papers.venue_ids, papers.years, tracks):
            counts[(vid, year)][track] += 1
        main = {key: c.most_common(1)[0][0] for key, c in counts.items()}
        mask = np.array([track == main[(vid, year)]
                         for vid, year, track in zip(papers.venue_ids, papers.years,
                                                     tracks)])
        self.skipped["non_main_volume"] += int((~mask).sum())
        papers.keep(mask)

    # ------------------------------------------------------------------
    # Step 2: co-authorship graph with edge -> papers provenance
    # ------------------------------------------------------------------
    def build_graph(self):
        num_authors = len(self.papers.author_names)
        pair_cache = {}
        us, vs, ps = [], [], []

        for pid, authors in enumerate(self.papers.author_ids):
            k = len(authors)
            if k < 2:
                continue
            if k not in pair_cache:
                pair_cache[k] = np.triu_indices(k, 1)
            i, j = pair_cache[k]
            us.append(authors[i])
            vs.append(authors[j])
            ps.append(np.full(len(i), pid, dtype=np.int64))

        u = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
        v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64)
        p = np.concatenate(ps) if ps else np.zeros(0, dtype=np.int64)

        # Group (author pair, paper) incidences by pair; u < v already holds
        pair_key = u * num_authors + v
        order = np.argsort(pair_key, kind="stable")
        pair_key, p = pair_key[order], p[order]
        uniq, start = np.unique(pair_key, return_index=True)

        edge_u, edge_v = uniq // num_authors, uniq % num_authors
        edge_ptr = np.append(start, len(p))
        logger.info(f"Co-authorship graph: {len(edge_u):,} edges from "
                    f"{len(p):,} (pair, paper) incidences")

        node_ids = np.arange(num_authors)
        if self.largest_component:
            adj = self._adjacency(edge_u, edge_v, num_authors)
            _, labels = connected_components(adj, directed=False)
            lcc = np.bincount(labels).argmax()
            node_ids = np.flatnonzero(labels == lcc)
            keep = labels[edge_u] == lcc
            edge_ptr, p = self._filter_csr(edge_ptr, p, keep)
            edge_u, edge_v = edge_u[keep], edge_v[keep]
            logger.info(f"Largest component: {len(node_ids):,} nodes, "
                        f"{len(edge_u):,} edges")

        # Re-index nodes to 0..n-1
        remap = np.full(num_authors, -1, dtype=np.int64)
        remap[node_ids] = np.arange(len(node_ids))

        self.node_author_ids = node_ids
        self.edge_u = remap[edge_u]
        self.edge_v = remap[edge_v]
        self.edge_ptr = edge_ptr
        self.edge_papers = p
        self.num_nodes = len(node_ids)
        self.adj = self._adjacency(self.edge_u, self.edge_v, self.num_nodes)
        self._author_to_node = remap

    @staticmethod
    def _adjacency(u: np.ndarray, v: np.ndarray, n: int) -> sp.csr_matrix:
        data = np.ones(2 * len(u), dtype=np.int32)
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    @staticmethod
    def _filter_csr(ptr: np.ndarray, values: np.ndarray, keep: np.ndarray):
        counts = np.diff(ptr)[keep]
        mask = np.repeat(keep, np.diff(ptr))
        return np.append(0, np.cumsum(counts)), values[mask]

    # ------------------------------------------------------------------
    # Step 3: venue-based ground-truth communities
    # ------------------------------------------------------------------
    def build_communities(self):
        venue_members = defaultdict(set)
        for pid, authors in enumerate(self.papers.author_ids):
            nodes = self._author_to_node[authors]
            nodes = nodes[nodes >= 0]
            if len(nodes):
                venue_members[self.papers.venue_ids[pid]].update(nodes.tolist())

        degree = np.asarray(self.adj.sum(axis=1)).ravel()
        num_edges = len(self.edge_u)
        median_degree = float(np.median(degree))

        records = []
        for vid in sorted(venue_members):
            members = np.fromiter(sorted(venue_members[vid]), dtype=np.int64)
            if len(members) < self.min_community_size:
                continue
            sub = self.adj[members][:, members]
            if self.community_mode == "venue":
                # One community per venue, as in Yang & Leskovec's Table I
                num_comp, labels = 1, np.zeros(len(members), dtype=np.int64)
            else:
                # One community per connected component, as SNAP documents
                num_comp, labels = connected_components(sub, directed=False)
            for c in range(num_comp):
                local = np.flatnonzero(labels == c)
                if len(local) < self.min_community_size:
                    continue
                scores = self._quality_scores(
                    sub[local][:, local], degree[members[local]],
                    num_edges, median_degree)
                records.append({
                    "venue_key": self.papers.venue_keys[vid],
                    "venue_name": self.papers.venue_names[vid],
                    "nodes": members[local],
                    **scores,
                })

        self.communities = pd.DataFrame(records)
        logger.info(f"Ground-truth communities (all): {len(self.communities):,}")
        self._rank_communities()

    def _quality_scores(self, sub: sp.csr_matrix, deg: np.ndarray,
                        num_edges: int, median_degree: float) -> Dict[str, float]:
        """Yang & Leskovec (2012) community scoring functions."""
        n_s = sub.shape[0]
        k_in = np.asarray(sub.sum(axis=1)).ravel()
        m_s = k_in.sum() / 2.0
        vol = deg.sum()
        c_s = vol - 2.0 * m_s
        # Number of triangles through each node inside the community
        triangles = np.asarray(sub.multiply(sub @ sub).sum(axis=1)).ravel()

        return {
            "size": n_s,
            "internal_edges": int(m_s),
            "boundary_edges": int(c_s),
            "conductance": c_s / (2.0 * m_s + c_s) if (m_s + c_s) > 0 else 0.0,
            "flake_odf": float(np.mean(k_in < deg / 2.0)),
            "fomd": float(np.mean(k_in > median_degree)),
            "tpr": float(np.mean(triangles > 0)),
            "cut_ratio": c_s / (n_s * (self.num_nodes - n_s))
                         if self.num_nodes > n_s else 0.0,
            "modularity": 0.25 * (m_s - vol ** 2 / (4.0 * num_edges)),
        }

    def _rank_communities(self):
        if self.communities.empty:
            return
        ranks = []
        for name, higher_better in QUALITY_SCORES.items():
            values = self.communities[name].to_numpy(dtype=float)
            ranks.append(rankdata(-values if higher_better else values))
        self.communities["avg_rank"] = np.mean(ranks, axis=0)
        self.communities = (self.communities
                            .sort_values("avg_rank", kind="stable")
                            .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Step 4: export
    # ------------------------------------------------------------------
    def save(self, out_dir: str, config: Dict):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        names = self.papers.author_names
        pd.DataFrame({
            "node_id": np.arange(self.num_nodes),
            "name": [names[a] for a in self.node_author_ids],
        }).to_csv(out / "authors.tsv", sep="\t", index=False)

        # Edges with the papers behind them (indices into papers.jsonl)
        used_papers = np.unique(self.edge_papers)
        paper_remap = np.full(len(self.papers), -1, dtype=np.int64)
        paper_remap[used_papers] = np.arange(len(used_papers))
        edge_papers = paper_remap[self.edge_papers]
        ptr = self.edge_ptr
        pd.DataFrame({
            "edge_id": np.arange(len(self.edge_u)),
            "u": self.edge_u,
            "v": self.edge_v,
            "num_papers": np.diff(ptr),
            "paper_ids": [";".join(map(str, edge_papers[ptr[e]:ptr[e + 1]]))
                          for e in range(len(self.edge_u))],
        }).to_csv(out / "edges.tsv", sep="\t", index=False)

        with open(out / "papers.jsonl", "w", encoding="utf-8") as f:
            for new_id, pid in enumerate(used_papers):
                nodes = self._author_to_node[self.papers.author_ids[pid]]
                f.write(json.dumps({
                    "paper_id": new_id,
                    "key": self.papers.keys[pid],
                    "title": self.papers.titles[pid],
                    "doi": self.papers.dois[pid],
                    "booktitle": self.papers.booktitles[pid],
                    "crossref": self.papers.crossrefs[pid],
                    "year": self.papers.years[pid],
                    "venue_key": self.papers.venue_keys[self.papers.venue_ids[pid]],
                    "author_nodes": nodes[nodes >= 0].tolist(),
                }, ensure_ascii=False) + "\n")

        # SNAP-style community files: one community per line, tab-separated
        top = self.communities.head(self.top_k)
        for path, frame in ((out / "communities.all.txt", self.communities),
                            (out / f"communities.top{self.top_k}.txt", top)):
            with open(path, "w") as f:
                for nodes in frame["nodes"]:
                    f.write("\t".join(map(str, nodes)) + "\n")

        meta = self.communities.drop(columns="nodes")
        meta.insert(0, "community_id", np.arange(len(meta)))
        meta["in_top"] = meta["community_id"] < self.top_k
        meta.to_csv(out / "communities.meta.tsv", sep="\t", index=False)

        stats = self.stats()
        with open(out / "stats.json", "w") as f:
            json.dump({"config": config, "stats": stats,
                       "snap_reference": SNAP_REFERENCE,
                       "paper_reference": PAPER_REFERENCE,
                       "skipped_papers": dict(self.skipped)}, f, indent=2)
        logger.info(f"Saved dataset to {out}")
        return stats

    def stats(self) -> Dict:
        top = self.communities.head(self.top_k)
        memberships = np.zeros(self.num_nodes, dtype=np.int64)
        for nodes in top["nodes"]:
            memberships[nodes] += 1
        covered = memberships > 0
        papers_per_edge = np.diff(self.edge_ptr)
        all_sizes = self.communities["size"] if len(self.communities) else []
        return {
            "nodes": int(self.num_nodes),
            "edges": int(len(self.edge_u)),
            "papers_kept": int(len(self.papers)),
            "papers_backing_edges": int(len(np.unique(self.edge_papers))),
            "mean_papers_per_edge": float(papers_per_edge.mean())
                                    if len(papers_per_edge) else 0.0,
            "communities_all": int(len(self.communities)),
            # Yang & Leskovec's S and A: mean size, memberships per node
            "all_mean_size": float(np.mean(all_sizes)) if len(all_sizes) else 0.0,
            "all_memberships_per_node": float(np.sum(all_sizes) / self.num_nodes),
            "communities_top": int(len(top)),
            "top_mean_size": float(top["size"].mean()) if len(top) else 0.0,
            "top_node_coverage": float(covered.mean()),
            "top_overlapping_nodes": int((memberships > 1).sum()),
            "top_mean_memberships_of_covered": float(memberships[covered].mean())
                                               if covered.any() else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild SNAP com-DBLP from DBLP XML with edge provenance.")
    parser.add_argument("--xml", default="data/dblp.xml",
                        help="DBLP XML; dblp.dtd must be in the same folder")
    parser.add_argument("--out", default="data/com_dblp_ef")
    parser.add_argument("--download", action="store_true",
                        help="download dblp.xml and dblp.dtd first")
    parser.add_argument("--pub-types", nargs="+",
                        default=["article", "inproceedings"])
    parser.add_argument("--include-informal", action="store_true",
                        help="keep informal records such as CoRR preprints")
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--max-authors", type=int, default=50,
                        help="skip papers with more authors (0 = no limit)")
    parser.add_argument("--min-community-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--community-mode", choices=["venue", "component"],
                        default="venue",
                        help="venue: one community per venue (paper's Table I); "
                             "component: split venues into connected components "
                             "(SNAP release)")
    parser.add_argument("--main-volume-only", action="store_true",
                        help="keep only the main track per venue and year "
                             "(drops co-located workshops)")
    parser.add_argument("--no-lcc", action="store_true",
                        help="keep all components instead of the largest")
    parser.add_argument("--limit", type=int,
                        help="stop after this many kept papers (debugging)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    if args.download:
        DBLPStreamParser.download_dblp(args.xml)

    config = vars(args).copy()
    builder = ComDBLPBuilder(
        pub_types=args.pub_types,
        include_informal=args.include_informal,
        min_year=args.min_year,
        max_year=args.max_year,
        max_authors=args.max_authors,
        min_community_size=args.min_community_size,
        top_k=args.top_k,
        largest_component=not args.no_lcc,
        community_mode=args.community_mode,
        main_volume_only=args.main_volume_only,
    )

    t0 = time.time()
    stream = DBLPStreamParser(args.xml, chunk_size=500000,
                              pub_types=args.pub_types).parse()
    builder.collect(stream, limit=args.limit)
    builder.build_graph()
    builder.build_communities()
    stats = builder.save(args.out, config)

    logger.info(f"Done in {time.time() - t0:.1f}s")
    for key, value in stats.items():
        refs = [f"{name}: {table[key]}" for name, table in
                (("paper", PAPER_REFERENCE), ("SNAP", SNAP_REFERENCE))
                if key in table]
        logger.info(f"  {key}: {value}" + (f"  ({', '.join(refs)})" if refs else ""))


if __name__ == "__main__":
    main()
