"""
Yeast Protein Complexes with STRING Evidence Channels

A second benchmark in the subset format of dblp_markbp.py: nodes are yeast
proteins, edges are STRING v12 functional links, each edge's features are its
STRING evidence-channel scores, and the communities are the curated protein
complexes of the Complex Portal (a protein can be in several complexes).

Leakage control: STRING's "database" channels import curated pathways and
complexes, so they are dropped both from the edge features and from the score
that defines the graph. The graph keeps pairs whose combined score, recomputed
from the remaining channels with STRING's prior correction, is at least
--min-score (default 700, STRING's "high confidence"), and its largest
connected component.

Feature sets (one row per edge of edges.tsv, scores divided by 1000):
    edge_feats_evidence         the 10 non-database channels
    edge_feats_evidence_notext  the same without text mining, which reads the
                                literature that also curates the complexes

Complexes are the "expanded participant list" of Complex Portal (subcomplexes
replaced by their members), mapped to STRING through its UniProt aliases; only
members in the graph are kept, then complexes with at least --min-size members,
duplicates removed. eval_nodes.npy lists the proteins in at least one complex;
scoring is restricted to them, as for the affiliation ground truth.

Usage:
    python src/yeast.py --raw data/raw/yeast --out data/subsets/yeast_complexes
"""

import argparse
import collections
import gzip
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

PRIOR = 0.041  # STRING's prior probability of a link
CHANNELS = ["neighborhood", "neighborhood_transferred", "fusion", "cooccurence",
            "coexpression", "coexpression_transferred", "experiments",
            "experiments_transferred", "textmining", "textmining_transferred"]
# fusion/cooccurence have no transferred column in v12; homology is only a
# correction inside STRING and is left out, as are the database channels.
TEXT = ["textmining", "textmining_transferred"]


def combined_score(scores: np.ndarray) -> np.ndarray:
    """STRING's combination of channel scores in [0, 1] (prior-corrected)."""
    s = np.clip((scores - PRIOR) / (1 - PRIOR), 0, 1)
    total = 1 - np.prod(1 - s, axis=1)
    return total + PRIOR * (1 - total)


def load_links(raw: Path, min_score: float):
    links = pd.read_csv(raw / "4932.protein.links.full.v12.0.txt.gz", sep=" ")
    links = links[links.protein1 < links.protein2]  # each pair is listed twice
    scores = links[CHANNELS].to_numpy(dtype=np.float64) / 1000
    keep = combined_score(scores) >= min_score
    logger.info(f"{keep.sum()} of {len(links)} links with score >= {min_score} "
                "without the database channels")
    return links[keep].reset_index(drop=True), scores[keep]


def uniprot_to_string(raw: Path):
    aliases = pd.read_csv(raw / "4932.protein.aliases.v12.0.txt.gz", sep="\t",
                          names=["string_id", "alias", "source"], skiprows=1)
    aliases = aliases[aliases.source.isin(["UniProt_AC", "Ensembl_UniProt"])]
    return dict(zip(aliases.alias, aliases.string_id))


def load_complexes(raw: Path, mapping: dict):
    table = pd.read_csv(raw / "559292.tsv", sep="\t")
    complexes = []
    for _, row in table.iterrows():
        ids = re.findall(r"([A-Z0-9]+)(?:-PRO_\d+)?\(\d+\)",
                         str(row["Expanded participant list"]))
        members = {mapping[i] for i in ids if i in mapping}
        complexes.append((row["#Complex ac"], row["Recommended name"], members))
    return complexes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/yeast")
    parser.add_argument("--out", default="data/subsets/yeast_complexes")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--min-size", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    raw, out = Path(args.raw), Path(args.out)

    links, scores = load_links(raw, args.min_score)
    proteins = np.unique(np.concatenate([links.protein1, links.protein2]))
    index = {p: i for i, p in enumerate(proteins)}
    u = links.protein1.map(index).to_numpy()
    v = links.protein2.map(index).to_numpy()
    adj = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(len(proteins),) * 2)
    _, comp = connected_components(adj, directed=False)
    largest = np.argmax(np.bincount(comp))
    keep_edges = comp[u] == largest
    nodes = proteins[comp == largest]
    node_id = {p: i for i, p in enumerate(nodes)}
    links, scores = links[keep_edges].reset_index(drop=True), scores[keep_edges]
    u = links.protein1.map(node_id).to_numpy()
    v = links.protein2.map(node_id).to_numpy()

    complexes, seen = [], set()
    for ac, name, members in load_complexes(raw, uniprot_to_string(raw)):
        ids = tuple(sorted(node_id[m] for m in members if m in node_id))
        if len(ids) >= args.min_size and ids not in seen:
            seen.add(ids)
            complexes.append((ac, name, ids))
    complexes.sort(key=lambda c: -len(c[2]))

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": np.arange(len(nodes)), "parent_node_id": nodes,
                  "name": [n.split(".", 1)[1] for n in nodes]}).to_csv(
        out / "nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"u": u, "v": v}).to_csv(out / "edges.tsv", sep="\t", index=False)
    text = [CHANNELS.index(c) for c in TEXT]
    feature_sets = {"edge_feats_evidence": scores,
                    "edge_feats_evidence_notext": np.delete(scores, text, axis=1)}
    for name, x in feature_sets.items():
        np.save(out / f"{name}.npy", x.astype(np.float32))
        columns = [c for i, c in enumerate(CHANNELS)
                   if name == "edge_feats_evidence" or i not in text]
        with open(out / f"{name}.json", "w") as f:
            json.dump({"channels": columns, "source": "STRING v12 links.full / 1000"}, f)
    with open(out / "communities.txt", "w") as f:
        for _, _, ids in complexes:
            f.write("\t".join(map(str, ids)) + "\n")
    pd.DataFrame({"complex_ac": [c[0] for c in complexes],
                  "name": [c[1] for c in complexes],
                  "size": [len(c[2]) for c in complexes]}).to_csv(
        out / "communities.meta.tsv", sep="\t", index=False)

    memberships = np.zeros(len(nodes), dtype=np.int64)
    for _, _, ids in complexes:
        memberships[list(ids)] += 1
    covered = memberships > 0
    np.save(out / "eval_nodes.npy", np.flatnonzero(covered))
    sizes = [len(c[2]) for c in complexes]
    stats = {"source": "STRING v12 (4932) + Complex Portal (559292)",
             "min_score": args.min_score, "min_size": args.min_size,
             "nodes": int(len(nodes)), "edges": int(len(u)),
             "communities": len(complexes),
             "evaluated_nodes": int(covered.sum()),
             "memberships_per_evaluated_node": float(memberships[covered].mean()),
             "overlapping_fraction_of_evaluated": float((memberships > 1).sum() / covered.sum()),
             "mean_community_size": float(np.mean(sizes)),
             "median_community_size": float(np.median(sizes))}
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(json.dumps(stats))


if __name__ == "__main__":
    main()
