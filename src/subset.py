"""
Benchmark Subsets

Cuts a smaller benchmark out of a com-DBLP build: pick a set of venues, take
the co-authorship subgraph induced by their authors (largest component by
default), and keep those venues as the ground-truth communities. Edges keep
their paper_ids, which index the parent build's papers.jsonl.

Usage:
    python src/subset.py --dataset data/com_dblp_ef_conf_le2006_venue \
        --preset cs-areas --out data/subsets/cs_areas_le2006
    python src/subset.py --dataset ... --random-k 50 --min-size 200 \
        --max-size 5000 --seed 0 --out data/subsets/rand50_s0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Major conferences across computer-science areas; their author groups
# overlap through people who publish in several areas.
PRESETS = {
    "cs-areas": [
        # data mining
        "conf/kdd", "conf/icdm", "conf/sdm", "conf/pkdd",
        # databases
        "conf/sigmod", "conf/vldb", "conf/icde", "conf/pods", "conf/edbt",
        # information retrieval and web
        "conf/sigir", "conf/cikm", "conf/www",
        # machine learning and AI
        "conf/icml", "conf/nips", "conf/colt", "conf/aaai", "conf/ijcai", "conf/uai",
        # vision
        "conf/cvpr", "conf/iccv", "conf/eccv",
        # natural language processing
        "conf/acl", "conf/emnlp", "conf/coling",
        # theory
        "conf/stoc", "conf/focs", "conf/soda",
    ],
    # Systems areas, no venue in common with cs-areas
    "cs-systems": [
        # networking
        "conf/sigcomm", "conf/infocom", "conf/mobicom", "conf/imc", "conf/icnp",
        # operating and distributed systems
        "conf/sosp", "conf/osdi", "conf/usenix", "conf/nsdi", "conf/podc", "conf/icdcs",
        # architecture
        "conf/isca", "conf/micro", "conf/hpca", "conf/asplos",
        # security
        "conf/ccs", "conf/sp", "conf/uss", "conf/ndss",
        # programming languages
        "conf/pldi", "conf/popl", "conf/oopsla", "conf/icfp",
        # software engineering
        "conf/icse", "conf/sigsoft",
    ],
}


def load_communities(dataset: Path) -> pd.DataFrame:
    meta = pd.read_csv(dataset / "communities.meta.tsv", sep="\t")
    with open(dataset / "communities.all.txt") as f:
        meta["nodes"] = [np.array(line.split("\t"), dtype=np.int64) for line in f]
    return meta


def select_venues(meta: pd.DataFrame, venues: Optional[List[str]],
                  random_k: Optional[int], min_size: int, max_size: Optional[int],
                  seed: int) -> pd.DataFrame:
    if venues:
        chosen = meta[meta["venue_key"].isin(venues)]
        missing = sorted(set(venues) - set(chosen["venue_key"]))
        if missing:
            logger.warning(f"Venues not in the dataset: {missing}")
        return chosen
    pool = meta[meta["size"] >= min_size]
    if max_size:
        pool = pool[pool["size"] <= max_size]
    if random_k > len(pool):
        raise ValueError(f"Only {len(pool)} venues match the size range")
    return pool.sample(n=random_k, random_state=seed)


def build_subset(dataset: str, out: str, venues: Optional[List[str]] = None,
                 random_k: Optional[int] = None, min_size: int = 3,
                 max_size: Optional[int] = None, seed: int = 0,
                 largest_component: bool = True, min_community_size: int = 3):
    dataset, out = Path(dataset), Path(out)
    meta = load_communities(dataset)
    chosen = select_venues(meta, venues, random_k, min_size, max_size, seed)

    nodes = np.unique(np.concatenate(chosen["nodes"].to_list()))
    edges = pd.read_csv(dataset / "edges.tsv", sep="\t",
                        dtype={"paper_ids": str})
    in_subset = np.zeros(int(edges[["u", "v"]].to_numpy().max()) + 1, dtype=bool)
    in_subset[nodes] = True
    edges = edges[in_subset[edges["u"]] & in_subset[edges["v"]]]

    if largest_component:
        remap = np.full(len(in_subset), -1, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        u, v = remap[edges["u"]], remap[edges["v"]]
        adj = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(len(nodes),) * 2)
        _, labels = connected_components(adj, directed=False)
        lcc = np.bincount(labels).argmax()
        nodes = nodes[labels == lcc]
        in_subset[:] = False
        in_subset[nodes] = True
        edges = edges[in_subset[edges["u"]] & in_subset[edges["v"]]]

    remap = np.full(len(in_subset), -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))

    communities, kept = [], []
    for _, row in chosen.iterrows():
        members = remap[row["nodes"]]
        members = np.sort(members[members >= 0])
        if len(members) >= min_community_size:
            communities.append(members)
            kept.append({"venue_key": row["venue_key"],
                         "venue_name": row["venue_name"],
                         "size": len(members)})

    out.mkdir(parents=True, exist_ok=True)
    authors = pd.read_csv(dataset / "authors.tsv", sep="\t", keep_default_na=False)
    pd.DataFrame({
        "node_id": np.arange(len(nodes)),
        "parent_node_id": nodes,
        "name": authors["name"].to_numpy()[nodes],
    }).to_csv(out / "nodes.tsv", sep="\t", index=False)

    pd.DataFrame({
        "u": remap[edges["u"]],
        "v": remap[edges["v"]],
        "parent_edge_id": edges["edge_id"].to_numpy(),
        "num_papers": edges["num_papers"].to_numpy(),
        "paper_ids": edges["paper_ids"].to_numpy(),
    }).to_csv(out / "edges.tsv", sep="\t", index=False)

    with open(out / "communities.txt", "w") as f:
        for members in communities:
            f.write("\t".join(map(str, members)) + "\n")
    pd.DataFrame(kept).to_csv(out / "communities.meta.tsv", sep="\t", index=False)

    memberships = np.zeros(len(nodes), dtype=np.int64)
    for members in communities:
        memberships[members] += 1
    stats = {
        "parent_dataset": str(dataset),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "communities": len(communities),
        "mean_community_size": float(np.mean([len(c) for c in communities])),
        "memberships_per_node": float(memberships.mean()),
        "overlapping_node_fraction": float((memberships > 1).mean()),
        "selection": {"venues": venues, "random_k": random_k, "min_size": min_size,
                      "max_size": max_size, "seed": seed,
                      "largest_component": largest_component},
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Subset saved to {out}: {json.dumps({k: v for k, v in stats.items() if k != 'selection'})}")
    return stats


def load_subset(path: str):
    """Return (num_nodes, edge_index [2, E], communities list) of a subset."""
    path = Path(path)
    edges = pd.read_csv(path / "edges.tsv", sep="\t", usecols=["u", "v"])
    with open(path / "communities.txt") as f:
        communities = [np.array(line.split("\t"), dtype=np.int64) for line in f]
    num_nodes = len(pd.read_csv(path / "nodes.tsv", sep="\t", usecols=["node_id"]))
    return num_nodes, edges[["u", "v"]].to_numpy().T, communities


def main():
    parser = argparse.ArgumentParser(description="Cut a venue subset benchmark.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(PRESETS))
    group.add_argument("--venues", nargs="+", help="venue keys, e.g. conf/kdd")
    group.add_argument("--random-k", type=int)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-lcc", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    venues = PRESETS[args.preset] if args.preset else args.venues
    build_subset(args.dataset, args.out, venues=venues, random_k=args.random_k,
                 min_size=args.min_size, max_size=args.max_size, seed=args.seed,
                 largest_component=not args.no_lcc)


if __name__ == "__main__":
    main()
