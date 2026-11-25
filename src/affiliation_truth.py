"""
Affiliation Ground Truth

A second, content-independent ground truth for a benchmark subset: an author
belongs to every institution that OpenAlex lists for them on at least
--min-papers of their papers (all papers of the parent build, not only those in
the subset's venues), and institutions with at least --min-size members in the
subset become the communities. Overlap comes from moves and joint affiliations.

The output is a new subset folder with the same graph and edge features (hard
links to the source subset's files) and these communities. About half of the
authors have no qualifying institution; eval_nodes.npy lists the authors that
do, and scoring is restricted to them (dblp_markbp.score).

Authors are matched to OpenAlex authorships by position, on papers whose
OpenAlex author list has the same length as DBLP's.

Usage:
    python src/affiliation_truth.py --subset data/subsets/cs_areas_le2006_main \
        --dataset data/com_dblp_ef_conf_le2006_venue_main \
        --out data/subsets/cs_areas_le2006_aff
"""

import argparse
import collections
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def institutions_per_author(dataset: Path, cache: Path, authors: set):
    """{parent node id: Counter(institution id -> papers)} and institution names."""
    authorships = {}
    with open(cache, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "found":
                authorships[r["doi"].lower()] = r.get("authorships") or []
    counts, names = collections.defaultdict(collections.Counter), {}
    matched = total = 0
    with open(dataset / "papers.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if not authors.intersection(p["author_nodes"]):
                continue
            total += 1
            a = authorships.get((p.get("doi") or "").lower())
            if a is None or len(a) != len(p["author_nodes"]):
                continue
            matched += 1
            for node, au in zip(p["author_nodes"], a):
                if node in authors:
                    for inst in au["institutions"]:
                        counts[node][inst["id"]] += 1
                        names[inst["id"]] = inst.get("name")
    logger.info(f"{matched} of {total} papers matched to OpenAlex authorships")
    return counts, names


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--dataset", required=True, help="parent build (papers.jsonl)")
    parser.add_argument("--cache", default="data/openalex_cache.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-papers", type=int, default=1)
    parser.add_argument("--min-size", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    subset, out = Path(args.subset), Path(args.out)
    nodes = pd.read_csv(subset / "nodes.tsv", sep="\t")
    parent = nodes["parent_node_id"].to_numpy()
    counts, names = institutions_per_author(Path(args.dataset), Path(args.cache), set(parent))

    members = collections.defaultdict(list)
    for node, parent_id in enumerate(parent):
        for inst, c in counts.get(parent_id, {}).items():
            if c >= args.min_papers:
                members[inst].append(node)
    kept = sorted((i for i, m in members.items() if len(m) >= args.min_size),
                  key=lambda i: -len(members[i]))

    out.mkdir(parents=True, exist_ok=True)
    for path in subset.iterdir():
        if path.suffix == ".npy" or path.name in ("nodes.tsv", "edges.tsv"):
            if not (out / path.name).exists():
                link(path, out / path.name)
    with open(out / "communities.txt", "w") as f:
        for inst in kept:
            f.write("\t".join(map(str, sorted(members[inst]))) + "\n")
    pd.DataFrame({"institution_id": kept, "name": [names[i] for i in kept],
                  "size": [len(members[i]) for i in kept]}).to_csv(
        out / "communities.meta.tsv", sep="\t", index=False)

    memberships = np.zeros(len(parent), dtype=np.int64)
    for inst in kept:
        memberships[members[inst]] += 1
    np.save(out / "eval_nodes.npy", np.flatnonzero(memberships > 0))
    covered = memberships > 0
    stats = {"source_subset": str(subset), "ground_truth": "OpenAlex institutions",
             "min_papers": args.min_papers, "min_size": args.min_size,
             "nodes": int(len(parent)), "communities": len(kept),
             "evaluated_nodes": int(covered.sum()),
             "memberships_per_evaluated_node": float(memberships[covered].mean()),
             "overlapping_fraction_of_evaluated": float((memberships > 1).sum() / covered.sum()),
             "mean_community_size": float(np.mean([len(members[i]) for i in kept]))}
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(json.dumps(stats))


if __name__ == "__main__":
    main()
