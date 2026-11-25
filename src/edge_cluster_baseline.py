"""
Edge-feature clustering baseline on a benchmark subset: k-means on the edge
features alone (no structure), then node memberships from the edges (a node
joins every cluster holding at least 25% of its edges).

Usage:
    python src/edge_cluster_baseline.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title edge_feats_title_outside --seeds 0 1 2 \
        --out results/dblp_edge_kmeans.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import evaluate
    from .poc import edge_kmeans
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from metrics import evaluate
    from poc import edge_kmeans
    from subset import load_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--features", nargs="+", required=True,
                        help="feature file stems inside the subset folder")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    n, edge_index, truth = load_subset(args.subset)
    with open(args.out, "w") as f:
        for stem in args.features:
            feats = np.load(Path(args.subset) / f"{stem}.npy")
            for seed in args.seeds:
                member, _ = edge_kmeans(n, edge_index, feats, len(truth), seed)
                pred = [np.flatnonzero(member[:, c]) for c in range(member.shape[1])
                        if member[:, c].any()]
                record = {"features": stem, "seed": seed, **evaluate(pred, truth, n)}
                print(record)
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
