"""
Homophily-preserving placebo edge features for a benchmark subset.

Shuffling features across edges destroys the homophily that MarkBP-auto's first
test measures, so it is an easy placebo. This one keeps the homophily but
breaks any link to the ground truth:

1. Cluster the subset's real edge features into K topics (spherical k-means).
2. Give every node a random topic g_u.
3. Give every edge (u, v) a real edge feature drawn from the topic of one of
   its endpoints, chosen at random.

Edges at the same node then share a topic, as real text does, and each feature
is a real text embedding, but the topics are assigned at random, so they carry
no information about the communities.

    python src/placebo_features.py --subset data/subsets/cs_areas_le2006_main \
        --source edge_feats_title
writes <subset>/<source>_placebo.npy and .json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .kmeans_cache import cached_kmeans
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from kmeans_cache import cached_kmeans
    from subset import load_subset


def placebo(feats: np.ndarray, edge_index: np.ndarray, num_nodes: int, k: int,
            seed: int = 0, max_fit: int = 50000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    has = np.linalg.norm(feats, axis=1) > 0
    pool = np.flatnonzero(has)
    sample = np.sort(rng.choice(pool, min(max_fit, len(pool)), replace=False))
    centers, _ = cached_kmeans(feats[sample], k, seed)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    topic = np.argmax(feats[pool] @ centers.T, axis=1)
    members = [pool[topic == t] for t in range(k)]

    groups = rng.integers(0, k, num_nodes)
    u, v = edge_index
    side = rng.integers(0, 2, len(u)).astype(bool)
    chosen = groups[np.where(side, u, v)]
    out = np.empty_like(feats)
    for t in range(k):
        idx = np.flatnonzero(chosen == t)
        if len(members[t]) == 0:  # empty topic: fall back to any real feature
            src = rng.choice(pool, len(idx))
        else:
            src = rng.choice(members[t], len(idx))
        out[idx] = feats[src]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--source", default="edge_feats_title")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = Path(args.subset)
    n, ei, truth = load_subset(path)
    feats = np.load(path / f"{args.source}.npy")
    out = placebo(feats, ei, n, len(truth), args.seed)
    name = f"{args.source}_placebo"
    np.save(path / f"{name}.npy", out.astype(np.float32))
    with open(path / f"{name}.json", "w") as f:
        json.dump({"placebo": "homophily-preserving", "source": args.source,
                   "topics": len(truth), "seed": args.seed, "edges": int(ei.shape[1])},
                  f, indent=2)
    print(f"wrote {path / name}.npy {out.shape}")


if __name__ == "__main__":
    main()
