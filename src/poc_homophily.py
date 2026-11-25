"""
Evaluate the edge-feature homophily test (feature_test.homophily_test) on the
proof-of-concept graphs, with real and shuffled features. No model is fitted.

Usage:
    python src/poc_homophily.py --workers 12 --out results/poc_homophily.jsonl
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .feature_test import homophily_test
    from .poc import GRAPH
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from feature_test import homophily_test
    from poc import GRAPH
    from synthetic import generate


def run_job(job):
    data = generate(seed=job["seed"], noise=job["noise"], snr=job["snr"], **GRAPH)
    ei, k = data["edge_index"], len(data["communities"])
    records = []
    for features in ("real", "shuffled"):
        feats = data["edge_features"]
        if features == "shuffled":
            feats = feats[np.random.default_rng(job["seed"] + 12345).permutation(len(feats))]
        records.append({**job, "features": features,
                        **homophily_test(ei, feats, k, job["seed"])})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="results/poc_homophily.jsonl")
    args = parser.parse_args()
    jobs = [{"noise": nz, "snr": s, "seed": seed}
            for nz in args.noise for s in args.snr for seed in range(args.seeds)]
    from multiprocessing import Pool
    with Pool(args.workers) as pool, open(Path(args.out), "w") as f:
        for records in pool.imap_unordered(run_job, jobs):
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
