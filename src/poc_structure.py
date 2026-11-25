"""
Measure the structure gain (structure_test.py) on the proof-of-concept graphs.

Graph structure does not depend on the feature signal s, so one graph per
(background level, seed) is enough. The K-community NOCD fit comes from the
cache written by MarkBP-auto runs.

Usage:
    python src/poc_structure.py --workers 12 --out results/poc_structure.jsonl
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

try:
    from .markbp_auto import MarkBPAuto
    from .poc import GRAPH
    from .structure_test import structure_gain
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from markbp_auto import MarkBPAuto
    from poc import GRAPH
    from structure_test import structure_gain
    from synthetic import generate


def run_job(job):
    import torch
    torch.set_num_threads(1)
    data = generate(seed=job["seed"], noise=job["noise"], snr=0.0, **GRAPH)
    n, ei, k = data["num_nodes"], data["edge_index"], len(data["communities"])
    auto = MarkBPAuto(k, seed=job["seed"], cache_dir="data/cache/nocd_poc")
    nocd = auto._structure_model(n, ei)
    return {**job, **structure_gain(n, ei, nocd, job["seed"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="results/poc_structure.jsonl")
    args = parser.parse_args()
    jobs = [{"noise": nz, "seed": s} for nz in args.noise for s in range(args.seeds)]
    out = Path(args.out)
    from multiprocessing import Pool
    with Pool(args.workers) as pool, open(out, "w") as f:
        for record in pool.imap_unordered(run_job, jobs):
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
