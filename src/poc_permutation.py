"""
Evaluate the label-free feature-alignment test on the proof-of-concept graphs.

For each graph (and real or shuffled features) fit structure-only NOCD, run the
permutation test of feature_test.py, and record the decision. The outcomes of
both MarkBP starts on the same graphs come from poc_selection.py, so the rule
"feature-led start if p < alpha, else structure-led" can be scored directly.

Usage:
    python src/poc_permutation.py --workers 12 --out results/poc_permutation.jsonl
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .feature_test import alignment_test
    from .ocd_models import OverlappingCommunityModel
    from .poc import GRAPH
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from feature_test import alignment_test
    from ocd_models import OverlappingCommunityModel
    from poc import GRAPH
    from synthetic import generate


def run_job(job):
    import torch
    torch.set_num_threads(1)
    data = generate(seed=job["seed"], noise=job["noise"], snr=job["snr"], **GRAPH)
    n, ei, k = data["num_nodes"], data["edge_index"], len(data["communities"])
    model = OverlappingCommunityModel(k, seed=job["seed"], refit=False)
    model.fit(n, ei)
    structure_labels = model.edge_assignments(ei)
    records = []
    for features in ("real", "shuffled"):
        feats = data["edge_features"]
        if features == "shuffled":
            feats = feats[np.random.default_rng(job["seed"] + 12345).permutation(len(feats))]
        records.append({**job, "features": features,
                        **alignment_test(structure_labels, feats, k, job["seed"])})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="results/poc_permutation.jsonl")
    args = parser.parse_args()

    jobs = [{"noise": nz, "snr": s, "seed": seed}
            for nz in args.noise for s in args.snr for seed in range(args.seeds)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    with Pool(args.workers) as pool, open(out, "w") as f:
        for records in pool.imap_unordered(run_job, jobs):
            for record in records:
                f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
