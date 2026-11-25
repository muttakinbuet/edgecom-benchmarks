"""
Can held-out edges choose between the two starts of the coupled model?

For each synthetic graph, fit MarkBP-coupled from a uniform coupling (the
structure-led start) and from the edge-feature clusters (the feature-led start),
with real or shuffled edge features, and pick the start with the lower held-out
edge loss. No labels are used for the choice; ONMI only scores it.

Usage:
    python src/poc_selection.py --workers 12 --out results/poc_selection.jsonl
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import evaluate
    from .ocd_models import OverlappingCommunityModel
    from .poc import GRAPH, edge_readout, node_scores
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from metrics import evaluate
    from ocd_models import OverlappingCommunityModel
    from poc import GRAPH, edge_readout, node_scores
    from synthetic import generate


def run_job(job):
    import torch
    torch.set_num_threads(1)
    data = generate(seed=job["seed"], noise=job["noise"], snr=job["snr"], **GRAPH)
    n, ei, truth = data["num_nodes"], data["edge_index"], data["communities"]
    feats = data["edge_features"]
    if job["features"] == "shuffled":
        feats = feats[np.random.default_rng(job["seed"] + 12345).permutation(len(feats))]
    k = len(truth)
    record = dict(job)
    for start in ("uniform", "features"):
        model = OverlappingCommunityModel(k, seed=job["seed"], refit=False, mark_weight=1.0,
                                          mark_model="coupled", coupled_init=start)
        model.fit(n, ei, edge_features=feats)
        member = edge_readout(n, ei, model.edge_assignments(ei, feats), k)
        pred = [np.flatnonzero(member[:, c]) for c in range(k) if member[:, c].any()]
        record[f"val_{start}"] = model.best_val
        record[f"onmi_{start}"] = evaluate(pred, truth, n)["onmi"]
        record[f"overlap_exact_{start}"] = node_scores(member, truth, n)["overlap_exact"]
    picked = min(("uniform", "features"), key=lambda s: record[f"val_{s}"])
    record["picked"] = picked
    record["onmi_picked"] = record[f"onmi_{picked}"]
    record["overlap_exact_picked"] = record[f"overlap_exact_{picked}"]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="results/poc_selection.jsonl")
    args = parser.parse_args()

    jobs = [{"noise": nz, "snr": s, "seed": seed, "features": f}
            for nz in args.noise for s in args.snr for seed in range(args.seeds)
            for f in ("real", "shuffled")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    with Pool(args.workers) as pool, open(out, "w") as f:
        for record in pool.imap_unordered(run_job, jobs):
            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
