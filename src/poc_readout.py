"""
Readout selection on the controlled graphs.

For MarkBP-auto (with and without input features), score every candidate
readout (threshold, edge shares tau in readout.TAUS) and the label-free choice
of readout.select against the ground truth.

Usage:
    python src/poc_readout.py --workers 4 --out results/poc_readout.jsonl
"""

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from .markbp_auto import MarkBPAuto
    from .metrics import evaluate
    from .poc import GRAPH, node_scores
    from .readout import select
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from markbp_auto import MarkBPAuto
    from metrics import evaluate
    from poc import GRAPH, node_scores
    from readout import select
    from synthetic import generate

logger = logging.getLogger(__name__)


def run_job(job):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    data = generate(seed=job["seed"], noise=job["noise"], snr=job["snr"], **GRAPH)
    n, ei, truth = data["num_nodes"], data["edge_index"], data["communities"]
    k, seed = len(truth), job["seed"]
    model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_poc",
                       input_features=job["method"] == "markbp_autoin")
    model.fit(n, ei, edge_features=data["edge_features"])
    labels = model.edge_assignments(ei)
    # Keep the fit, so further readout rules can be scored without refitting
    fits = Path("data/cache/readout_poc")
    fits.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fits / "{method}_n{noise}_s{snr}_seed{seed}.npz".format(**job),
                        F=model.F, threshold=model.threshold, labels=labels)
    best, _, scores, cands = select(model.F, model.threshold, n, ei, labels, k, seed)
    records = []
    for name, member in cands.items():
        pred = [np.flatnonzero(member[:, c]) for c in range(member.shape[1])
                if member[:, c].any()]
        records.append({**job, "readout": name, "selected": name == best,
                        "agm_score": scores.get(name), "choice": model.choice,
                        **evaluate(pred, truth, n), **node_scores(member, truth, n),
                        "seconds": round(time.time() - t0, 1)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=["markbp_auto", "markbp_autoin"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/poc_readout.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["noise"], r["snr"], r["seed"], r["method"])
                for r in map(json.loads, open(out))}
    jobs = [{"noise": nz, "snr": s, "seed": seed, "method": m}
            for nz in args.noise for s in args.snr for seed in range(args.seeds)
            for m in args.methods if (nz, s, seed, m) not in done]
    logger.info(f"{len(jobs)} jobs on {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            with open(out, "a") as fh:
                for rec in f.result():
                    fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                logger.info(f"{i}/{len(jobs)} done")


if __name__ == "__main__":
    main()
