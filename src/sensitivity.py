"""
Sensitivity of MarkBP-auto (+ input features, union readout) to its settings.

Refitted settings (one at a time, the others at their defaults):
    K       number of communities: K_true/2, K_true, 2 K_true (M = K always:
            the feature-led start needs one feature cluster per community)
    lambda  mark weight: 0.03, 0.1, 0.3
NOCD and NOCD + averaged features are refitted at the same K values, since a
wrong K affects every method.

Readout settings, evaluated on the same fits without retraining:
    threshold  F_uc > t for the threshold readout, t in THRESHOLDS
    tau        minimum share of a node's edges for the edge readout, in TAUS
The union readout is scored over t at tau = 0.25 and over tau at t = 0.5.

Usage:
    python src/sensitivity.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title_abstract --seeds 0 1 2 --workers 7 \
        --out results/sensitivity/cs_areas.jsonl
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

try:
    from .dblp_markbp import eval_nodes, score
    from .markbp_auto import MarkBPAuto
    from .ocd_models import OverlappingCommunityModel
    from .run_experiment import edge_feature_node_mean
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from dblp_markbp import eval_nodes, score
    from markbp_auto import MarkBPAuto
    from ocd_models import OverlappingCommunityModel
    from run_experiment import edge_feature_node_mean
    from subset import load_subset

logger = logging.getLogger(__name__)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
TAUS = [0.1, 0.15, 0.25, 0.35, 0.5]
DEFAULT_T, DEFAULT_TAU = 0.5, 0.25


def edge_share_readout(n, ei, labels, k, tau):
    """Node joins every community holding at least tau of its assigned edges."""
    has = labels >= 0
    u, v = ei[:, has]
    counts = np.zeros((n, k))
    np.add.at(counts, (u, labels[has]), 1)
    np.add.at(counts, (v, labels[has]), 1)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) >= tau


def configs(k_true):
    half, double = max(2, round(k_true / 2)), 2 * k_true
    out = []
    for k in (half, k_true, double):
        out += [{"method": "nocd", "k": k}, {"method": "nocd_eagg", "k": k},
                {"method": "ours", "k": k, "m": None, "lam": 0.1}]
    # M is not a free setting: the feature-led start maps each of the M feature
    # clusters to one community, so it requires M = K (OverlappingCommunityModel
    # raises otherwise); K above therefore varies M with it
    for lam in (0.03, 0.3):
        out.append({"method": "ours", "k": k_true, "m": None, "lam": lam})
    return out


def run_job(job):
    import torch
    torch.set_num_threads(1)
    try:
        n, ei, truth = load_subset(job["subset"])
        feats = np.load(Path(job["subset"]) / f"{job['features']}.npy")
        k, seed, nodes = job["k"], job["seed"], eval_nodes(job["subset"])
        t0 = time.time()
        extra = {}
        if job["method"] == "nocd":
            model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_dblp")
            model.fit(n, ei)  # no features: the (cached) NOCD fit
            F, labels = model.F, model.edge_assignments(ei)
        elif job["method"] == "nocd_eagg":
            model = OverlappingCommunityModel(k, seed=seed, refit=False)
            model.fit(n, ei, input_edge_features=feats)
            F, labels = model.F, model.edge_assignments(ei)
        else:
            model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_dblp",
                               input_features=True, mark_weight=job["lam"],
                               feature_clusters=job["m"])
            model.fit(n, ei, edge_features=feats)
            F, labels = model.F, model.edge_assignments(ei)
            extra = {"choice": model.choice, "test_z": model.test["z"],
                     "homophily_z": model.test["homophily_z"]}
        seconds = round(time.time() - t0, 1)

        thr = {t: F > t for t in THRESHOLDS}
        edg = {tau: edge_share_readout(n, ei, labels, k, tau) for tau in TAUS}
        readouts = [("threshold", t, None, thr[t]) for t in THRESHOLDS]
        readouts += [("edges", None, tau, edg[tau]) for tau in TAUS]
        readouts += [("union", t, DEFAULT_TAU, thr[t] | edg[DEFAULT_TAU]) for t in THRESHOLDS]
        readouts += [("union", DEFAULT_T, tau, thr[DEFAULT_T] | edg[tau])
                     for tau in TAUS if tau != DEFAULT_TAU]
        keys = ("seed", "method", "features", "k", "m", "lam")
        return [{**{key: job.get(key) for key in keys}, "readout": r, "t": t, "tau": tau,
                 **extra, **score(member, truth, n, nodes), "seconds": seconds}
                for r, t, tau, member in readouts]
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "job": job}


def job_key(j):
    return (j["seed"], j["method"], j["features"], j["k"], j.get("m"), j.get("lam"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--features", default="edge_feats_title_abstract")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/sensitivity/cs_areas.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    k_true = len(load_subset(args.subset)[2])
    jobs = [{**c, "seed": s, "features": args.features, "subset": args.subset}
            for s in args.seeds for c in configs(k_true)]
    # NOCD first, so the cached NOCD fits exist when MarkBP-auto needs them
    jobs.sort(key=lambda j: (j["method"] != "nocd", j["seed"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {job_key(r) for r in map(json.loads, open(out)) if r}
        jobs = [j for j in jobs if job_key(j) not in done]
    logger.info(f"K_true = {k_true}; {len(jobs)} jobs on {args.workers} workers")

    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(run_job, jobs):
            if isinstance(result, dict):
                logger.error(f"failed {result['job']}: {result['error']}\n{result['traceback']}")
                continue
            with open(out, "a") as f:
                for record in result:
                    f.write(json.dumps(record) + "\n")
            best = next(r for r in result if r["readout"] == "union" and r["t"] == DEFAULT_T
                        and r["tau"] == DEFAULT_TAU)
            logger.info(json.dumps({key: best.get(key) for key in
                                    ("seed", "method", "k", "m", "lam", "f1", "seconds")}))


if __name__ == "__main__":
    main()
