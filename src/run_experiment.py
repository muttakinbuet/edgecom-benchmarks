"""
Experiment Runner

Runs overlapping community detection methods on a synthetic graph or a
benchmark subset and appends one JSON line per (method, seed) to a results file.

Methods:
    bigclam        structure only (numpy BigCLAM)
    nocd           NOCD-G: GCN encoder on the adjacency, structure only
    nocd_eagg      NOCD with node features = mean of incident edge features
    markbp         NOCD + edge-feature marks (ours)
    markbp_shuffle MarkBP with edge features permuted across edges (placebo)
    markbp_eagg    MarkBP whose encoder also gets the nocd_eagg node features
    markbp_auto    MarkBP with the mark weight chosen by held-out edge likelihood
    <m>_shuffle    any method above with edge features permuted across edges

Usage:
    python src/run_experiment.py --synthetic --snr 0 0.5 1 2 --methods nocd markbp
    python src/run_experiment.py --subset data/subsets/cs_areas_le2006_main \
        --edge-features data/subsets/cs_areas_le2006_main/edge_feats_title.npy
"""

import os

# Single-threaded numerics in every process (also spawned workers): threaded
# BLAS/OpenMP reductions in k-means and torch make results non-deterministic
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

try:
    from .bigclam import BigCLAM
    from .metrics import evaluate, membership_matrix
    from .ocd_models import OverlappingCommunityModel
    from .subset import load_subset
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from bigclam import BigCLAM
    from metrics import evaluate, membership_matrix
    from ocd_models import OverlappingCommunityModel
    from subset import load_subset
    from synthetic import generate

logger = logging.getLogger(__name__)


def edge_feature_node_mean(num_nodes: int, edge_index: np.ndarray,
                           feats: np.ndarray) -> np.ndarray:
    """Node features as the mean of incident edge features."""
    u, v = edge_index
    rows = np.concatenate([u, v])
    cols = np.concatenate([np.arange(len(u)), np.arange(len(u))])
    inc = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(num_nodes, len(u)))
    deg = np.asarray(inc.sum(axis=1)).ravel()
    return (inc @ feats) / np.maximum(deg, 1)[:, None]


def run_method(method, k, num_nodes, edge_index, edge_feats, seed, args):
    # "<method>_shuffle": same method with edge features permuted across edges
    if method.endswith("_shuffle") and method != "markbp_shuffle":
        edge_feats = edge_feats[np.random.default_rng(seed).permutation(len(edge_feats))]
        method = method[:-len("_shuffle")]
    if method == "bigclam":
        model = BigCLAM(k, seed=seed)
        model.fit(num_nodes, edge_index)
        return model, None

    common = dict(epochs=args.epochs, seed=seed, hidden_dims=tuple(args.hidden),
                  init_seed=None if args.restart is None else seed * 1000 + args.restart)
    if method == "nocd":
        model = OverlappingCommunityModel(k, **common)
        model.fit(num_nodes, edge_index)
    elif method == "nocd_eagg":
        model = OverlappingCommunityModel(k, **common)
        model.fit(num_nodes, edge_index, input_edge_features=edge_feats)
    elif method in ("markbp", "markbp_shuffle", "markbp_eagg"):
        feats = edge_feats
        if method == "markbp_shuffle":
            feats = edge_feats[np.random.default_rng(seed).permutation(len(edge_feats))]
        node_inputs = feats if method == "markbp_eagg" else None
        model = OverlappingCommunityModel(k, mark_weight=args.mark_weight,
                                          kappa=args.kappa, mu_init=args.mu_init,
                                          **common)
        model.fit(num_nodes, edge_index, edge_features=feats, input_edge_features=node_inputs)
        return model, feats
    elif method == "markbp_auto":
        # Pick the mark weight without labels: the one whose model best
        # predicts held-out edges (weight 0 is plain NOCD)
        best = None
        for weight in args.auto_weights:
            model = OverlappingCommunityModel(k, mark_weight=weight, kappa=args.kappa,
                                              mu_init=args.mu_init, **common)
            model.fit(num_nodes, edge_index,
                      edge_features=edge_feats if weight > 0 else None)
            logger.info(f"markbp_auto: weight {weight} -> val {model.best_val:.4f}")
            if best is None or model.best_val < best.best_val:
                best = model
        best.chosen_weight = best.mark_weight
        return best, edge_feats if best.mark_weight > 0 else None
    else:
        raise ValueError(method)
    return model, None


_DATA_CACHE = {}


def load_data(name, params, seed, edge_features_path):
    """Graph, ground truth, edge features and (synthetic only) edge labels."""
    key = (name, tuple(sorted(params.items())), seed if name == "synthetic" else None)
    if key not in _DATA_CACHE:
        if name == "synthetic":
            data = generate(seed=seed, **params)
            _DATA_CACHE[key] = (data["num_nodes"], data["edge_index"], data["communities"],
                                data["edge_features"], data["edge_labels"])
        else:
            num_nodes, edge_index, truth = load_subset(name)
            feats = np.load(edge_features_path) if edge_features_path else None
            _DATA_CACHE[key] = (num_nodes, edge_index, truth, feats, None)
    return _DATA_CACHE[key]


def job_key(name, params, seed, method, mark_weight, restart):
    """Identity of a job; also derivable from its result record (for resuming)."""
    weight = mark_weight if method.startswith("markbp") and "auto" not in method else None
    return (name, json.dumps(params, sort_keys=True), seed, method, weight, restart)


def run_job(job):
    """Run one job, returning its record or, on failure, an error description."""
    try:
        return _run_job(job)
    except Exception as e:  # one failing job must not stop the others
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "job": {k: v for k, v in job.items() if k != "args"}}


def _run_job(job):
    """Run one (data, seed, method, mark weight) job single-threaded; return its record."""
    import torch
    torch.set_num_threads(1)  # deterministic sparse ops; parallelism is across jobs
    args = job["args"]
    overrides = {"restart": job.get("restart")}
    if job["mark_weight"] is not None:
        overrides["mark_weight"] = job["mark_weight"]
    args = argparse.Namespace(**{**vars(args), **overrides})
    num_nodes, edge_index, truth, feats, labels = load_data(
        job["name"], job["params"], job["seed"], args.edge_features)
    k = len(truth)

    t0 = time.time()
    model, used_feats = run_method(job["method"], k, num_nodes, edge_index, feats,
                                   job["seed"], args)
    pred = model.communities()
    record = {"data": job["name"], **job["params"], "method": job["method"],
              "seed": job["seed"], "restart": job.get("restart"),
              "tag": args.tag, "mark_weight": args.mark_weight,
              "kappa": args.kappa, "mu_init": args.mu_init, "k": k,
              **evaluate(pred, truth, num_nodes), "predicted": len(pred),
              "best_val": getattr(model, "best_val", None),
              "best_epoch": getattr(model, "best_epoch", None),
              "chosen_weight": getattr(model, "chosen_weight", None),
              "seconds": round(time.time() - t0, 1)}
    if labels is not None and hasattr(model, "edge_assignments"):
        record["edge_acc"] = edge_accuracy(model, truth, edge_index, used_feats,
                                           labels, num_nodes)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run OCD experiments.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--subset")
    parser.add_argument("--edge-features", help=".npy aligned with subset edges.tsv")
    parser.add_argument("--snr", type=float, nargs="+", default=[1.0])
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--methods", nargs="+",
                        default=["bigclam", "nocd", "nocd_eagg", "markbp", "markbp_shuffle"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--mark-weight", type=float, default=0.3)
    parser.add_argument("--mark-weights", type=float, nargs="+",
                        help="run markbp-type methods once per weight (sweep)")
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--auto-weights", type=float, nargs="+",
                        default=[0.0, 0.03, 0.1, 0.3, 1.0],
                        help="mark weights tried by markbp_auto")
    parser.add_argument("--mu-init", choices=["random", "kmeans"], default="kmeans")
    parser.add_argument("--restarts", type=int, default=1,
                        help="nocd runs per seed with different inits on the same "
                             "held-out split (for a best-of-R baseline)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel single-threaded jobs")
    parser.add_argument("--out", default="results/results.jsonl")
    parser.add_argument("--tag", default="", help="free-form label stored with each record")
    args = parser.parse_args()
    args.restart = None

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        settings = [("synthetic", dict(snr=snr, noise=args.noise)) for snr in args.snr]
    else:
        settings = [(args.subset, {})]
    has_feats = args.synthetic or args.edge_features is not None

    jobs = []
    for name, params in settings:
        for seed in args.seeds:
            for method in args.methods:
                if not has_feats and method not in ("bigclam", "nocd"):
                    continue
                sweep = args.mark_weights if (args.mark_weights and
                                              method.startswith("markbp") and
                                              "auto" not in method) else [None]
                restarts = range(args.restarts) if method == "nocd" else [None]
                for weight in sweep:
                    for restart in restarts:
                        jobs.append({"name": name, "params": params, "seed": seed,
                                     "method": method, "mark_weight": weight,
                                     "restart": restart, "args": args})
    if out.exists():
        param_names = sorted(settings[0][1])
        done = set()
        for line in open(out):
            if line.strip():
                r = json.loads(line)
                done.add(job_key(r["data"], {p: r[p] for p in param_names}, r["seed"],
                                 r["method"], r["mark_weight"], r.get("restart")))
        before = len(jobs)
        jobs = [j for j in jobs if job_key(
            j["name"], j["params"], j["seed"], j["method"],
            args.mark_weight if j["mark_weight"] is None else j["mark_weight"],
            j["restart"]) not in done]
        logger.info(f"Resuming: {before - len(jobs)} of {before} jobs already in {out}")
    logger.info(f"{len(jobs)} jobs on {args.workers} worker(s)")

    failures = 0

    def write(record):
        nonlocal failures
        if "error" in record:
            failures += 1
            logger.error(f"job failed: {record['job']} {record['error']}\n"
                         f"{record['traceback']}")
            return
        logger.info(json.dumps(record))
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for record in pool.imap_unordered(run_job, jobs):
                write(record)
    else:
        for job in jobs:
            write(run_job(job))
    if failures:
        logger.error(f"{failures} job(s) failed; rerun the same command to retry them")


def edge_accuracy(model, truth, edge_index, feats, labels, num_nodes) -> float:
    """
    Accuracy of edge-to-community assignment on community-created edges, after
    matching model communities to true ones by member overlap (Hungarian).
    """
    member = model.F > model.threshold
    a = sp.csc_matrix(member.astype(np.float64))
    b = membership_matrix(truth, num_nodes)
    overlap = (a.T @ b).toarray()
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.full(model.k, -2)
    mapping[rows] = cols
    pred = model.edge_assignments(edge_index, feats)
    mask = labels >= 0
    return float(np.mean(mapping[pred[mask]] == labels[mask]))


if __name__ == "__main__":
    main()
