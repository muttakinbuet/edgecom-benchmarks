"""
Proof-of-Concept Experiments

Small synthetic graphs, many seeds, fixed pre-declared hyperparameters and no
model selection, so that each experiment isolates one claim:

  grid      (PoC 1-3) structure strength x feature signal, comparing
            structure only, per-node use of edge features, edge features
            alone, MarkBP, and MarkBP with shuffled / noise / oracle features.

Every model is read out twice: by thresholding its affiliations (F > 0.5, as in
NOCD) and from its edge assignments (a node joins each community holding at
least EDGE_SHARE of its edges). Node-level metrics compare each node's detected
set of communities with its
true set, after matching detected to true communities by member overlap,
separately for overlapping nodes (2+ true communities) and single nodes.

Usage:
    python src/poc.py grid --workers 12 --out results/poc_grid.jsonl
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
    from .cesna import CESNA, binary_attributes
    from .kmeans_cache import cached_kmeans
    from .link_communities import link_communities
    from .markbp_auto import MarkBPAuto
    from .metrics import evaluate, membership_matrix
    from .ocd_models import OverlappingCommunityModel
    from .run_experiment import edge_feature_node_mean
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from cesna import CESNA, binary_attributes
    from kmeans_cache import cached_kmeans
    from link_communities import link_communities
    from markbp_auto import MarkBPAuto
    from metrics import evaluate, membership_matrix
    from ocd_models import OverlappingCommunityModel
    from run_experiment import edge_feature_node_mean
    from synthetic import generate

logger = logging.getLogger(__name__)

# Fixed before running; no tuning on these graphs
GRAPH = dict(num_nodes=400, num_communities=4, degree=8.0, dim=32,
             membership_probs=(0.5, 0.35, 0.15))
MARK_WEIGHT = 0.3
KAPPA = 10.0
EDGE_SHARE = 0.25   # edge-feature clustering: node joins c if >= 25% of its edges are in c
METHODS = ["nocd", "nocd_eagg", "edge_kmeans", "markbp", "markbp_shuffle",
           "markbp_noise", "markbp_oracle"]
# MarkBP v2 (learned per-community concentrations) and its controls; the
# properly normalised likelihood needs no tuned weight
V2_METHODS = ["markbp2", "markbp2_shuffle", "markbp2_noise", "markbp2_oracle"]
# MarkBP v3: features coupled to communities through learned cluster coupling
V3_METHODS = ["markbp3", "markbp3_shuffle", "markbp3_noise", "markbp3_oracle"]
# MarkBP v4: v3 started from the edge-feature clusters
V4_METHODS = ["markbp4", "markbp4_shuffle", "markbp4_noise", "markbp4_oracle"]
# MarkBP-auto: alignment test, then MarkBP v4 or NOCD (the complete method)
AUTO_METHODS = ["markbp_auto", "markbp_auto_shuffle", "markbp_auto_noise"]
# MarkBP-auto whose MarkBP branch also takes node-averaged edge features as input
AUTOIN_METHODS = ["markbp_autoin", "markbp_autoin_shuffle", "markbp_autoin_noise"]
# Published baselines: edge-aware GNN encoder (GINE) and graph attention (GAT)
# inside NOCD, CESNA (attributes from edge features), link communities
BASELINES = ["nocd_gine", "nocd_gat", "cesna", "link_comm", "dynares", "dynares_eagg"]


def match(pred_member: np.ndarray, truth, num_nodes: int) -> np.ndarray:
    """Map predicted community columns to true communities (Hungarian on overlap)."""
    b = membership_matrix(truth, num_nodes)
    overlap = (sp.csc_matrix(pred_member.astype(np.float64)).T @ b).toarray()
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.full(pred_member.shape[1], -1)
    mapping[rows] = cols
    return mapping


def node_scores(pred_member: np.ndarray, truth, num_nodes: int) -> dict:
    """Exact-set accuracy and set Jaccard per node, split by true overlap."""
    k = len(truth)
    true_member = np.zeros((num_nodes, k), dtype=bool)
    for c, nodes in enumerate(truth):
        true_member[nodes, c] = True
    mapping = match(pred_member, truth, num_nodes)
    mapped = np.zeros((num_nodes, k), dtype=bool)
    for col, c in enumerate(mapping):
        if c >= 0:
            mapped[:, c] |= pred_member[:, col]
    inter = (mapped & true_member).sum(axis=1)
    union = (mapped | true_member).sum(axis=1)
    exact = np.all(mapped == true_member, axis=1)
    jacc = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    overlapping = true_member.sum(axis=1) >= 2
    single = true_member.sum(axis=1) == 1
    return {"overlap_exact": float(exact[overlapping].mean()),
            "overlap_jaccard": float(jacc[overlapping].mean()),
            "single_exact": float(exact[single].mean()),
            "single_jaccard": float(jacc[single].mean()),
            "overlap_fraction": float(overlapping.mean())}


def edge_accuracy(pred_member, edge_pred, truth, labels, num_nodes) -> float:
    mapping = match(pred_member, truth, num_nodes)
    mask = labels >= 0
    return float(np.mean(mapping[edge_pred[mask]] == labels[mask]))


def edge_readout(num_nodes, edge_index, edge_labels, k):
    """Node memberships from edge assignments: a node joins every community
    that holds at least EDGE_SHARE of its edges."""
    has = edge_labels >= 0  # edges without an assignment (-1) are skipped
    u, v = edge_index[:, has]
    counts = np.zeros((num_nodes, k))
    np.add.at(counts, (u, edge_labels[has]), 1)
    np.add.at(counts, (v, edge_labels[has]), 1)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return share >= EDGE_SHARE


def edge_kmeans(num_nodes, edge_index, feats, k, seed):
    """Cluster edges by their features alone, then read out node memberships."""
    _, labels = cached_kmeans(feats, k, seed)
    return edge_readout(num_nodes, edge_index, labels, k), labels


def feature_variant(method, data, seed):
    feats, labels = data["edge_features"], data["edge_labels"]
    rng = np.random.default_rng(seed + 12345)
    for version in ("markbp2", "markbp3", "markbp4"):
        method = method.replace(version, "markbp")
    if method == "markbp_shuffle":
        return feats[rng.permutation(len(feats))]
    if method == "markbp_noise":
        noise = rng.normal(size=feats.shape)
        return (noise / np.linalg.norm(noise, axis=1, keepdims=True)).astype(np.float32)
    if method == "markbp_oracle":
        # The true creating community as a one-hot vector; background edges get 0
        oracle = np.zeros((len(labels), len(data["communities"])), dtype=np.float32)
        oracle[labels >= 0, labels[labels >= 0]] = 1.0
        return oracle
    return feats


def run_job(job):
    import torch
    torch.set_num_threads(1)
    try:
        graph = {**GRAPH, "degree": job.get("degree", GRAPH["degree"])}
        data = generate(seed=job["seed"], noise=job["noise"], snr=job["snr"],
                        align=job.get("align", 1.0), **graph)
        n, ei, truth = data["num_nodes"], data["edge_index"], data["communities"]
        k, method, seed = len(truth), job["method"], job["seed"]
        t0 = time.time()
        readouts = {}
        extra = {}
        if method == "edge_kmeans":
            member, edge_pred = edge_kmeans(n, ei, data["edge_features"], k, seed)
            readouts["edges"] = member
        elif method == "cesna":
            attrs = binary_attributes(n, ei, data["edge_features"], 32)
            model = CESNA(k, seed=seed)
            model.fit(n, ei, attrs)
            readouts["threshold"] = model.F > model.threshold
            edge_pred = np.argmax(model.F[ei[0]] * model.F[ei[1]], axis=1)
            readouts["edges"] = edge_readout(n, ei, edge_pred, k)
        elif method == "link_comm":
            comms, edge_pred, _ = link_communities(n, ei)
            member = np.zeros((n, max(len(comms), 1)), dtype=bool)
            for c, nodes in enumerate(comms):
                member[nodes, c] = True
            readouts["native"] = member
            edge_pred = np.zeros(ei.shape[1], dtype=np.int64)  # no K-community labels
        elif method in ("dynares", "dynares_eagg"):
            # Official DynaResGCN encoder: -G (identity input) or -X with
            # node-averaged edge features as the node attributes
            model = OverlappingCommunityModel(k, seed=seed, refit=False, encoder="dynares")
            model.fit(n, ei, input_edge_features=data["edge_features"]
                      if method == "dynares_eagg" else None)
            edge_pred = model.edge_assignments(ei)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, edge_pred, k)
        elif method in ("nocd_gine", "nocd_gat"):
            enc = method.split("_")[1]
            feats = data["edge_features"] if enc == "gine" else None
            model = OverlappingCommunityModel(k, seed=seed, refit=False, encoder=enc)
            model.fit(n, ei, edge_features=feats)
            edge_pred = model.edge_assignments(ei)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, edge_pred, k)
        elif method.startswith("markbp_auto"):
            # Complete method: alignment test, then feature-led MarkBP or NOCD
            inputs = method.startswith("markbp_autoin")
            feats = feature_variant(method.replace("markbp_autoin", "markbp")
                                    .replace("markbp_auto", "markbp"), data, seed)
            model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_poc",
                               input_features=inputs,
                               mark_weight=job.get("mark_weight", 0.1))
            model.fit(n, ei, edge_features=feats)
            edge_pred = model.edge_assignments(ei)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, edge_pred, k)
            extra = {"test_z": model.test["z"], "homophily_z": model.test["homophily_z"],
                     "choice": model.choice, "uses_features": model.uses_features}
        else:
            feats = None
            model = OverlappingCommunityModel(k, seed=seed, refit=False)
            if method == "nocd":
                model.fit(n, ei)
            elif method == "nocd_eagg":
                model.fit(n, ei, input_edge_features=data["edge_features"])
            else:
                feats = feature_variant(method, data, seed)
                v2 = method.startswith("markbp2")
                v3, v4 = method.startswith("markbp3"), method.startswith("markbp4")
                model = OverlappingCommunityModel(
                    k, seed=seed, refit=False,
                    mark_weight=1.0 if (v2 or v3 or v4) else MARK_WEIGHT, kappa=KAPPA,
                    mu_init="kmeans", learn_kappa=v2,
                    mark_model="coupled" if (v3 or v4) else "vmf",
                    coupled_init="features" if v4 else "uniform")
                model.fit(n, ei, edge_features=feats)
            edge_pred = model.edge_assignments(ei, feats)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, edge_pred, k)
        if "threshold" in readouts and "edges" in readouts:
            # A node joins c when either readout puts it there
            readouts["union"] = readouts["threshold"] | readouts["edges"]
        records = []
        for readout, member in readouts.items():
            pred = [np.flatnonzero(member[:, c]) for c in range(member.shape[1])
                    if member[:, c].any()]
            records.append({
                **{k_: job[k_] for k_ in ("noise", "snr", "seed", "method")},
                "degree": job.get("degree", GRAPH["degree"]),
                "align": job.get("align", 1.0),
                **({"mark_weight": job["mark_weight"]} if "mark_weight" in job else {}),
                "readout": readout, **extra, **evaluate(pred, truth, n),
                "kappa_mean": (float(model.concentrations().mean())
                               if method.startswith("markbp2") else None),
                **node_scores(member, truth, n),
                # link communities have no K-community label per edge
                "edge_acc": (None if method == "link_comm" else
                             edge_accuracy(member, edge_pred, truth,
                                           data["edge_labels"], n)),
                "seconds": round(time.time() - t0, 1)})
        return records
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "job": job}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["grid"])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--degree", type=float, nargs="+", default=[GRAPH["degree"]],
                        help="expected within-community degree (structure strength)")
    parser.add_argument("--align", type=float, nargs="+", default=[1.0],
                        help="share of the edge signal about the true community (the dial)")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--mark-weight", type=float, default=None,
                        help="mark weight of MarkBP-auto (default 0.1); use a separate --out")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="results/poc_grid.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r.get("degree", GRAPH["degree"]), r.get("align", 1.0), r["noise"],
                      r["snr"], r["seed"], r["method"]))
    jobs = [{"degree": d, "align": a, "noise": nz, "snr": s, "seed": seed, "method": m}
            for d in args.degree for a in args.align for nz in args.noise for s in args.snr
            for seed in range(args.seeds) for m in args.methods
            if (d, a, nz, s, seed, m) not in done]
    if args.mark_weight is not None:
        for job in jobs:
            job["mark_weight"] = args.mark_weight
    logger.info(f"{len(jobs)} jobs ({len(done)} already done) on {args.workers} workers")

    def write(result):
        if isinstance(result, dict) and "error" in result:
            logger.error(f"failed {result['job']}: {result['error']}\n"
                         f"{result['traceback']}")
            return
        with open(out, "a") as f:
            for rec in result:
                f.write(json.dumps(rec) + "\n")

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_job, jobs), 1):
                write(rec)
                if i % 100 == 0:
                    logger.info(f"{i}/{len(jobs)} done")
    else:
        for job in jobs:
            write(run_job(job))


if __name__ == "__main__":
    main()
