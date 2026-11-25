"""
DBLP-EF experiments with the current method.

For each seed and edge-feature set of a benchmark subset, runs:
    nocd                      structure only (threshold and edge readouts)
    edge_kmeans               edge features only (edge readout)
    markbp_struct             MarkBP, coupled marks, structure-led start
    markbp_feat               MarkBP, coupled marks, feature-led start
    markbp_feat_shuffle       markbp_feat with features permuted across edges
    markbp_auto               MarkBP-auto: alignment test, then markbp_feat or NOCD
    markbp_auto_shuffle       markbp_auto with features permuted across edges
    markbp_autoin             markbp_auto; MarkBP encoder also gets node-averaged features
    markbp_autoin_shuffle     markbp_autoin with features permuted across edges
    markbp_autoin_check       markbp_autoin with the optional held-out check
    nocd_gine / nocd_gat      NOCD with an edge-aware (GINE) / attention (GAT) encoder
    nocd_eagg                 NOCD with node features = mean incident edge features
    dynares                   official DynaResGCN encoder (-G, identity input)
    dynares_eagg              DynaResGCN-X with node-averaged edge features
    cesna                     CESNA with binary attributes from edge features
    link_comm                 link communities (Ahn et al., 2010)
Models with F record three readouts: threshold, edges and their union.
Training: 10% held-out edges for early stopping, no refit, no model selection.
Results are appended to a JSON-lines file; finished jobs are skipped on rerun.

Usage:
    python src/dblp_markbp.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title edge_feats_title_abstract \
        --shuffle-features edge_feats_title_abstract --seeds 0 1 2 --workers 4 \
        --out results/dblp_markbp.jsonl
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
    from .cesna import CESNA, binary_attributes
    from .link_communities import link_communities
    from .markbp_auto import MarkBPAuto
    from .metrics import evaluate
    from .run_experiment import edge_feature_node_mean
    from .ocd_models import OverlappingCommunityModel
    from .poc import edge_kmeans, edge_readout
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from cesna import CESNA, binary_attributes
    from link_communities import link_communities
    from markbp_auto import MarkBPAuto
    from metrics import evaluate
    from run_experiment import edge_feature_node_mean
    from ocd_models import OverlappingCommunityModel
    from poc import edge_kmeans, edge_readout
    from subset import load_subset

logger = logging.getLogger(__name__)

# Methods that do not use edge features: run once per seed
STRUCTURE_ONLY = {"nocd", "nocd_gat", "link_comm", "dynares"}


def eval_nodes(subset):
    """Nodes to score, for ground truths that label only part of the graph."""
    path = Path(subset) / "eval_nodes.npy"
    return np.load(path) if path.exists() else None


def score(member, truth, n, nodes=None):
    if nodes is not None:
        # Score on the labelled nodes only, renumbered 0..len(nodes)-1
        index = np.full(n, -1)
        index[nodes] = np.arange(len(nodes))
        member, n = member[nodes], len(nodes)
        truth = [index[c] for c in truth]
    pred = [np.flatnonzero(member[:, c]) for c in range(member.shape[1]) if member[:, c].any()]
    return {**evaluate(pred, truth, n), "predicted": len(pred)}


def run_job(job):
    import torch
    torch.set_num_threads(1)
    try:
        n, ei, truth = load_subset(job["subset"])
        k, seed, method = len(truth), job["seed"], job["method"]
        feats = (np.load(Path(job["subset"]) / f"{job['features']}.npy")
                 if job["features"] else None)
        if method.endswith("_shuffle"):
            feats = feats[np.random.default_rng(seed + 12345).permutation(len(feats))]
        t0 = time.time()
        readouts = {}
        extra = {}
        # Restart r > 0 reinitialises the fitted model (weights, batches, mark
        # clusters) with seed * 1000 + r; the held-out split stays the seed's
        restart = job.get("restart") or 0
        init_seed = seed * 1000 + restart if restart else None
        if restart:
            extra["restart"] = restart
        if method == "edge_kmeans":
            readouts["edges"], _ = edge_kmeans(n, ei, feats, k, seed)
        elif method == "cesna":
            model = CESNA(k, seed=seed)
            model.fit(n, ei, binary_attributes(n, ei, feats, 64))
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(
                n, ei, np.argmax(model.F[ei[0]] * model.F[ei[1]], axis=1), k)
        elif method == "link_comm":
            comms, _, _ = link_communities(n, ei)
            member = np.zeros((n, max(len(comms), 1)), dtype=bool)
            for c, nodes in enumerate(comms):
                member[nodes, c] = True
            readouts["native"] = member
            extra = {"link_communities": len(comms)}
        elif method.removesuffix("_shuffle") in ("nocd_gine", "nocd_gat", "nocd_eagg",
                                                 "dynares", "dynares_eagg"):
            enc = {"nocd_gine": "gine", "nocd_gat": "gat", "nocd_eagg": "gcn",
                   "dynares": "dynares", "dynares_eagg": "dynares"}[method.removesuffix("_shuffle")]
            eagg = method.removesuffix("_shuffle") in ("nocd_eagg", "dynares_eagg")
            model = OverlappingCommunityModel(k, seed=seed, refit=False, encoder=enc,
                                              init_seed=init_seed)
            node_feats = edge_feature_node_mean(n, ei, feats) if eagg else None
            model.fit(n, ei, node_features=node_feats,
                      edge_features=feats if enc == "gine" else None)
            extra["val"] = float(model.best_val)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, model.edge_assignments(ei), k)
        elif method.startswith("markbp_auto"):
            # EDGECOM_NOCD_CACHE=none disables the NOCD cache (for timing runs)
            cache = os.environ.get("EDGECOM_NOCD_CACHE", "data/cache/nocd_dblp")
            model = MarkBPAuto(k, seed=seed, cache_dir=None if cache == "none" else cache,
                               input_features=method.startswith("markbp_autoin"),
                               holdout_check=method.endswith("_check"),
                               joint_init_seed=init_seed)
            model.fit(n, ei, edge_features=feats)
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, model.edge_assignments(ei), k)
            extra |= {"test_z": model.test["z"], "homophily_z": model.test["homophily_z"],
                     "joint_val": model.test["joint_val"], "holdout_z": model.test["holdout_z"],
                     "structure_val": model.test["structure_val"],
                     "choice": model.choice, "uses_features": model.uses_features}
        else:
            if method == "nocd":
                model = OverlappingCommunityModel(k, seed=seed, refit=False)
                model.fit(n, ei)
                used = None
            else:
                start = "features" if method.startswith("markbp_feat") else "uniform"
                model = OverlappingCommunityModel(k, seed=seed, refit=False, mark_weight=1.0,
                                                  mark_model="coupled", coupled_init=start)
                model.fit(n, ei, edge_features=feats)
                used = feats
            readouts["threshold"] = model.F > model.threshold
            readouts["edges"] = edge_readout(n, ei, model.edge_assignments(ei, used), k)
        if "threshold" in readouts and "edges" in readouts:
            # A node joins c when either readout puts it there
            readouts["union"] = readouts["threshold"] | readouts["edges"]
        seconds = round(time.time() - t0, 1)
        return [{**{key: job[key] for key in ("seed", "method", "features")},
                 "readout": readout, **extra,
                 **score(member, truth, n, eval_nodes(job["subset"])),
                 "seconds": seconds}
                for readout, member in readouts.items()]
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "job": job}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--features", nargs="+", required=True,
                        help="feature file stems inside the subset folder")
    parser.add_argument("--shuffle-features", nargs="*", default=[],
                        help="feature stems that also get the shuffled placebo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+",
                        default=["nocd", "edge_kmeans", "markbp_struct", "markbp_feat",
                                 "markbp_feat_shuffle"],
                        help="subset of methods to run (shuffled variants use "
                             "--shuffle-features)")
    parser.add_argument("--restarts", type=int, nargs="+", default=[0],
                        help="restart indices of the fitted models (0 is the "
                             "default initialisation); structure-only methods and "
                             "edge_kmeans run once")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/dblp_markbp.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    jobs = []
    for seed in args.seeds:
        for method in args.methods:
            if method in STRUCTURE_ONLY:
                jobs.append({"seed": seed, "method": method, "features": None})
        for stem in args.features:
            for method in args.methods:
                if method not in STRUCTURE_ONLY and not method.endswith("_shuffle"):
                    jobs.append({"seed": seed, "method": method, "features": stem})
        for stem in args.shuffle_features:
            for method in args.methods:
                if method.endswith("_shuffle"):
                    jobs.append({"seed": seed, "method": method, "features": stem})
    jobs = [{**j, "restart": r} for j in jobs
            for r in (args.restarts if j["method"] not in STRUCTURE_ONLY | {"edge_kmeans"}
                      and not j["method"].endswith("_shuffle") else [0])]
    for job in jobs:
        job["subset"] = args.subset
    # Interleave seeds, so parallel workers start on different seeds and each
    # seed's NOCD fit is computed once and then read from the cache
    jobs.sort(key=lambda j: (j["restart"], j["method"], j["features"] or "", j["seed"]))

    def key(r):
        return (r["seed"], r["method"], r["features"], r.get("restart") or 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {key(r) for r in map(json.loads, open(out)) if r}
        jobs = [j for j in jobs if key(j) not in done]
    logger.info(f"{len(jobs)} jobs on {args.workers} workers")

    def write(result):
        if isinstance(result, dict) and "error" in result:
            logger.error(f"failed {result['job']}: {result['error']}\n{result['traceback']}")
            return
        with open(out, "a") as f:
            for record in result:
                f.write(json.dumps(record) + "\n")
        logger.info(json.dumps({k: result[-1][k] for k in ("seed", "method", "features",
                                                             "f1", "seconds")}))

    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(run_job, jobs):
            write(result)


if __name__ == "__main__":
    main()
