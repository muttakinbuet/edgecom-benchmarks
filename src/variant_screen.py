"""
Screen of MarkBP-auto variants on one DBLP-EF subset.

Every variant is MarkBP-auto with input features; only the MarkBP branch
changes (the tests and fallbacks do not):
    base        feature-led start, mark weight 1, M = K feature clusters (current)
    uniform     structure-led start (uniform coupling, no pretraining to the clusters)
    uniform_m2  structure-led, M = 2K feature clusters
    uniform_m4  structure-led, M = 4K feature clusters
    feat_l03    feature-led, mark weight 0.3
    feat_l01    feature-led, mark weight 0.1
    uniform_l03 structure-led, mark weight 0.3
    l0          mark weight 0: NOCD with averaged edge features as input
    struct_first     100 epochs without marks, then marks with the coupling
                     estimated from those communities (mark weight 1)
    struct_first_l03 the same with mark weight 0.3
    struct_first_m2  the same with M = 2K feature clusters
Each record keeps the held-out Bernoulli-Poisson loss of the fit (best_val), so
label-free selection among variants can be checked against the labels.

Usage:
    python src/variant_screen.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title_abstract edge_feats_title_abstract_outside \
        --seeds 0 1 2 --workers 7 --out results/variant_screen_cs_areas.jsonl
"""

import argparse
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from .dblp_markbp import eval_nodes, score
    from .markbp_auto import MarkBPAuto
    from .poc import edge_readout
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from dblp_markbp import eval_nodes, score
    from markbp_auto import MarkBPAuto
    from poc import edge_readout
    from subset import load_subset

logger = logging.getLogger(__name__)


def variants(k):
    # Every variant states its mark weight: MarkBPAuto's default changed from
    # 1.0 to 0.1, and the screen compares against the earlier default 1.0
    return {
        "base": {"mark_weight": 1.0},
        "uniform": {"coupled_init": "uniform", "mark_weight": 1.0},
        "uniform_m2": {"coupled_init": "uniform", "feature_clusters": 2 * k, "mark_weight": 1.0},
        "uniform_m4": {"coupled_init": "uniform", "feature_clusters": 4 * k, "mark_weight": 1.0},
        "feat_l03": {"mark_weight": 0.3},
        "feat_l01": {"mark_weight": 0.1},
        "uniform_l03": {"coupled_init": "uniform", "mark_weight": 0.3},
        "l0": {"mark_weight": 0.0},
        "struct_first": {"coupled_init": "structure", "mark_weight": 1.0},
        "struct_first_l03": {"coupled_init": "structure", "mark_weight": 0.3},
        "struct_first_m2": {"coupled_init": "structure", "feature_clusters": 2 * k, "mark_weight": 1.0},
    }


def run_job(job):
    import torch
    torch.set_num_threads(1)
    try:
        n, ei, truth = load_subset(job["subset"])
        k, seed = len(truth), job["seed"]
        feats = np.load(Path(job["subset"]) / f"{job['features']}.npy")
        t0 = time.time()
        model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_dblp", input_features=True,
                           **variants(k)[job["variant"]])
        model.fit(n, ei, edge_features=feats)
        labels = model.edge_assignments(ei)
        readouts = {"threshold": model.F > model.threshold,
                    "edges": edge_readout(n, ei, labels, k)}
        readouts["union"] = readouts["threshold"] | readouts["edges"]
        # Keep the fit for later analysis of the memberships
        fits = Path("data/cache/variant_screen")
        fits.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fits / "{}_{variant}_{features}_seed{seed}.npz".format(
            Path(job["subset"]).name, **job), F=model.F, threshold=model.threshold, labels=labels)
        best_val = getattr(model.model, "best_val", None)
        seconds = round(time.time() - t0, 1)
        return [{**{key: job[key] for key in ("seed", "variant", "features")},
                 "readout": readout, "choice": model.choice,
                 "best_val": None if best_val is None else float(best_val),
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
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=list(variants(1)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    jobs = [{"subset": args.subset, "seed": s, "variant": v, "features": f}
            for s in args.seeds for f in args.features for v in args.variants]
    out = Path(args.out)
    if out.exists():
        done = {(r["seed"], r["variant"], r["features"]) for r in map(json.loads, open(out))}
        jobs = [j for j in jobs if (j["seed"], j["variant"], j["features"]) not in done]
    logger.info(f"{len(jobs)} jobs on {args.workers} workers")
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(run_job, jobs):
            if isinstance(result, dict):
                logger.error(f"failed {result['job']}: {result['error']}\n{result['traceback']}")
                continue
            with open(out, "a") as f:
                for record in result:
                    f.write(json.dumps(record) + "\n")
            logger.info(json.dumps({k: result[-1][k] for k in ("seed", "variant", "features",
                                                                 "f1", "onmi", "seconds")}))


if __name__ == "__main__":
    main()
