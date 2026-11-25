"""
Readout rules on DBLP-EF.

Fits MarkBP-auto with input features (the same configuration as
dblp_markbp.py --methods markbp_autoin) and saves F, the threshold and the
edge assignments, so readout rules can be compared without refitting.

Usage:
    python src/dblp_readout.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title ... --seeds 0 1 2 --workers 2
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    from .markbp_auto import MarkBPAuto
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from markbp_auto import MarkBPAuto
    from subset import load_subset

logger = logging.getLogger(__name__)


def fit_path(subset, stem, seed):
    return Path("data/cache/readout_dblp") / f"{Path(subset).name}_{stem}_seed{seed}.npz"


def run_job(job):
    import torch
    torch.set_num_threads(1)
    path = fit_path(job["subset"], job["features"], job["seed"])
    if path.exists():
        return str(path)
    n, ei, truth = load_subset(job["subset"])
    feats = np.load(Path(job["subset"]) / f"{job['features']}.npy")
    model = MarkBPAuto(len(truth), seed=job["seed"], cache_dir="data/cache/nocd_dblp",
                       input_features=True)
    model.fit(n, ei, edge_features=feats)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, F=model.F, threshold=model.threshold,
                        labels=model.edge_assignments(ei), choice=model.choice)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    jobs = [{"subset": args.subset, "features": f, "seed": s}
            for s in args.seeds for f in args.features]
    with ProcessPoolExecutor(args.workers) as pool:
        for path in pool.map(run_job, jobs):
            logger.info(f"saved {path}")


if __name__ == "__main__":
    main()
