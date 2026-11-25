"""
Run the label-free feature-alignment test on a DBLP-EF subset.

For each seed, fit structure-only NOCD (same settings and seed as in
dblp_markbp.py), assign every edge to a community, and test each edge-feature
set, real and shuffled, against that structure.

Usage:
    python src/dblp_feature_test.py --subset data/subsets/cs_areas_le2006_main \
        --features edge_feats_title edge_feats_title_abstract \
        edge_feats_title_outside edge_feats_title_abstract_outside \
        --seeds 0 1 2 --workers 3 --out results/dblp_feature_test.jsonl
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
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from feature_test import alignment_test
    from ocd_models import OverlappingCommunityModel
    from subset import load_subset


def run_seed(job):
    import torch
    torch.set_num_threads(1)
    n, ei, truth = load_subset(job["subset"])
    k, seed = len(truth), job["seed"]
    model = OverlappingCommunityModel(k, seed=seed, refit=False)
    model.fit(n, ei)
    labels = model.edge_assignments(ei)
    records = []
    for stem in job["features"]:
        feats = np.load(Path(job["subset"]) / f"{stem}.npy")
        for variant in ("real", "shuffled"):
            x = feats if variant == "real" else \
                feats[np.random.default_rng(seed + 12345).permutation(len(feats))]
            records.append({"seed": seed, "features": stem, "variant": variant,
                            **alignment_test(labels, x, k, seed)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="results/dblp_feature_test.jsonl")
    args = parser.parse_args()

    jobs = [{"subset": args.subset, "seed": s, "features": args.features}
            for s in args.seeds]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    with Pool(args.workers) as pool, open(out, "w") as f:
        for records in pool.imap_unordered(run_seed, jobs):
            for r in records:
                f.write(json.dumps(r) + "\n")
                print({k: r[k] for k in ("seed", "features", "variant", "z", "use_features")},
                      flush=True)


if __name__ == "__main__":
    main()
