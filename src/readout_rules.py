"""
Fixed readout rules, scored offline on saved fits.

Reads the fits saved by poc_readout.py (data/cache/readout_poc) and
dblp_readout.py (data/cache/readout_dblp) and scores every rule in rules()
against the ground truth. This is the comparison behind the union readout
(threshold OR edge share >= 0.25).

Usage:
    python src/readout_rules.py
"""

import glob
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

try:
    from .metrics import evaluate
    from .poc import GRAPH
    from .subset import load_subset
    from .synthetic import generate
except ImportError:  # Fallback for direct script execution
    from metrics import evaluate
    from poc import GRAPH
    from subset import load_subset
    from synthetic import generate

SUBSET = "data/subsets/cs_areas_le2006_main"


def edge_counts(num_nodes, edge_index, edge_labels, k):
    """Number of each node's edges assigned to each community."""
    has = edge_labels >= 0
    u, v = edge_index[:, has]
    counts = np.zeros((num_nodes, k))
    np.add.at(counts, (u, edge_labels[has]), 1)
    np.add.at(counts, (v, edge_labels[has]), 1)
    return counts


def soft_counts(edge_index, F):
    """Expected number of each node's edges in each community, from F."""
    r = F[edge_index[0]] * F[edge_index[1]] + 1e-10
    r = r / r.sum(axis=1, keepdims=True)
    counts = np.zeros_like(F)
    np.add.at(counts, edge_index[0], r)
    np.add.at(counts, edge_index[1], r)
    return counts


def rules(num_nodes, edge_index, F, threshold, edge_labels, k):
    """Membership matrices of every fixed readout rule."""
    counts = edge_counts(num_nodes, edge_index, edge_labels, k)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    soft = soft_counts(edge_index, F)
    soft_share = soft / np.maximum(soft.sum(axis=1, keepdims=True), 1e-12)
    out = {"thr": F > threshold}
    for tau in (0.15, 0.2, 0.25, 0.33):
        out[f"share{tau}"] = share >= tau
    for m in (1, 2, 3):
        out[f"count>={m}"] = counts >= m
    out["soft>=1"] = soft >= 1
    out["soft>=2"] = soft >= 2
    # Expected edges to other members under the Bernoulli-Poisson model
    out["expect>=1"] = F * (F.sum(axis=0, keepdims=True) - F) >= 1
    out["softshare.25"] = soft_share >= 0.25
    # Count relative to the node's largest community
    out["rel.5"] = counts >= 0.5 * counts.max(axis=1, keepdims=True)
    out["rel.33"] = counts >= 0.33 * counts.max(axis=1, keepdims=True)
    out["thr|share.25"] = out["thr"] | out["share0.25"]  # the union readout
    out["count>=2&share.1"] = (counts >= 2) & (share >= 0.1)
    return out


def score_rules(F, threshold, edge_labels, num_nodes, edge_index, truth):
    for name, member in rules(num_nodes, edge_index, F, threshold, edge_labels,
                              len(truth)).items():
        pred = [np.flatnonzero(member[:, c]) for c in range(member.shape[1])
                if member[:, c].any()]
        yield name, member, evaluate(pred, truth, num_nodes)


def poc_job(path):
    m = re.search(r"(markbp_autoin|markbp_auto)_n([\d.]+)_s([\d.]+)_seed(\d+)", path)
    method, noise, snr, seed = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
    data = generate(seed=seed, noise=noise, snr=snr, **GRAPH)
    fit = np.load(path)
    return [{"method": method, "noise": noise, "snr": snr, "seed": seed, "rule": name,
             "onmi": r["onmi"]}
            for name, _, r in score_rules(fit["F"], float(fit["threshold"]), fit["labels"],
                                          data["num_nodes"], data["edge_index"],
                                          data["communities"])]


def dblp_job(path):
    m = re.search(r"main_(edge_feats_\w+?)_seed(\d+)", path)
    n, ei, truth = load_subset(SUBSET)
    fit = np.load(path)
    return [{"features": m.group(1), "seed": int(m.group(2)), "choice": str(fit["choice"]),
             "rule": name, "memberships": member.sum(axis=1).mean(),
             **{k: r[k] for k in ("f1", "jaccard", "onmi")}}
            for name, member, r in score_rules(fit["F"], float(fit["threshold"]),
                                               fit["labels"], n, ei, truth)]


def main():
    with ProcessPoolExecutor(4) as pool:
        poc = pd.DataFrame([r for rs in pool.map(poc_job, glob.glob("data/cache/readout_poc/*.npz"))
                            for r in rs])
        dblp = pd.DataFrame([r for rs in pool.map(dblp_job,
                                                   sorted(glob.glob("data/cache/readout_dblp/*.npz")))
                             for r in rs])
    poc.to_csv("results/readout_rules_poc.csv", index=False)
    dblp.to_csv("results/readout_rules_dblp.csv", index=False)
    poc = poc[poc["method"] == "markbp_autoin"]
    table = pd.concat([poc.groupby("rule")["onmi"].mean().rename("synthetic_onmi"),
                       dblp.groupby("rule")[["f1", "jaccard", "onmi"]].mean().add_prefix("dblp_")],
                      axis=1)
    print(table.sort_values("synthetic_onmi", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
