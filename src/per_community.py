"""
Per-community results on DBLP-EF CS-Areas (title + abstract features, seeds 0-2).

For every ground-truth venue, the F1 of its best-matching predicted community
(the truth side of the bimatching F1), for NOCD (structure only, threshold
readout), NOCD + averaged features (threshold readout, as in the case study)
and MarkBP-auto + input features (union readout). Uses the fits cached by
case_study.py and the NOCD cache; nothing is refitted.

Each venue is also described by two properties fixed before any fit:
  conductance  share of its members' edge endpoints that leave the venue
               (high = structurally hard to find);
  text_cohesion  mean cosine between the features of its internal edges and
               their mean direction (high = topically distinctive text).

Usage:
    python src/per_community.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from subset import load_subset

SUBSET = Path("data/subsets/cs_areas_le2006_main")
FEATURES = "edge_feats_title_abstract"
CASE = Path("data/cache/case_study")
NOCD = Path("data/cache/nocd_dblp")
SEEDS = (0, 1, 2)
OUT = Path("results/per_community/cs_areas.json")


def truth_f1(member, truth_member):
    """F1 of each truth community's best-matching predicted community."""
    member = member[:, member.any(0)].astype(float)
    inter = member.T @ truth_member                                    # P x T
    denom = member.sum(0)[:, None] + truth_member.sum(0)[None, :]
    return (2 * inter / np.maximum(denom, 1e-9)).max(axis=0)


def main():
    n, ei, truth = load_subset(SUBSET)
    k = len(truth)
    meta = pd.read_csv(SUBSET / "communities.meta.tsv", sep="\t")
    names = meta["venue_name"].str.replace(" Conference", "").tolist()
    truth_member = np.zeros((n, k))
    for c, m in enumerate(truth):
        truth_member[m, c] = 1

    # Venue properties
    feats = np.load(SUBSET / f"{FEATURES}.npy")
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    tm = truth_member.astype(bool)
    conductance, cohesion = [], []
    for c in range(k):
        a, b = tm[ei[0], c], tm[ei[1], c]
        inside, cut = (a & b).sum(), (a ^ b).sum()
        conductance.append(cut / (2 * inside + cut))
        x = feats[a & b]
        mu = x.mean(0)
        cohesion.append(float((x @ (mu / np.linalg.norm(mu))).mean()))

    nocd_files = {s: next(NOCD.glob(f"nocd_*_n{n}_k{k}_s{s}.npz")) for s in SEEDS}
    scores = {"nocd": [], "nocd_eagg": [], "ours": []}
    for s in SEEDS:
        scores["nocd"].append(truth_f1(np.load(nocd_files[s])["F"] > 0.5, truth_member))
        suffix = "" if s == 0 else f"_s{s}"
        scores["nocd_eagg"].append(truth_f1(np.load(CASE / f"nocd_eagg{suffix}.npz")["threshold"], truth_member))
        scores["ours"].append(truth_f1(np.load(CASE / f"ours{suffix}.npz")["union"], truth_member))
    mean = {m: np.mean(v, axis=0) for m, v in scores.items()}

    df = pd.DataFrame({"venue": names, "size": meta["size"], "conductance": conductance,
                       "text_cohesion": cohesion, **mean})
    df["gain_vs_nocd"] = df["ours"] - df["nocd"]
    df["gain_vs_eagg"] = df["ours"] - df["nocd_eagg"]
    df = df.sort_values("gain_vs_eagg", ascending=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))

    summary = {
        "mean": {m: round(float(v.mean()), 4) for m, v in mean.items()},
        "ours_above_nocd": int((df["gain_vs_nocd"] > 0).sum()),
        "ours_above_eagg": int((df["gain_vs_eagg"] > 0).sum()),
        "ours_below_eagg_by_0.05": int((df["gain_vs_eagg"] < -0.05).sum()),
        "wilcoxon_ours_vs_eagg_p": float(wilcoxon(df["ours"], df["nocd_eagg"]).pvalue),
        "venues": k,
    }
    for prop in ("conductance", "text_cohesion", "size"):
        for target in ("nocd", "ours", "gain_vs_nocd", "gain_vs_eagg"):
            r, p = spearmanr(df[prop], df[target])
            summary[f"spearman_{prop}_{target}"] = [round(float(r), 3), round(float(p), 4)]
    print(json.dumps(summary, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "venues": df.round(4).to_dict("records")},
                              indent=1))


if __name__ == "__main__":
    main()
