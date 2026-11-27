"""Paired significance tests for the comparisons quoted in the paper text.

Seeds are the unit of replication. A method's score for a seed is its mean
over the feature sets of the comparison, so a structure-only method (one run
per seed) is never counted twice. For each comparison we report the seeds won,
a two-sided exact Wilcoxon signed-rank test and a two-sided paired t-test on the
per-seed means, and Holm-adjusted p-values over the whole family.

    python -m src.significance
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

OURS = "markbp_autoin"
OUR_READOUT = "union"
TEXT = ["edge_feats_title", "edge_feats_title_abstract"]
OUTSIDE = ["edge_feats_title_outside", "edge_feats_title_abstract_outside"]
ALL4 = TEXT + OUTSIDE

# (label, file, ours, rival, feature sets)
DBLP = {"CS-Areas": "results/dblp_cs_areas_le2006_main_ours_l01.jsonl",
        "CS-Systems": "results/dblp_cs_systems_le2006_main_ours_l01.jsonl",
        "Random-30": "results/dblp_rand30_s1_ours_l01.jsonl"}
DBLP_BASE = {"CS-Areas": "results/dblp_cs_areas_10seeds.jsonl",
             "CS-Systems": "results/dblp_cs_systems_le2006_main.jsonl",
             "Random-30": "results/dblp_rand30_s1.jsonl"}
INST = "results/dblp_cs_areas_aff.jsonl"
PAT = "results/patents_g.jsonl"
ASG = "results/patents_g_assignee.jsonl"


def load(*paths):
    return pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)


def load_patents(path):
    """Patent runs; MarkBP-auto with outside features comes from the run without
    the held-out check (*_nocheck.jsonl), as in the paper's table."""
    df = load(path)
    nocheck = load(path.replace(".jsonl", "_nocheck.jsonl"))
    nocheck["method"] = nocheck["method"].str.replace("_nocheck", "")
    drop = (df["method"] == OURS) & df["features"].isin(OUTSIDE)
    return pd.concat([df[~drop], nocheck[nocheck["method"] == OURS]], ignore_index=True)


def per_seed(df, method, features, metric="f1"):
    """Per-seed mean over the given feature sets, at the table readout."""
    g = df[df["method"] == method]
    if method == OURS:
        readout = OUR_READOUT
    else:  # best readout by mean over all the method's runs, as in the tables
        readout = g.groupby("readout")[metric].mean().idxmax()
    if g["features"].notna().any():
        g = g[g["features"].isin(features)]
    g = g[g["readout"] == readout]
    # duplicate (seed, features) rows: keep the last, as the tables do
    g = g.drop_duplicates(["seed", "features"], keep="last")
    return g.groupby("seed")[metric].mean()


def compare(label, df, rival, features, ours=OURS, metric="f1"):
    a, b = per_seed(df, ours, features, metric), per_seed(df, rival, features, metric)
    seeds = a.index.intersection(b.index)
    a, b = a[seeds], b[seeds]
    d = a - b
    p_w = wilcoxon(d, method="exact").pvalue if (d != 0).any() else 1.0
    p_t = ttest_rel(a, b).pvalue
    return dict(comparison=label, rival=rival, n=len(d), ours=a.mean(), theirs=b.mean(),
                won=int((d > 0).sum()), p_wilcoxon=p_w, p_t=p_t)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    rows = []
    for name in DBLP:
        df = pd.concat([load(DBLP_BASE[name]).query("method != @OURS"), load(DBLP[name])])
        rows.append(compare(f"DBLP {name}, all features", df, "nocd_eagg", ALL4))
        rows.append(compare(f"DBLP {name}, all features", df, "dynares_eagg", ALL4))
    inst = load(INST)
    for rival in ["nocd", "nocd_eagg", "dynares_eagg"]:
        rows.append(compare("Institutions, text", inst, rival, TEXT))
    pat = load_patents(PAT)
    for rival in ["nocd", "edge_kmeans", "nocd_gine", "dynares_eagg", "nocd_eagg"]:
        rows.append(compare("Pat-G subclass, text", pat, rival, TEXT))
    for rival in ["nocd_eagg", "dynares_eagg"]:
        rows.append(compare("Pat-G subclass, outside", pat, rival, OUTSIDE))
    asg = load_patents(ASG)
    for rival in ["nocd", "dynares_eagg", "nocd_eagg"]:
        rows.append(compare("Pat-G assignee, text", asg, rival, TEXT))
    rows.append(compare("Pat-G assignee, outside", asg, "nocd", OUTSIDE))

    out = pd.DataFrame(rows)
    out["holm_wilcoxon"] = holm(out["p_wilcoxon"])
    out["holm_t"] = holm(out["p_t"])
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv("results/significance.csv", index=False)


if __name__ == "__main__":
    main()
