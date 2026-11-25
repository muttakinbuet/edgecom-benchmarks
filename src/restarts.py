"""
Restarts of the joint fit: does keeping the restart with the lowest held-out
loss avoid poor optima?

Each (method, feature set, seed) is fitted with restarts 0..R-1
(src/dblp_markbp.py --restarts). All restarts of a seed share its held-out
split, so their held-out losses are comparable. Selection is label-free:
ours (MarkBP-auto) keeps the restart with the lowest joint held-out loss
(joint_val), the averaged-feature baseline (nocd_eagg) the one with the
lowest held-out loss (val). Labels are used only to score.

Reported per benchmark and feature set, as means over seeds:
    single     restart 0 (the main-table run)
    selected   the restart picked by held-out loss
    mean/worst over all restarts
    below      runs under structure alone (NOCD of the same seed), for the
               single run and for the selected one
and the Spearman correlation between held-out loss and F1 within seeds.

Usage:
    python src/restarts.py            # prints the summary, writes JSON and LaTeX
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUR_READOUT = "union"
BENCHMARKS = [
    # name, restart file, file with the NOCD structure-only runs
    ("CS-Areas", "results/restarts/cs_areas.jsonl", "results/dblp_cs_areas_10seeds.jsonl"),
    ("Pat-G subclasses", "results/restarts/patents_g.jsonl", "results/patents_g.jsonl"),
    ("Pat-G assignees", "results/restarts/patents_g_assignee.jsonl",
     "results/patents_g_assignee.jsonl"),
    ("Pat-G subclasses (bge)", "results/restarts/patents_g_bge.jsonl", "results/patents_g.jsonl"),
]
FEATURE_NAMES = {"edge_feats_title": "title",
                 "edge_feats_title_abstract": "title+abstract",
                 "edge_feats_title_outside": "title, outside",
                 "edge_feats_title_abstract_outside": "title+abstract, outside",
                 "edge_feats_title_abstract_bge": "title+abstract",
                 "edge_feats_title_abstract_outside_bge": "title+abstract, outside"}
METHOD_NAMES = {"markbp_autoin": "ours", "nocd_eagg": "NOCD + avg."}
LOSS = {"markbp_autoin": "joint_val", "nocd_eagg": "val"}


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    df = pd.DataFrame(rows)
    if "restart" not in df:
        df["restart"] = np.nan
    df["restart"] = df["restart"].fillna(0).astype(int)
    return df


def readout_for(df, method):
    """Ours: the method's union readout; baselines: their best readout at restart 0."""
    if method == "markbp_autoin":
        return OUR_READOUT
    g = df[(df["method"] == method) & (df["restart"] == 0)]
    return g.groupby("readout")["f1"].mean().idxmax()


def summarise(name, path, structure_path):
    df = load(path)
    nocd = load(structure_path)
    nocd = nocd[(nocd["method"] == "nocd") & (nocd["readout"] == OUR_READOUT)]
    structure = nocd.set_index("seed")["f1"]
    out = []
    for method in [m for m in LOSS if m in set(df["method"])]:
        readout = readout_for(df, method)
        g = df[(df["method"] == method) & (df["readout"] == readout)]
        for feats, gf in g.groupby("features"):
            seeds, rhos = [], []
            for seed, gs in gf.groupby("seed"):
                gs = gs.sort_values("restart")
                if 0 not in set(gs["restart"]) or len(gs) < 2:
                    continue
                loss = gs[LOSS[method]].astype(float)
                if loss.isna().any():
                    # Tests chose structure alone or edge clustering: restarts
                    # do not change the fit
                    pick = gs.iloc[0]
                else:
                    pick = gs.loc[loss.idxmin()]
                    if loss.nunique() > 2:
                        rhos.append(spearmanr(loss, gs["f1"]).statistic)
                single = gs[gs["restart"] == 0].iloc[0]
                ref = structure.get(seed, np.nan)
                seeds.append({"seed": seed, "restarts": len(gs),
                              "single": single["f1"], "selected": pick["f1"],
                              "single_onmi": single["onmi"], "selected_onmi": pick["onmi"],
                              "mean": gs["f1"].mean(), "worst": gs["f1"].min(),
                              "best": gs["f1"].max(), "picked": int(pick["restart"]),
                              "structure": ref,
                              "single_below": bool(single["f1"] < ref),
                              "selected_below": bool(pick["f1"] < ref),
                              "any_below": bool((gs["f1"] < ref).any())})
            if not seeds:
                continue
            s = pd.DataFrame(seeds)
            out.append({"benchmark": name, "method": method, "features": feats,
                        "readout": readout, "seeds": len(s),
                        "restarts": int(s["restarts"].min()),
                        **{k: float(s[k].mean()) for k in
                           ("single", "selected", "mean", "worst", "best",
                            "single_onmi", "selected_onmi", "structure")},
                        "single_sd": float(s["single"].std(ddof=0)),
                        "selected_sd": float(s["selected"].std(ddof=0)),
                        "picked_not_0": int((s["picked"] != 0).sum()),
                        "single_below": int(s["single_below"].sum()),
                        "selected_below": int(s["selected_below"].sum()),
                        "runs_below": int(s["any_below"].sum()),
                        "spearman_loss_f1": float(np.mean(rhos)) if rhos else None,
                        "per_seed": seeds})
    return out


def latex(rows, path):
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Benchmark & Features & Method & Restart 0 & Selected & Worst & "
             r"Below str.\ (0 / sel.) & Seeds \\", r"\midrule"]
    last = None
    for r in rows:
        bench = r["benchmark"] if r["benchmark"] != last else ""
        if last is not None and bench:
            lines.append(r"\midrule")
        last = r["benchmark"]
        lines.append(f"{bench} & {FEATURE_NAMES.get(r['features'], r['features'])} & "
                     f"{METHOD_NAMES[r['method']]} & {r['single']:.3f} & {r['selected']:.3f} & "
                     f"{r['worst']:.3f} & {r['single_below']} / {r['selected_below']} & "
                     f"{r['seeds']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = []
    for name, path, structure_path in BENCHMARKS:
        if Path(path).exists():
            rows += summarise(name, path, structure_path)
    if not rows:
        print("no restart results yet")
        return
    Path("results/restarts").mkdir(parents=True, exist_ok=True)
    Path("results/restarts/summary.json").write_text(json.dumps(rows, indent=1, default=float))
    latex(rows, "paper/tables/restarts.tex")
    cols = ["benchmark", "method", "features", "seeds", "restarts", "single", "selected",
            "mean", "worst", "best", "structure", "picked_not_0", "single_below",
            "selected_below", "runs_below", "spearman_loss_f1"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.precision", 3):
        print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
