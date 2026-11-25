"""
LaTeX tables for MarkBP-auto, from its end-to-end runs.

* poc_auto.tex: controlled graphs (degree 8, 20 seeds, three background
  levels). Mean homophily and alignment z-scores with real features, the
  homophily z with shuffled features, ONMI of NOCD, edge-feature clustering and
  MarkBP-auto (edge readout; real and shuffled features), and how often each
  branch was chosen with real features.
* dblp_feature_test.tex: both tests and the chosen branch on DBLP-EF.

Usage:
    python src/auto_tables.py --out paper/tables
"""

import argparse
import json
from pathlib import Path

import pandas as pd

STEMS = [("edge_feats_title", "Title"),
         ("edge_feats_title_abstract", "Title + abstract"),
         ("edge_feats_title_outside", "Title, outside"),
         ("edge_feats_title_abstract_outside", "Title + abs., outside")]
BRANCH = {"markbp": "M", "edge_clustering": "C", "structure": "S"}


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


def poc_table(auto, grid, metric="onmi"):
    keys = ["noise", "snr"]
    auto = auto[auto["readout"] == "edges"]
    real = auto[auto["method"] == "markbp_auto"]
    shuf = auto[auto["method"] == "markbp_auto_shuffle"]
    nocd = grid[(grid["method"] == "nocd") & (grid["readout"] == "threshold")]
    kmeans = grid[grid["method"] == "edge_kmeans"]
    counts = pd.crosstab([real["noise"], real["snr"]], real["choice"])
    t = pd.DataFrame({
        "hz": real.groupby(keys)["homophily_z"].mean(),
        "az": real.groupby(keys)["test_z"].mean(),
        "hz_shuf": shuf.groupby(keys)["homophily_z"].mean(),
        "nocd": nocd.groupby(keys)[metric].mean(),
        "kmeans": kmeans.groupby(keys)[metric].mean(),
        "auto": real.groupby(keys)[metric].mean(),
        "auto_shuf": shuf.groupby(keys)[metric].mean(),
    })
    lines = [r"\begin{tabular}{cc rrr cccc c}", r"\toprule",
             r" & & \multicolumn{2}{c}{Real features} & Shuffled & "
             r"\multicolumn{4}{c}{ONMI} & Branch \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-9}",
             r"Background & $s$ & $z_{\text{hom}}$ & $z_{\text{align}}$ & $z_{\text{hom}}$ & "
             r"NOCD & Edge clust. & \methodauto & shuffled & M/C/S \\", r"\midrule"]
    for (noise, snr), row in t.iterrows():
        c = counts.loc[(noise, snr)] if (noise, snr) in counts.index else {}
        branch = "/".join(str(int(c.get(b, 0))) for b in BRANCH)
        az = "--" if pd.isna(row.az) else f"{row.az:.1f}"
        best = max(row.nocd, row.kmeans, row.auto)
        cells = [f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
                 for v in (row.nocd, row.kmeans, row.auto)]
        lines.append(f"{int(noise * 100)}\\% & {snr:g} & {row.hz:.1f} & {az} & "
                     f"{row.hz_shuf:.1f} & " + " & ".join(cells)
                     + f" & {row.auto_shuf:.3f} & {branch} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", t


def dblp_test_table(auto):
    auto = auto[auto["readout"] == "threshold"]
    lines = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Real features} & \multicolumn{3}{c}{Shuffled features} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Edge features & $z_{\text{hom}}$ & $z_{\text{align}}$ & Branch & "
             r"$z_{\text{hom}}$ & $z_{\text{align}}$ & Branch \\", r"\midrule"]
    for stem, label in STEMS:
        cells = []
        for method in ("markbp_auto", "markbp_auto_shuffle"):
            g = auto[(auto["features"] == stem) & (auto["method"] == method)]
            az = g["test_z"].dropna()
            branches = "/".join(str(int((g["choice"] == b).sum())) for b in BRANCH)
            cells += [f"{g['homophily_z'].mean():,.1f}",
                      "--" if az.empty else f"{az.mean():,.0f}", branches]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-auto", default="results/poc_auto3.jsonl")
    parser.add_argument("--poc-grid", default="results/poc_grid.jsonl")
    parser.add_argument("--dblp-auto", default="results/dblp_markbp_auto3.jsonl")
    parser.add_argument("--out", default="paper/tables")
    args = parser.parse_args()

    out = Path(args.out)
    tex, t = poc_table(load(args.poc_auto), load(args.poc_grid))
    (out / "poc_auto.tex").write_text(tex)
    print(t.round(3).to_markdown())
    (out / "dblp_feature_test.tex").write_text(dblp_test_table(load(args.dblp_auto)))
    print((out / "dblp_feature_test.tex").read_text())


if __name__ == "__main__":
    main()
