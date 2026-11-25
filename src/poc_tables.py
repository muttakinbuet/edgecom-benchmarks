"""
LaTeX table comparing method variants on the proof-of-concept graphs
(degree 8, seeds 0-9): baselines and MarkBP-vMF from the grid run, and the
coupled model from its two starts (structure-led, feature-led) from the
start-selection run, with real and shuffled edge features.

Usage:
    python src/poc_tables.py --grid results/poc_grid.jsonl \
        --selection results/poc_selection.jsonl --out paper/tables/poc_variants.tex
"""

import argparse
import json

import pandas as pd


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--metric", default="overlap_exact")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    m = args.metric

    grid = load(args.grid)
    grid = grid[(grid["seed"] < args.seeds) & grid["noise"].isin([0.1, 0.3])]
    sel = load(args.selection)
    sel = sel[sel["seed"] < args.seeds]

    columns = {}
    for label, method, readout in (("NOCD", "nocd", "threshold"),
                                   ("NOCD + avg.", "nocd_eagg", "threshold"),
                                   ("Edge clust.", "edge_kmeans", "edges"),
                                   ("vMF", "markbp", "edges")):
        g = grid[(grid["method"] == method) & (grid["readout"] == readout)]
        columns[label] = g.groupby(["noise", "snr"])[m].mean()
    real, shuf = sel[sel["features"] == "real"], sel[sel["features"] == "shuffled"]
    columns["Coupled, struct.\\ start"] = real.groupby(["noise", "snr"])[f"{m}_uniform"].mean()
    columns["Coupled, feat.\\ start"] = real.groupby(["noise", "snr"])[f"{m}_features"].mean()
    columns["Feat.\\ start, shuffled"] = shuf.groupby(["noise", "snr"])[f"{m}_features"].mean()
    table = pd.DataFrame(columns)

    best_cols = [c for c in table.columns if "shuffled" not in c]
    lines = ["\\begin{tabular}{cc" + "c" * len(table.columns) + "}", "\\toprule",
             "Background & $s$ & " + " & ".join(table.columns) + " \\\\", "\\midrule"]
    for (noise, snr), row in table.iterrows():
        best = row[best_cols].max()
        cells = [f"\\textbf{{{v:.2f}}}" if (c in best_cols and v == best) else f"{v:.2f}"
                 for c, v in row.items()]
        lines.append(f"{int(noise * 100)}\\% & {snr:g} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(table.round(2).to_markdown())


if __name__ == "__main__":
    main()
