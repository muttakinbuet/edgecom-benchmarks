"""
Figure for the proof-of-concept grid: exact recovery of each node's community
set, for overlapping and single-community nodes, as feature signal grows,
one panel per structure level (fraction of background edges).

Usage:
    python src/plot_poc.py results/poc_grid.jsonl --out paper/figures/poc_grid
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (method, readout) -> label, color (validated categorical slots 1-5), marker, style
SERIES = [
    (("markbp", "edges"), "MarkBP (ours)", "#2a78d6", "o", "-"),
    (("edge_kmeans", "edges"), "Edge features only", "#eb6834", "s", "-"),
    (("nocd_eagg", "threshold"), "NOCD + averaged edge features", "#1baf7a", "^", "-"),
    (("nocd", "threshold"), "NOCD (structure only)", "#eda100", "D", "-"),
    (("markbp_shuffle", "edges"), "MarkBP, shuffled features", "#e87ba4", "v", "--"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
ROWS = [("overlap_exact", "Overlapping nodes:\nexact community set"),
        ("single_exact", "Single-community nodes:\nexact community set")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--out", default="paper/figures/poc_grid")
    args = parser.parse_args()

    df = pd.DataFrame([json.loads(l) for l in open(args.results) if l.strip()])
    noises = sorted(df["noise"].unique())
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(len(ROWS), len(noises), figsize=(7.0, 4.2), sharex=True,
                             sharey=True)
    for r, (metric, ylabel) in enumerate(ROWS):
        for c, noise in enumerate(noises):
            ax = axes[r, c]
            for (method, readout), label, color, marker, style in SERIES:
                g = df[(df["method"] == method) & (df["readout"] == readout)
                       & (df["noise"] == noise)].groupby("snr")[metric]
                mean, sem = g.mean(), g.std() / np.sqrt(g.count())
                ax.errorbar(mean.index, mean.values, yerr=sem.values, color=color,
                            marker=marker, markersize=4.5, linewidth=1.6, linestyle=style,
                            capsize=0, elinewidth=1.0, label=label,
                            markeredgecolor="white", markeredgewidth=0.6)
            ax.set_ylim(0, 1)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(f"{int(noise * 100)}% background edges", color=INK, fontsize=8.5)
            if r == len(ROWS) - 1:
                ax.set_xlabel("Edge-feature signal $s$")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(df["snr"].unique()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
