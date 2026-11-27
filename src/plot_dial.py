"""
The alignment dial on controlled graphs: F1 against the share of the edge text
that is about the true communities (align = 1) rather than an unrelated
node grouping (align = 0). Writes a figure and a LaTeX table.

    python src/plot_dial.py results/poc_dial.jsonl \
        --fig paper/figures/poc_dial --table paper/tables/poc_dial.tex
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUR_READOUT = "union"
# method -> label, colour (validated categorical slots), marker
SERIES = [
    ("markbp_autoin", r"MarkBP-auto + input features", "#2a78d6", "o"),
    ("nocd_eagg", "NOCD + averaged features", "#1baf7a", "^"),
    ("dynares_eagg", "DynaResGCN + averaged features", "#e87ba4", "v"),
    ("edge_kmeans", "Edge-feature clustering", "#eb6834", "s"),
    ("nocd", "NOCD (structure only)", "#eda100", "D"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def scored(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Each method at its table readout: union for ours, best mean for baselines."""
    parts = []
    for method, g in df.groupby("method"):
        readout = OUR_READOUT if method == "markbp_autoin" else \
            g.groupby("readout")[metric].mean().idxmax()
        parts.append(g[g["readout"] == readout])
    return pd.concat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--fig", default="paper/figures/poc_dial")
    parser.add_argument("--table", default="paper/tables/poc_dial.tex")
    args = parser.parse_args()
    df = scored(pd.read_json(args.results, lines=True))
    stats = df.groupby(["noise", "align", "method"])["f1"].agg(["mean", "sem"]).reset_index()

    noises = sorted(stats["noise"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(3.4 * len(noises), 2.9), sharey=True)
    for ax, noise in zip(axes, noises):
        s = stats[stats["noise"] == noise]
        for method, label, color, marker in SERIES:
            m = s[s["method"] == method].sort_values("align")
            ax.errorbar(m["align"], m["mean"], yerr=m["sem"], color=color, marker=marker,
                        markersize=5, linewidth=2, capsize=0, label=label,
                        linestyle="--" if method == "nocd" else "-")
        ax.set_title(f"{int(noise * 100)}% background edges", fontsize=9, color=INK)
        ax.set_xlabel(r"share of edge text about the true communities ($\alpha$)",
                      fontsize=8, color=INK)
        ax.set_xticks(sorted(s["align"].unique()))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("F1", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.fig}.{ext}", dpi=200, bbox_inches="tight")

    # Table: mean F1 per method and alignment, one block per background level
    aligns = sorted(stats["align"].unique(), reverse=True)
    lines = [r"\begin{tabular}{ll" + "c" * len(aligns) + "}", r"\toprule",
             r"Background & Method & " + " & ".join(f"$\\alpha = {a:g}$" for a in aligns)
             + r" \\", r"\midrule"]
    for i, noise in enumerate(noises):
        if i:
            lines.append(r"\midrule")
        for j, (method, label, *_ ) in enumerate(SERIES):
            row = stats[(stats["noise"] == noise) & (stats["method"] == method)] \
                .set_index("align")["mean"]
            label = label.replace("MarkBP-auto", r"\methodauto")
            first = f"{int(noise * 100)}\\%" if j == 0 else ""
            lines.append(f"{first} & {label} & "
                         + " & ".join(f"{row[a]:.3f}" for a in aligns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(args.table).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.fig}.pdf/.png and {args.table}")


if __name__ == "__main__":
    main()
