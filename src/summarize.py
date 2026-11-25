"""
Summarise experiment results (JSON lines from run_experiment.py) as mean +- std
tables, in Markdown and optionally LaTeX.

Usage:
    python src/summarize.py results/synthetic_snr.jsonl --by snr
    python src/summarize.py results/dblp_cs_areas_title.jsonl --latex paper/tables/dblp.tex
"""

import argparse
import json

import pandas as pd

METRICS = ["onmi", "f1", "jaccard", "edge_acc"]
METHOD_ORDER = ["bigclam", "nocd", "nocd_eagg", "markbp", "markbp_eagg", "markbp_shuffle"]


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by):
    metrics = [m for m in METRICS if m in df and df[m].notna().any()]
    keys = [*by, "method"]
    agg = df.groupby(keys)[metrics].agg(["mean", "std", "count"])
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    agg = agg.reset_index()
    agg["_order"] = agg["method"].map(order).fillna(len(order))
    agg = agg.sort_values([*by, "_order"]).drop(columns="_order")
    return agg, metrics


def fmt(agg, metrics, by, latex=False):
    pm = " $\\pm$ " if latex else " ± "
    out = pd.DataFrame({k: agg[k] for k in [*by, "method"]})
    for m in metrics:
        out[m] = [f"{mean:.3f}{pm}{std:.3f}" if pd.notna(std) else f"{mean:.3f}"
                  for mean, std in zip(agg[(m, "mean")], agg[(m, "std")])]
    out["runs"] = agg[(metrics[0], "count")].astype(int)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--by", nargs="*", default=[], help="extra grouping keys, e.g. snr")
    parser.add_argument("--tag", help="only rows with this tag")
    parser.add_argument("--latex", help="also write a LaTeX tabular to this path")
    args = parser.parse_args()

    df = load(args.results)
    if args.tag:
        df = df[df["tag"] == args.tag]
    agg, metrics = summarize(df, args.by)
    print(fmt(agg, metrics, args.by).to_markdown(index=False))
    if args.latex:
        table = fmt(agg, metrics, args.by, latex=True)
        with open(args.latex, "w") as f:
            f.write(table.to_latex(index=False, escape=False))


if __name__ == "__main__":
    main()
