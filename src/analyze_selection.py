"""
Label-free model selection on a weight sweep, and the resulting result tables.

For every (setting, seed), all runs share one held-out edge split, and every
choice is the candidate with the lowest held-out loss (best_val):

    nocd_single   NOCD, restart 0
    nocd_bestR    best of the R NOCD restarts
    markbp        best of {R - W NOCD restarts, MarkBP at each of W weights}
    placebo       same, with MarkBP on shuffled edge features

so every choice is made among the same number R of trained models.

No community labels are used for any choice; they only score the chosen model.
markbp - placebo isolates what the edge features contribute.

Usage:
    python src/analyze_selection.py results/synthetic_selection.jsonl --by snr \
        --latex paper/tables/synthetic.tex
"""

import argparse
import json

import numpy as np
import pandas as pd

COLUMNS = {"nocd_single": "NOCD", "nocd_best": "NOCD best-of-$R$",
           "markbp": "\\method", "placebo": "\\method, shuffled"}


HEADERS = {"snr": "Signal $s$"}


def pick(g: pd.DataFrame, metric: str) -> float:
    return g.loc[g["best_val"].idxmin(), metric] if len(g) else np.nan


def select(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    nocd = df["method"] == "nocd"
    df = df.copy()
    df.loc[nocd, "mark_weight"] = 0.0
    if "restart" not in df:
        df["restart"] = np.nan
    df.loc[nocd & df["restart"].isna(), "restart"] = 0
    rows = []
    for key, g in df.groupby([*by, "seed"]):
        key = key if isinstance(key, tuple) else (key,)
        restarts = g[g["method"] == "nocd"]
        mark_runs = g[g["method"] == "markbp"]
        placebo_runs = g[g["method"] == "markbp_shuffle"]
        # Budget matching: MarkBP gets as many NOCD restarts as leave R in total
        keep = restarts[restarts["restart"] < len(restarts) - len(mark_runs)]
        mark = pd.concat([keep, mark_runs])
        placebo = pd.concat([restarts[restarts["restart"] < len(restarts) - len(placebo_runs)],
                             placebo_runs])
        rows.append({**dict(zip([*by, "seed"], key)),
                     "nocd_single": restarts.loc[restarts["restart"] == 0, metric].iloc[0],
                     "nocd_best": pick(restarts, metric),
                     "markbp": pick(mark, metric),
                     "placebo": pick(placebo, metric),
                     "restarts": len(restarts),
                     "markbp_weight": pick(mark, "mark_weight"),
                     "placebo_weight": pick(placebo, "mark_weight")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--by", nargs="*", default=[])
    parser.add_argument("--metrics", nargs="+", default=["onmi", "edge_acc"])
    parser.add_argument("--latex", help="write a LaTeX table (mean +- std) here")
    args = parser.parse_args()

    df = pd.concat([pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
                    for p in args.results], ignore_index=True)
    blocks = []
    for metric in args.metrics:
        if metric not in df or df[metric].isna().all():
            continue
        res = select(df, args.by, metric)
        cols = list(COLUMNS)
        grouped = res.groupby(args.by) if args.by else res.assign(_all=0).groupby("_all")
        mean, std = grouped[cols].mean(), grouped[cols].std()
        print(f"\n{metric}: mean ± std over {res['seed'].nunique()} seeds "
              f"(R = {res['restarts'].iloc[0]} restarts)")
        print((mean.round(3).astype(str) + " ± " + std.round(3).astype(str)).to_markdown())
        print("mean chosen weight:", grouped["markbp_weight"].mean().round(3).to_dict())
        blocks.append((metric, mean, std))

    if args.latex:
        names = {"onmi": "ONMI", "edge_acc": "Edge assignment accuracy",
                 "f1": "F1", "jaccard": "Jaccard"}
        lines = ["\\begin{tabular}{l" + "c" * len(COLUMNS) + "}", "\\toprule",
                 " & ".join([" & ".join(HEADERS.get(b, b) for b in args.by),
                             *COLUMNS.values()]) + " \\\\"]
        for metric, mean, std in blocks:
            lines += ["\\midrule",
                      f"\\multicolumn{{{len(COLUMNS) + 1}}}{{l}}{{\\emph{{{names.get(metric, metric)}}}}} \\\\"]
            for idx in mean.index:
                # The placebo is a control, so it is never highlighted as best
                best = mean.loc[idx, [c for c in COLUMNS if c != "placebo"]].idxmax()
                cells = []
                for c in COLUMNS:
                    cell = f"{mean.loc[idx, c]:.3f} $\\pm$ {std.loc[idx, c]:.3f}"
                    cells.append(f"\\textbf{{{cell}}}" if c == best else cell)
                lines.append(f"{idx:g} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(args.latex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.latex}")


if __name__ == "__main__":
    main()
