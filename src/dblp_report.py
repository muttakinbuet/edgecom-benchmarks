"""
LaTeX tables for the DBLP-EF experiments.

* Main table: BigCLAM, NOCD (single run, best of R restarts), MarkBP and the
  shuffled placebo, all chosen by held-out edge loss, for title features and
  for the outside-venue leakage control.
* Weight table: every method at each fixed mark weight, to show the effect of
  edge features independently of model selection.

Usage:
    python src/dblp_report.py --subset data/subsets/cs_areas_le2006_main \
        --title results/dblp_cs_areas.jsonl \
        --outside results/dblp_cs_areas_outside.jsonl --out paper/tables
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .analyze_selection import select
except ImportError:  # Fallback for direct script execution
    from analyze_selection import select

METRICS = [("f1", "F1"), ("jaccard", "Jaccard"), ("onmi", "ONMI")]
# These runs used the vMF mark model (first MarkBP version) with threshold readout
METHOD_LABEL = r"\methodvmf"


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


def cell(mean, std, bold=False):
    text = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def main_table(bigclam, title, outside, edge_kmeans=None):
    nocd = title[title["method"] == "nocd"]
    placebo = title[title["method"] == "markbp_shuffle"]
    variants = {"Title": title,
                "Title, outside venues": pd.concat([nocd, outside, placebo])}
    rows = {"BigCLAM": {m: (bigclam[m]["mean"], bigclam[m]["std"]) for m, _ in METRICS}}
    for name, df in variants.items():
        sel = {m: select(df, [], m) for m, _ in METRICS}
        if name == "Title":
            r = sel["f1"]["restarts"].iloc[0]
            for col, label in (("nocd_single", "NOCD"), ("nocd_best", f"NOCD best-of-{r}")):
                rows[label] = {m: (sel[m][col].mean(), sel[m][col].std()) for m, _ in METRICS}
            rows[f"{METHOD_LABEL}, shuffled"] = {
                m: (sel[m]["placebo"].mean(), sel[m]["placebo"].std()) for m, _ in METRICS}
        rows[f"{METHOD_LABEL} ({name.lower()})"] = {
            m: (sel[m]["markbp"].mean(), sel[m]["markbp"].std()) for m, _ in METRICS}

    order = ["BigCLAM", "NOCD", next(k for k in rows if k.startswith("NOCD best"))]
    if edge_kmeans is not None:
        for stem, label in (("edge_feats_title", "Edge-feature clustering (title)"),
                            ("edge_feats_title_outside",
                             "Edge-feature clustering (title, outside venues)")):
            g = edge_kmeans[edge_kmeans["features"] == stem]
            rows[label] = {m: (g[m].mean(), g[m].std()) for m, _ in METRICS}
            order.append(label)
    order += sorted((k for k in rows if k.startswith(METHOD_LABEL)),
                    key=lambda k: "shuffled" in k)  # placebo last
    best = {m: max((rows[k][m][0], k) for k in order if "shuffled" not in k)[1]
            for m, _ in METRICS}
    lines = ["\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
             "Method & " + " & ".join(label for _, label in METRICS) + " \\\\", "\\midrule"]
    for k in order:
        lines.append(k + " & " + " & ".join(cell(*rows[k][m], bold=best[m] == k)
                                             for m, _ in METRICS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def weight_table(title, outside):
    df = pd.concat([title, outside.assign(method="markbp_outside")])
    df.loc[df["method"] == "nocd", "mark_weight"] = 0.0
    names = {"markbp": "Title", "markbp_outside": "Title, outside venues",
             "markbp_shuffle": "Shuffled"}
    nocd = df[df["method"] == "nocd"]
    weights = sorted(df.loc[df["method"] != "nocd", "mark_weight"].unique())
    lines = ["\\begin{tabular}{l" + "c" * (len(weights) + 1) + "}", "\\toprule",
             "Edge features & $\\lambda = 0$ (NOCD) & "
             + " & ".join(f"$\\lambda = {w:g}$" for w in weights) + " \\\\", "\\midrule"]
    for method, label in names.items():
        g = df[df["method"] == method].groupby("mark_weight")["f1"]
        cells = [f"{nocd['f1'].mean():.3f}"] + [f"{g.mean()[w]:.3f}" for w in weights]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--current", help="dblp_markbp.py results: write the current-method "
                                          "tables and exit")
    parser.add_argument("--title")
    parser.add_argument("--outside")
    parser.add_argument("--edge-kmeans", help="results of edge_cluster_baseline.py")
    parser.add_argument("--auto", help="end-to-end MarkBP-auto results: add its rows "
                                       "to the current tables")
    parser.add_argument("--out", default="paper/tables")
    args = parser.parse_args()
    if args.current:
        write_current(args.subset, args.current, args.out, args.auto)
        return

    with open(Path(args.subset) / "results_bigclam.json") as f:
        bigclam = json.load(f)["summary"]
    title, outside = load(args.title), load(args.outside)
    # Only seeds whose full candidate set (NOCD restarts, MarkBP, placebo) is done
    complete = (set(title.loc[title["method"] == "markbp", "seed"])
                & set(title.loc[title["method"] == "markbp_shuffle", "seed"]))
    title = title[title["seed"].isin(complete)]
    outside = outside[outside["seed"].isin(complete)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    edge_kmeans = load(args.edge_kmeans) if args.edge_kmeans else None
    (out / "dblp_main.tex").write_text(main_table(bigclam, title, outside, edge_kmeans))
    (out / "dblp_weights.tex").write_text(weight_table(title, outside))
    print((out / "dblp_main.tex").read_text())
    print((out / "dblp_weights.tex").read_text())



# ---------------------------------------------------------------------------
# Current method (coupled MarkBP), from dblp_markbp.py results
# ---------------------------------------------------------------------------
FEATURE_SETS = [("edge_feats_title", "Title"),
                ("edge_feats_title_abstract", "Title + abstract"),
                ("edge_feats_title_outside", "Title, outside"),
                ("edge_feats_title_abstract_outside", "Title + abs., outside")]
MARKBP_ROWS = [("edge_kmeans", "edges", "Edge-feature clustering"),
               ("markbp_struct", "threshold", r"\method, structure-led (threshold)"),
               ("markbp_struct", "edges", r"\method, structure-led (edge readout)"),
               ("markbp_feat", "threshold", r"\method, feature-led (threshold)"),
               ("markbp_feat", "edges", r"\method, feature-led (edge readout)"),
               ("markbp_feat_shuffle", "threshold", r"\method, feature-led, shuffled")]


def features_table(df, bigclam, metric="f1"):
    """Rows: methods; columns: edge-feature sets; cells: mean +- std over seeds."""
    ncol = len(FEATURE_SETS)
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             "Method & " + " & ".join(label for _, label in FEATURE_SETS) + r" \\",
             r"\midrule"]
    nocd = df[df["method"] == "nocd"]
    for label, mean, std in (
            ("BigCLAM", bigclam[metric]["mean"], bigclam[metric]["std"]),
            ("NOCD (threshold)", *nocd[nocd["readout"] == "threshold"][metric].agg(["mean", "std"])),
            ("NOCD (edge readout)", *nocd[nocd["readout"] == "edges"][metric].agg(["mean", "std"]))):
        lines.append(rf"{label} & \multicolumn{{{ncol}}}{{c}}{{{cell(mean, std)} "
                     r"(no edge features)} \\")
    lines.append(r"\midrule")
    cells = {}
    for method, readout, label in MARKBP_ROWS:
        for stem, _ in FEATURE_SETS:
            g = df[(df["method"] == method) & (df["readout"] == readout)
                   & (df["features"] == stem)][metric]
            cells[(label, stem)] = (g.mean(), g.std()) if len(g) else None
    for stem, _ in FEATURE_SETS:  # best non-placebo entry per column in bold
        best = max((cells[(lab, stem)][0], lab) for _, _, lab in MARKBP_ROWS
                   if "shuffled" not in lab and cells[(lab, stem)] is not None)
        cells[(best[1], stem)] = (*cells[(best[1], stem)], True)
    for _, _, label in MARKBP_ROWS:
        row = []
        for stem, _ in FEATURE_SETS:
            c = cells[(label, stem)]
            row.append("--" if c is None else cell(c[0], c[1], bold=len(c) == 3))
        lines.append(label + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


AUTO_ROWS = [("markbp_auto", "threshold", r"\methodauto (threshold)"),
             ("markbp_auto", "edges", r"\methodauto (edge readout)"),
             ("markbp_auto_shuffle", "threshold", r"\methodauto, shuffled")]


def write_current(subset, markbp_path, out, auto_results=None):
    with open(Path(subset) / "results_bigclam.json") as f:
        bigclam = json.load(f)["summary"]
    df = load(markbp_path)
    if auto_results:
        # End-to-end MarkBP-auto runs (dblp_markbp.py --methods markbp_auto ...)
        df = pd.concat([df, load(auto_results)], ignore_index=True)
        for row in AUTO_ROWS:
            if row not in MARKBP_ROWS:
                MARKBP_ROWS.append(row)
    out = Path(out)
    for metric in ("f1", "jaccard", "onmi"):
        (out / f"dblp_current_{metric}.tex").write_text(features_table(df, bigclam, metric))
    print((out / "dblp_current_f1.tex").read_text())




if __name__ == "__main__":
    main()
