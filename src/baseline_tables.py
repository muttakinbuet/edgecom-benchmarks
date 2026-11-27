"""
LaTeX tables comparing MarkBP-auto with the published baselines.

* poc_baselines.tex: controlled graphs (degree 8, 20 seeds), ONMI per setting.
* dblp_baselines_{f1,jaccard,onmi}.tex: DBLP-EF CS-Areas, mean +- std over
  ten seeds, one column per edge-feature set; dblp_baselines_<subset>_*.tex
  the same for the other subsets (five seeds).
* dblp_subsets_{f1,jaccard,onmi}.tex: all subsets, mean over seeds and feature
  sets. A star marks the better of MarkBP-auto + input features and the
  strongest baseline when a paired Wilcoxon test over (seed, feature set)
  pairs gives p < 0.05.

Restarts: MarkBP-auto + input features and the two strongest baselines (NOCD and
DynaResGCN with averaged features) are also fitted with three restarts
(results/restarts/). For every (seed, feature set) the restart with the lowest
held-out loss is kept, without labels; these runs form the "best of 3 restarts"
rows, and the star test is repeated among them.

MarkBP-auto + input features comes from the *_ours_l01 files (mark weight 0.1,
the current default); the ablation without input features keeps its runs at
mark weight 1, and the baselines come from the main result files.

Readout: each baseline is scored with whichever of its readouts (threshold,
edge or their union) is better on average over the whole benchmark, a single
choice per method and benchmark. This favours the baselines. MarkBP-auto keeps
the union readout of the method section on both benchmarks. Link communities
has only its own readout.

Usage:
    python src/baseline_tables.py --out paper/tables
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

FEATURE_SETS = [("edge_feats_title", "Title"),
                ("edge_feats_title_abstract", "Title + abstract"),
                ("edge_feats_title_outside", "Title, outside"),
                ("edge_feats_title_abstract_outside", "Title + abs., outside")]
# (method, label, uses edge features)
POC_ROWS = [("nocd", "NOCD", False), ("nocd_gat", "NOCD-GAT", False),
            ("link_comm", "Link comm.", False), ("dynares", "DynaResGCN", False),
            ("nocd_gine", "NOCD-GINE", True), ("nocd_eagg", "NOCD + avg.", True),
            ("dynares_eagg", "DynaResGCN + avg.", True),
            ("cesna", "CESNA", True), ("edge_kmeans", "Edge clust.", True)]
DBLP_ROWS = [("nocd", "NOCD"), ("nocd_gat", "NOCD-GAT"), ("link_comm", "Link communities"),
             ("dynares", "DynaResGCN"),
             ("nocd_gine", "NOCD-GINE"), ("nocd_eagg", "NOCD + averaged features"),
             ("nocd_eagg_shuffle", "NOCD + averaged features, shuffled"),
             ("dynares_eagg", "DynaResGCN + averaged features"),
             ("cesna", "CESNA"), ("edge_kmeans", "Edge-feature clustering")]
OURS = [("markbp_auto", r"\methodauto, no input features ($\lambda = 1$)"),
        ("markbp_autoin", r"\methodauto + input features")]
OUR_READOUT = "union"
# (baseline results, MarkBP-auto results, subset label, table suffix)
SUBSETS = [("results/dblp_cs_areas_10seeds.jsonl",
            "results/dblp_cs_areas_le2006_main_ours_l01.jsonl", r"\textsc{CS-Areas}", ""),
           ("results/dblp_cs_systems_le2006_main.jsonl",
            "results/dblp_cs_systems_le2006_main_ours_l01.jsonl", r"\textsc{CS-Systems}",
            "_cs_systems"),
           ("results/dblp_rand30_s1.jsonl",
            "results/dblp_rand30_s1_ours_l01.jsonl", r"\textsc{Random-30}", "_rand30")]
# Best of three restarts by held-out loss (method -> its restart row)
RESTARTED = [("nocd_eagg", "NOCD + averaged features"),
             ("dynares_eagg", "DynaResGCN + averaged features"),
             ("markbp_autoin", r"\methodauto + input features")]
R_SUFFIX = "_r3"
RESTART_LABEL = ", best of 3 restarts"
RESTART_FILES = {"": "results/restarts/cs_areas.jsonl",
                 "_cs_systems": "results/restarts/cs_systems.jsonl",
                 "_rand30": "results/restarts/rand30.jsonl"}
OURS_SHUFFLED = [("markbp_auto_shuffle", r"\methodauto, no input features, shuffled"),
                 ("markbp_autoin_shuffle", r"\methodauto + input features, shuffled")]


def load(*paths):
    frames = [pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
              for p in paths if Path(p).exists()]
    return pd.concat(frames, ignore_index=True)


def best_of_restarts(path, n_restarts=3):
    """Runs of the restarted methods, keeping per (seed, method, features) the
    restart with the lowest held-out loss (ties: the lowest restart index).
    MarkBP-auto selects by the joint fit's loss; when the tests choose NOCD or
    edge clustering, every restart gives the same result and restart 0 is kept.
    Methods are renamed <method>_r3. Groups with missing restarts are dropped."""
    if not Path(path).exists():
        return pd.DataFrame()
    df = load(path)
    df = df[df["method"].isin([m for m, _ in RESTARTED])].copy()
    if "restart" not in df:
        df["restart"] = 0
    df["restart"] = df["restart"].fillna(0).astype(int)
    loss = df["val"] if "val" in df else pd.Series(float("nan"), index=df.index)
    if "joint_val" in df:
        loss = loss.where(df["method"] != "markbp_autoin", df["joint_val"])
    df["loss"] = loss.fillna(0.0)
    keys = ["seed", "method", "features"]
    runs = df.drop_duplicates(keys + ["restart"])
    complete = runs.groupby(keys)["restart"].transform("nunique") >= n_restarts
    runs = runs[complete].sort_values(keys + ["loss", "restart"])
    chosen = runs.groupby(keys, as_index=False).first()[keys + ["restart"]]
    out = df.merge(chosen, on=keys + ["restart"])
    out["method"] = out["method"] + R_SUFFIX
    return out.drop(columns=["loss"])


def best_readout(df, method, metric, placebo_of=None):
    """The method's readout with the higher benchmark-wide mean (placebos follow their method)."""
    g = df[df["method"] == (placebo_of or method)]
    if g.empty:
        return None
    return g.groupby("readout")[metric].mean().idxmax()


def poc_table(df, metric="onmi"):
    keys = ["noise", "snr"]
    cols = {}
    for method, label, _ in POC_ROWS:
        readout = best_readout(df, method, metric)
        g = df[(df["method"] == method) & (df["readout"] == readout)]
        cols[label] = g.groupby(keys)[metric].mean()
    for method, label in [("markbp_auto", "Ours, no inputs ($\\lambda{=}1$)"),
                          ("markbp_autoin", "Ours + inputs")]:
        g = df[(df["method"] == method) & (df["readout"] == OUR_READOUT)]
        if len(g):
            cols[label] = g.groupby(keys)[metric].mean()
    t = pd.DataFrame(cols)
    labels = list(t.columns)
    header = " & ".join(l.replace("Ours", r"\methodauto") for l in labels)
    lines = [r"\begin{tabular}{cc " + "c" * len(labels) + "}", r"\toprule",
             r"Background & $s$ & " + header + r" \\", r"\midrule"]
    for (noise, snr), row in t.iterrows():
        best = row.max()
        cells = [f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}" for v in row]
        lines.append(f"{int(noise * 100)}\\% & {snr:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{2}{c}{Mean} & "
              + " & ".join(f"{v:.3f}" for v in t.mean()) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", t


def cell(g, bold=False):
    s = f"{g.mean():.3f} $\\pm$ {g.std():.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def dblp_table(df, metric="f1"):
    ncol = len(FEATURE_SETS)
    rows = []  # (label, {stem: series} or series for structure-only, placebo)
    for method, label in DBLP_ROWS:
        placebo_of = method.removesuffix("_shuffle") if method.endswith("_shuffle") else None
        readout = best_readout(df, method, metric, placebo_of)
        if readout is None:
            continue
        g = df[(df["method"] == method) & (df["readout"] == readout)]
        if g.empty:
            continue
        if g["features"].isna().all():
            rows.append((label, g[metric], False))
        else:
            rows.append((label, {s: g[g["features"] == s][metric] for s, _ in FEATURE_SETS},
                         placebo_of is not None))
    ours = []
    for method, label in OURS + OURS_SHUFFLED:
        g = df[(df["method"] == method) & (df["readout"] == OUR_READOUT)]
        if len(g):
            ours.append((label, {s: g[g["features"] == s][metric] for s, _ in FEATURE_SETS},
                         method.endswith("_shuffle")))
    restarted = []
    for method, label in RESTARTED:
        m = method + R_SUFFIX
        readout = OUR_READOUT if method == "markbp_autoin" else best_readout(df, m, metric)
        g = df[(df["method"] == m) & (df["readout"] == readout)] if readout else df.iloc[:0]
        if len(g):
            restarted.append((label + RESTART_LABEL,
                              {s: g[g["features"] == s][metric] for s, _ in FEATURE_SETS},
                              False))
    best = {s: max(r[1][s].mean() for r in rows + ours + restarted
                   if isinstance(r[1], dict) and not r[2] and len(r[1][s]))
            for s, _ in FEATURE_SETS}

    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             "Method & " + " & ".join(label for _, label in FEATURE_SETS) + r" \\",
             r"\midrule"]

    def emit(label, values, placebo):
        if not isinstance(values, dict):
            lines.append(rf"{label} & \multicolumn{{{ncol}}}{{c}}{{{cell(values)} "
                         r"(no edge features)} \\")
            return
        cells = []
        for s, _ in FEATURE_SETS:
            g = values[s]
            cells.append("--" if g.empty else
                         cell(g, bold=not placebo and abs(g.mean() - best[s]) < 1e-12))
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    for i, row in enumerate(rows):
        if i and not isinstance(row[1], pd.Series) and isinstance(rows[i - 1][1], pd.Series):
            lines.append(r"\midrule")
        emit(*row)
    lines.append(r"\midrule")
    for row in ours:
        emit(*row)
    if restarted:
        lines.append(r"\midrule")
        for row in restarted:
            emit(*row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def with_ours(baselines, ours):
    """Baseline runs plus the current MarkBP-auto + input features runs."""
    keep = ~baselines["method"].str.startswith("markbp_autoin")
    ours = ours[ours["method"].str.startswith("markbp_autoin")]
    return pd.concat([baselines[keep], ours], ignore_index=True)


def scored(df, method, metric):
    """The method's runs at its table readout, indexed by (seed, feature set)."""
    if method in dict(OURS + OURS_SHUFFLED) or method == "markbp_autoin" + R_SUFFIX:
        readout = OUR_READOUT
    else:
        readout = best_readout(df, method, metric,
                               method.removesuffix("_shuffle") if method.endswith("_shuffle")
                               else None)
    g = df[(df["method"] == method) & (df["readout"] == readout)]
    return g.set_index(["seed", "features"])[metric]


def subsets_table(frames, metric="f1"):
    """One column per subset: mean over seeds and feature sets."""
    labels = [label for *_, label, _ in SUBSETS]
    restarted = [(m + R_SUFFIX, label + RESTART_LABEL) for m, label in RESTARTED]
    methods = DBLP_ROWS + OURS + OURS_SHUFFLED + restarted
    means = pd.DataFrame({label: {m: scored(df, m, metric).mean() for m, _ in methods}
                          for label, df in zip(labels, frames)})
    star = {}  # (column, method) pairs that win a paired test
    tests = []
    single_baselines = [m for m, _ in DBLP_ROWS if not m.endswith("_shuffle")]
    restart_baselines = [m for m, _ in restarted if m != "markbp_autoin" + R_SUFFIX]
    for label, df in zip(labels, frames):
        for ours_m, pool in [("markbp_autoin", single_baselines),
                             ("markbp_autoin" + R_SUFFIX, restart_baselines)]:
            pool = [m for m in pool if pd.notna(means.loc[m, label])]
            if not pool or pd.isna(means.loc[ours_m, label]):
                continue
            rival = max(pool, key=lambda m: means.loc[m, label])
            # Seeds are the unit: average each method over feature sets per seed
            ours = scored(df, ours_m, metric).groupby("seed").mean()
            theirs = scored(df, rival, metric).groupby("seed").mean()
            seeds = ours.index.intersection(theirs.index)
            if len(seeds) > 1:
                tests.append(((label, ours_m, rival),
                              ttest_rel(ours[seeds], theirs[seeds]).pvalue,
                              (ours[seeds] - theirs[seeds]).mean()))
    # Holm correction over the comparisons in this table
    tests.sort(key=lambda t: t[1])
    running = 0.0
    for rank, ((label, ours_m, rival), p, diff) in enumerate(tests):
        running = max(running, (len(tests) - rank) * p)
        if running < 0.05:
            star[label, ours_m if diff > 0 else rival] = True
    best = {label: means.loc[[m for m, _ in methods if not m.endswith("_shuffle")], label].max()
            for label in labels}

    lines = [r"\begin{tabular}{l" + "c" * len(labels) + "}", r"\toprule",
             "Method & " + " & ".join(labels) + r" \\", r"\midrule"]
    for method, label in methods:
        if method in (OURS[0][0], restarted[0][0]):
            if method == restarted[0][0] and means.loc[method].isna().all():
                break
            lines.append(r"\midrule")
        cells = []
        for col in labels:
            v = means.loc[method, col]
            if pd.isna(v):
                cells.append("--")
                continue
            text = f"{v:.3f}" + ("$^{*}$" if star.get((col, method)) else "")
            if not method.endswith("_shuffle") and abs(v - best[col]) < 1e-12:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", means


VARIANTS = [("base", r"Feature-led, $\lambda = 1$ (earlier default)"),
            ("feat_l03", r"Feature-led, $\lambda = 0.3$"),
            ("feat_l01", r"Feature-led, $\lambda = 0.1$ (default)"),
            ("l0", r"No marks, $\lambda = 0$ (NOCD + averaged features)"),
            ("struct_first", r"Structure first, then marks, $\lambda = 1$"),
            ("struct_first_l03", r"Structure first, then marks, $\lambda = 0.3$"),
            ("uniform", r"Structure-led, $\lambda = 1$"),
            ("uniform_l03", r"Structure-led, $\lambda = 0.3$"),
            ("uniform_m2", r"Structure-led, $M = 2K$"),
            ("uniform_m4", r"Structure-led, $M = 4K$")]


def variant_table(path="results/variant_screen_cs_areas.jsonl"):
    """Screen of MarkBP branch variants (CS-Areas, 3 seeds x 2 feature sets)."""
    df = load(path)
    g = df[df["readout"] == OUR_READOUT].groupby("variant")[["f1", "jaccard", "onmi",
                                                             "best_val"]].mean()
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant & F1 & Jaccard & ONMI & Held-out loss \\", r"\midrule"]
    for variant, label in VARIANTS:
        if variant in g.index:
            r = g.loc[variant]
            lines.append(f"{label} & {r.f1:.3f} & {r.jaccard:.3f} & {r.onmi:.3f} & "
                         f"{r.best_val:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/tables")
    args = parser.parse_args()
    out = Path(args.out)

    poc = with_ours(load("results/poc_union.jsonl"), load("results/poc_ours_l01.jsonl"))
    tex, t = poc_table(poc)
    (out / "poc_baselines.tex").write_text(tex)
    print(t.round(3).to_string())
    print(t.mean().round(3).to_string())

    (out / "variant_screen.tex").write_text(variant_table())

    frames = []
    for path, ours_path, _, suffix in SUBSETS:
        dblp = with_ours(load(path), load(ours_path))
        if suffix == "":
            # Link communities is deterministic; its three seeds give identical results
            old = load("results/dblp_baselines.jsonl")
            dblp = pd.concat([dblp, old[old["method"] == "link_comm"]], ignore_index=True)
        dblp = pd.concat([dblp, best_of_restarts(RESTART_FILES[suffix])],
                         ignore_index=True)
        frames.append(dblp)
        for metric in ("f1", "jaccard", "onmi"):
            (out / f"dblp_baselines{suffix}_{metric}.tex").write_text(dblp_table(dblp, metric))
    for metric in ("f1", "jaccard", "onmi"):
        tex, means = subsets_table(frames, metric)
        (out / f"dblp_subsets_{metric}.tex").write_text(tex)
        print(metric)
        print(means.round(3).to_string())


if __name__ == "__main__":
    main()
