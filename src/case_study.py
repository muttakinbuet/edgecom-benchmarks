"""
Case study on DBLP-EF CS-Areas: which community does each co-authorship edge
get, and does it match the venue of the papers behind the edge?

Stage "fit" trains MarkBP-auto + input features and NOCD + averaged features
(seed 0, title + abstract features; the same settings as the main comparison)
and caches F and the edge assignments. Stage "analyse" names every predicted
community by its best-matching venue (F1 against the venue's members), scores
edge assignments against the venues of each edge's papers, picks a case-study
author by a fixed rule and writes the figure.

Usage:
    python src/case_study.py --stage fit --method ours
    python src/case_study.py --stage fit --method nocd_eagg
    python src/case_study.py --stage analyse
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

logger = logging.getLogger(__name__)

SUBSET = Path("data/subsets/cs_areas_le2006_main")
PAPERS = Path("data/com_dblp_ef_conf_le2006_venue_main/papers.jsonl")
FEATURES = "edge_feats_title_abstract"
SEED = 0
CACHE = Path("data/cache/case_study")


def cache_file(method, seed):
    return CACHE / (f"{method}.npz" if seed == SEED else f"{method}_s{seed}.npz")


def fit(method, seed=SEED):
    import torch
    torch.set_num_threads(1)
    from markbp_auto import MarkBPAuto
    from ocd_models import OverlappingCommunityModel
    from poc import edge_readout
    from run_experiment import edge_feature_node_mean
    from subset import load_subset

    n, ei, truth = load_subset(SUBSET)
    k = len(truth)
    feats = np.load(SUBSET / f"{FEATURES}.npy")
    if method == "ours":
        model = MarkBPAuto(k, seed=seed, cache_dir="data/cache/nocd_dblp", input_features=True)
        model.fit(n, ei, edge_features=feats)
        logger.info(f"choice {model.choice}")
    else:
        model = OverlappingCommunityModel(k, seed=seed, refit=False, encoder="gcn")
        model.fit(n, ei, input_edge_features=feats)
    labels = model.edge_assignments(ei)
    threshold = model.F > model.threshold
    member = threshold | edge_readout(n, ei, labels, k)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file(method, seed), F=model.F, labels=labels,
                        threshold=threshold, union=member)
    logger.info(f"saved {method}")


def analyse(out_dir):
    from sklearn.metrics import normalized_mutual_info_score
    from dblp_markbp import score
    from subset import load_subset

    n, ei, truth = load_subset(SUBSET)
    k = len(truth)
    meta = pd.read_csv(SUBSET / "communities.meta.tsv", sep="\t")
    venues = meta["venue_name"].str.replace(" Conference", "").tolist()
    venue_index = {key: i for i, key in enumerate(meta["venue_key"])}
    nodes = pd.read_csv(SUBSET / "nodes.tsv", sep="\t")
    edges = pd.read_csv(SUBSET / "edges.tsv", sep="\t")

    paper_ids = {int(p) for ids in edges["paper_ids"] for p in str(ids).split(";")}
    paper_venue, paper_title = {}, {}
    with open(PAPERS, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["paper_id"] in paper_ids:
                paper_venue[p["paper_id"]] = venue_index.get(p["venue_key"], -1)
                paper_title[p["paper_id"]] = p["title"]
    # Venues (within the subset) of the papers behind each edge
    edge_venues = [{paper_venue[int(p)] for p in str(ids).split(";")} - {-1}
                   for ids in edges["paper_ids"]]
    has_venue = np.array([len(v) > 0 for v in edge_venues])
    single = np.array([len(v) == 1 for v in edge_venues])
    single_venue = np.array([min(v) if len(v) == 1 else -1 for v in edge_venues])

    truth_member = np.zeros((n, k), dtype=bool)
    for c, m in enumerate(truth):
        truth_member[m, c] = True
    overlapping = truth_member.sum(axis=1) > 1

    results = {"edges": len(edge_venues), "edges_with_subset_venue": int(has_venue.sum())}
    fits = {}
    for method in ("ours", "nocd_eagg"):
        d = np.load(CACHE / f"{method}.npz")
        member, labels = d["union" if method == "ours" else "threshold"], d["labels"]
        # Name each predicted community by its best-matching venue (F1)
        inter = member.T.astype(float) @ truth_member.astype(float)          # K x K
        f1 = 2 * inter / (member.sum(0)[:, None] + truth_member.sum(0)[None, :] + 1e-9)
        name = f1.argmax(axis=1)
        correct = np.array([name[labels[e]] in edge_venues[e] for e in range(len(labels))])
        touches_overlap = overlapping[ei[0]] | overlapping[ei[1]]
        results[method] = {
            **{m: round(float(v), 4) for m, v in score(member, truth, n).items()},
            "edge_accuracy": round(float(correct[has_venue].mean()), 4),
            "edge_accuracy_overlapping": round(float(correct[has_venue & touches_overlap].mean()), 4),
            "distinct_venue_names": int(len(set(name[member.any(0)]))),
            # Mapping-free agreement of edge assignments with the edge's venue
            "edge_nmi_single_venue": round(float(normalized_mutual_info_score(
                single_venue[single], labels[single])), 4),
        }
        fits[method] = (member, labels, name, correct)
        logger.info(f"{method}: {results[method]}")

    # Case-study author, fixed rule: among authors with 15-30 co-authors whose
    # edges' papers span at least 4 subset venues, the one with most venues;
    # ties broken by degree, then node id. Chosen without looking at predictions.
    deg = np.bincount(ei.ravel(), minlength=n)
    inc = [[] for _ in range(n)]
    for e, (u, v) in enumerate(ei.T):
        inc[u].append(e)
        inc[v].append(e)
    def span(u):
        return len(set().union(*(edge_venues[e] for e in inc[u]))) if inc[u] else 0
    candidates = [u for u in range(n) if 15 <= deg[u] <= 30]
    author = max(candidates, key=lambda u: (span(u), deg[u], -u))
    results["author"] = {"node": int(author), "name": nodes["name"][author],
                         "degree": int(deg[author]), "venues": span(author)}
    logger.info(f"case-study author: {results['author']}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "case_study.json", "w") as f:
        json.dump(results, f, indent=1)

    # Edge table of the author, for the figure and for checking
    rows = []
    for e in inc[author]:
        other = int(ei[1, e] if ei[0, e] == author else ei[0, e])
        ids = [int(p) for p in str(edges["paper_ids"][e]).split(";")]
        rows.append({"edge": e, "coauthor": nodes["name"][other],
                     "true": sorted(venues[c] for c in edge_venues[e]),
                     **{f"{m}": venues[fits[m][2][fits[m][1][e]]] for m in fits},
                     **{f"{m}_correct": bool(fits[m][3][e]) for m in fits},
                     "titles": [paper_title[p] for p in ids][:3]})
    pd.DataFrame(rows).to_json(out_dir / "case_study_author.json", orient="records", indent=1)
    return results, rows, inc[author], author, venues, edge_venues, fits, ei, nodes


AREAS = [("DB", ["SIGMOD", "VLDB", "ICDE", "EDBT", "PODS"]),
         ("DM", ["KDD", "ICDM", "SDM", "PKDD", "CIKM"]),
         ("IR/Web", ["SIGIR", "WWW"]),
         ("ML", ["ICML", "NIPS", "COLT", "UAI"]),
         ("AI", ["AAAI", "IJCAI"]),
         ("NLP", ["ACL", "COLING", "EMNLP"]),
         ("CV", ["CVPR", "ICCV", "ECCV"]),
         ("Theory", ["STOC", "FOCS", "SODA"])]
# Reference categorical palette, fixed order (light mode, for print)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"


def composition(labels, edge_venues, order, k):
    """Community x venue share of each community's single-venue edges."""
    single = np.array([len(v) == 1 for v in edge_venues])
    venue = np.array([min(v) if len(v) == 1 else -1 for v in edge_venues])
    m = np.zeros((k, k))
    np.add.at(m, (labels[single], venue[single]), 1)
    m = m[:, order]
    used = m.sum(1) > 0
    share = m[used] / m[used].sum(1, keepdims=True)
    eff = np.exp(-(np.where(share > 0, share * np.log(share), 0)).sum(1))  # effective venues
    rows = np.lexsort((-share.max(1), share.argmax(1)))
    return share[rows], eff[rows], m[used].sum(1)[rows]


def plot(out_dir, fig_path):
    import matplotlib
    matplotlib.use("pdf")  # Agg is blocked by application control on this machine
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    results, rows, inc, author, venues, edge_venues, fits, ei, nodes = analyse(out_dir)
    k = len(venues)
    order = [venues.index(v) for _, vs in AREAS for v in vs]
    assert sorted(order) == list(range(k)), "AREAS must list every venue once"
    names = {"ours": "MarkBP-auto + input features", "nocd_eagg": "NOCD + averaged features"}

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.28)
    summary = {}
    for i, m in enumerate(("ours", "nocd_eagg")):
        share, eff, size = composition(fits[m][1], edge_venues, order, k)
        summary[m] = {"communities": int(len(share)),
                      "mean_effective_venues": round(float(np.average(eff, weights=None)), 2),
                      "edge_weighted_effective_venues": round(float(np.average(eff, weights=size)), 2)}
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(share, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(k), [venues[j] for j in order], rotation=90, fontsize=5)
        ax.set_yticks(range(len(share)), [venues[order[j]] for j in share.argmax(1)], fontsize=5)
        ax.set_ylabel("community (named by its main venue)")
        start = 0
        for area, vs in AREAS:  # area separators and labels
            if start:
                ax.axvline(start - 0.5, color="white", lw=1.2)
            ax.text(start + len(vs) / 2 - 0.5, -1.2, area, ha="center", va="bottom",
                    fontsize=5.5, color=MUTED)
            start += len(vs)
        ax.set_title(f"({'ab'[i]}) {names[m]}\n"
                     f"{summary[m]['mean_effective_venues']:.1f} effective venues "
                     "per community", fontsize=7, pad=14)
        ax.tick_params(length=0)

    # Ego network of the case-study author, same layout in both panels
    g = nx.Graph()
    for e in inc:
        g.add_edge(int(ei[0, e]), int(ei[1, e]), e=e)
    ego = set(g)
    among = [(int(u), int(v)) for u, v in ei.T if u != author and v != author
             and u in ego and v in ego]  # co-author pairs who also co-authored
    g.add_edges_from(among)
    pos = nx.spring_layout(g, seed=1)
    used = []
    for m in ("ours", "nocd_eagg"):
        for e in inc:
            name = venues[fits[m][2][fits[m][1][e]]]
            if name not in used:
                used.append(name)
    colour = {v: SERIES[i] if i < len(SERIES) else MUTED for i, v in enumerate(used)}
    for i, m in enumerate(("ours", "nocd_eagg")):
        ax = fig.add_subplot(gs[1, i])
        labels = fits[m][1]
        for u, v in among:
            ax.plot(*zip(pos[u], pos[v]), color=GRID, lw=0.7, zorder=0)
        for e in inc:
            u, v = int(ei[0, e]), int(ei[1, e])
            name = venues[fits[m][2][labels[e]]]
            ax.plot(*zip(pos[u], pos[v]), color=colour[name], lw=2, solid_capstyle="round", zorder=1)
        others = [u for u in g if u != author]
        ax.scatter(*np.array([pos[u] for u in others]).T, s=14, color="white",
                   edgecolor=MUTED, linewidth=0.8, zorder=2)
        ax.scatter(*pos[author], s=60, color=INK, zorder=3)
        ax.annotate(nodes["name"][author], pos[author], xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=6.5, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        present = sorted({venues[fits[m][2][labels[e]]] for e in inc}, key=used.index)
        counts = {v: sum(venues[fits[m][2][labels[e]]] == v for e in inc) for v in present}
        ax.legend([Line2D([], [], color=colour[v], lw=2) for v in present],
                  [f"{v}-like ({counts[v]})" for v in present], loc="upper center",
                  bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=6,
                  handlelength=1.5, columnspacing=1.2)
        ax.set_title(f"({'cd'[i]}) {names[m]}\n{nodes['name'][author]}'s "
                     f"{len(inc)} co-author edges by community", fontsize=7)
        ax.set_axis_off()
    fig.savefig(fig_path, bbox_inches="tight")
    results["composition"] = summary
    with open(Path(out_dir) / "case_study.json", "w") as f:
        json.dump(results, f, indent=1)
    logger.info(f"composition: {summary}")


def seeds(out_dir, seed_list):
    """Effective venues, edge NMI, F1 and ONMI of both methods per seed."""
    from sklearn.metrics import normalized_mutual_info_score
    from dblp_markbp import score
    from subset import load_subset

    n, ei, truth = load_subset(SUBSET)
    k = len(truth)
    meta = pd.read_csv(SUBSET / "communities.meta.tsv", sep="\t")
    venue_index = {key: i for i, key in enumerate(meta["venue_key"])}
    edges = pd.read_csv(SUBSET / "edges.tsv", sep="\t")
    paper_ids = {int(p) for ids in edges["paper_ids"] for p in str(ids).split(";")}
    paper_venue = {}
    with open(PAPERS, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["paper_id"] in paper_ids:
                paper_venue[p["paper_id"]] = venue_index.get(p["venue_key"], -1)
    edge_venues = [{paper_venue[int(p)] for p in str(ids).split(";")} - {-1}
                   for ids in edges["paper_ids"]]
    single = np.array([len(v) == 1 for v in edge_venues])
    single_venue = np.array([min(v) if len(v) == 1 else -1 for v in edge_venues])
    rows = []
    for seed in seed_list:
        for method in ("ours", "nocd_eagg"):
            d = np.load(cache_file(method, seed))
            member = d["union" if method == "ours" else "threshold"]
            share, eff, size = composition(d["labels"], edge_venues, list(range(k)), k)
            s = score(member, truth, n)
            rows.append({"seed": seed, "method": method, "f1": round(s["f1"], 4),
                         "onmi": round(s["onmi"], 4),
                         "effective_venues": round(float(np.average(eff, weights=size)), 2),
                         "effective_venues_unweighted": round(float(eff.mean()), 2),
                         "max_venue_share_median": round(float(np.median(share.max(1))), 3),
                         "edge_nmi": round(float(normalized_mutual_info_score(
                             single_venue[single], d["labels"][single])), 4)})
            logger.info(rows[-1])
    with open(Path(out_dir) / "case_study_seeds.json", "w") as f:
        json.dump(rows, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "analyse", "plot", "seeds"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--figure", default="paper/figures/case_study.pdf")
    parser.add_argument("--method", choices=["ours", "nocd_eagg"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="results/case_study")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.stage == "fit":
        fit(args.method, args.seed)
    elif args.stage == "analyse":
        analyse(args.out)
    elif args.stage == "seeds":
        seeds(args.out, args.seeds)
    else:
        plot(args.out, args.figure)


if __name__ == "__main__":
    main()
