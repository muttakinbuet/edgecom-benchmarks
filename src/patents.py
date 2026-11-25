"""
Patent Co-Inventor Benchmark (PatentsView)

A second text-attributed benchmark in the subset format of dblp_markbp.py,
built like the DBLP-EF subsets: nodes are inventors, two inventors are linked
when they share a patent, each edge keeps the patents behind it, and the edge
features are sentence embeddings of those patents' text.

Selection: utility patents granted in [--start, --end] whose primary CPC
subclass (cpc_sequence 0, the counterpart of a paper's venue) is one of the
chosen subclasses; patents with more than --max-inventors inventors are
dropped (as com_dblp.py drops papers with more than 50 authors); the graph is
the largest connected component.

Ground truths:
  communities.txt   primary CPC subclass: an inventor belongs to every chosen
                    subclass in which they have a patent of the selection
  <out>_assignee/   assignee (company) communities, the counterpart of the
                    institution ground truth: an inventor belongs to every
                    organisation assigned at least one of their patents granted
                    in the window (any subclass); organisations with at least
                    --assignee-min-size members; scored on covered inventors
                    only (eval_nodes.npy)

Edge features (L2-normalised mean of the edge's patents' embeddings, MiniLM as
in text_edge_features.py):
  edge_feats_title, edge_feats_title_abstract
  *_outside   leakage control: only the pair's patents in the window whose
              primary subclass is NOT one of the chosen ones; edges without
              such patents get a zero vector (as for DBLP-EF)

Usage:
    python src/patents.py --stage graph
    python src/patents.py --stage features
"""

import argparse
import collections
import itertools
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Section G subclasses with 1,000-10,000 primary patents granted 2006-2010
G_MID = ["G03G", "G06T", "G02F", "G09G", "G03F", "G06K", "G01S", "G07F", "G05B", "G03B",
         "G10L", "G06V", "G01C", "G01B", "G08B", "G01J", "G01V", "G01D", "G01F", "G01L",
         "G09F", "G01M", "G01T", "G05F", "G08G", "G09B", "G05D", "G07C", "G06N", "G10H"]


def read(raw: Path, name: str, cols, chunksize=2_000_000):
    z = zipfile.ZipFile(raw / f"{name}.tsv.zip")
    return pd.read_csv(z.open(z.namelist()[0]), sep="\t", usecols=cols, dtype=str,
                       chunksize=chunksize, keep_default_na=False)


def window_patents(raw: Path, start: str, end: str) -> set:
    keep = set()
    for c in read(raw, "g_patent", ["patent_id", "patent_type", "patent_date"]):
        y = c.patent_date.str[:4]
        keep.update(c.patent_id[(c.patent_type == "utility") & (y >= start) & (y <= end)])
    return keep


def primary_subclass(raw: Path, patents: set) -> pd.Series:
    rows = [c[(c.cpc_sequence == "0") & c.patent_id.isin(patents)]
            for c in read(raw, "g_cpc_current", ["patent_id", "cpc_sequence", "cpc_subclass"])]
    return pd.concat(rows).set_index("patent_id").cpc_subclass


def inventors(raw: Path, patents: set) -> pd.DataFrame:
    rows = [c[c.patent_id.isin(patents)] for c in
            read(raw, "g_inventor_disambiguated", ["patent_id", "inventor_id",
                                                     "disambig_inventor_name_first",
                                                     "disambig_inventor_name_last"])]
    return pd.concat(rows).drop_duplicates(["patent_id", "inventor_id"])


def build_graph(args):
    raw, out = Path(args.raw), Path(args.out)
    window = window_patents(raw, args.start, args.end)
    primary = primary_subclass(raw, window)
    logger.info(f"{len(window)} utility patents {args.start}-{args.end}")
    inv = inventors(raw, window)
    size = inv.groupby("patent_id").size()
    ok = set(size.index[size <= args.max_inventors])
    chosen = set(args.subclasses)
    selected = {p for p in ok if primary.get(p) in chosen}
    logger.info(f"{len(selected)} patents in the {len(chosen)} chosen subclasses")

    sel = inv[inv.patent_id.isin(selected)]
    ids = {a: i for i, a in enumerate(sel.inventor_id.unique())}
    pairs = collections.defaultdict(list)
    for p, group in sel.groupby("patent_id").inventor_id:
        for a, b in itertools.combinations(sorted(ids[x] for x in group), 2):
            pairs[(a, b)].append(p)
    uv = np.array(list(pairs.keys()))
    n_all = len(ids)
    adj = sp.coo_matrix((np.ones(len(uv)), (uv[:, 0], uv[:, 1])), shape=(n_all, n_all))
    _, comp = connected_components(adj, directed=False)
    largest = np.argmax(np.bincount(comp))
    old = np.flatnonzero(comp == largest)
    new = np.full(n_all, -1)
    new[old] = np.arange(len(old))
    keep_e = comp[uv[:, 0]] == largest
    uv_l = new[uv[keep_e]]
    edge_patents = [pairs[tuple(e)] for e in uv[keep_e]]
    logger.info(f"largest component: {len(old)} of {n_all} inventors, {len(uv_l)} edges")

    rev = np.array(list(ids.keys()))[old]
    names = sel.drop_duplicates("inventor_id").set_index("inventor_id")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": np.arange(len(old)), "parent_node_id": rev,
                  "name": (names.loc[rev, "disambig_inventor_name_first"] + " " +
                           names.loc[rev, "disambig_inventor_name_last"]).to_numpy()}).to_csv(
        out / "nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"u": uv_l[:, 0], "v": uv_l[:, 1], "num_papers": [len(p) for p in edge_patents],
                  "paper_ids": [";".join(p) for p in edge_patents]}).to_csv(
        out / "edges.tsv", sep="\t", index=False)

    # Ground truth 1: primary subclass of the selected patents
    node_of = dict(zip(rev, range(len(rev))))
    members = collections.defaultdict(set)
    for p, a in zip(sel.patent_id, sel.inventor_id):
        if a in node_of:
            members[primary[p]].add(node_of[a])
    comms = sorted((c for c in members if len(members[c]) >= 3), key=lambda c: -len(members[c]))
    with open(out / "communities.txt", "w") as f:
        for c in comms:
            f.write("\t".join(map(str, sorted(members[c]))) + "\n")
    pd.DataFrame({"subclass": comms, "size": [len(members[c]) for c in comms]}).to_csv(
        out / "communities.meta.tsv", sep="\t", index=False)
    memb = np.zeros(len(rev), dtype=int)
    for c in comms:
        memb[list(members[c])] += 1
    stats = {"source": "PatentsView (granted, disambiguated)", "years": [args.start, args.end],
             "subclasses": args.subclasses, "max_inventors": args.max_inventors,
             "nodes": int(len(rev)), "edges": int(len(uv_l)), "communities": len(comms),
             "memberships_per_node": float(memb.mean()),
             "overlapping_fraction": float((memb > 1).mean()),
             "patents": len({p for ps in edge_patents for p in ps})}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    logger.info(json.dumps(stats))

    # Window patents of pairs of nodes, for the outside features
    win = inv[inv.patent_id.isin(ok) & inv.inventor_id.isin(node_of)]
    outside = collections.defaultdict(list)
    edge_index = {tuple(e): i for i, e in enumerate(uv_l)}
    for p, group in win[~win.patent_id.isin(selected)].groupby("patent_id").inventor_id:
        for a, b in itertools.combinations(sorted(node_of[x] for x in group), 2):
            if (a, b) in edge_index:
                outside[edge_index[(a, b)]].append(p)
    with open(out / "edge_patents_outside.json", "w") as f:
        json.dump({str(k): v for k, v in outside.items()}, f)
    logger.info(f"{len(outside)} edges with outside patents")

    # Ground truth 2: assignees of the inventors' window patents
    assignee_truth(raw, out, inv[inv.patent_id.isin(ok)], node_of, args.assignee_min_size)


def assignee_truth(raw, out, inv_win, node_of, min_size):
    patents = set(inv_win.patent_id)
    rows = [c[c.patent_id.isin(patents) & (c.disambig_assignee_organization != "")]
            for c in read(raw, "g_assignee_disambiguated",
                          ["patent_id", "assignee_id", "disambig_assignee_organization"])]
    asg = pd.concat(rows).drop_duplicates(["patent_id", "assignee_id"])
    names = dict(zip(asg.assignee_id, asg.disambig_assignee_organization))
    link = inv_win[inv_win.inventor_id.isin(node_of)].merge(asg, on="patent_id")
    members = collections.defaultdict(set)
    for a, org in zip(link.inventor_id, link.assignee_id):
        members[org].add(node_of[a])
    kept = sorted((o for o in members if len(members[o]) >= min_size),
                  key=lambda o: -len(members[o]))
    dst = Path(str(out) + "_assignee")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("nodes.tsv", "edges.tsv"):
        if not (dst / name).exists():
            os.link(out / name, dst / name)
    with open(dst / "communities.txt", "w") as f:
        for o in kept:
            f.write("\t".join(map(str, sorted(members[o]))) + "\n")
    pd.DataFrame({"assignee_id": kept, "name": [names[o] for o in kept],
                  "size": [len(members[o]) for o in kept]}).to_csv(
        dst / "communities.meta.tsv", sep="\t", index=False)
    memb = np.zeros(len(node_of), dtype=int)
    for o in kept:
        memb[list(members[o])] += 1
    covered = memb > 0
    np.save(dst / "eval_nodes.npy", np.flatnonzero(covered))
    stats = {"ground_truth": "PatentsView assignee organisations", "min_size": min_size,
             "communities": len(kept), "evaluated_nodes": int(covered.sum()),
             "nodes": len(node_of),
             "memberships_per_evaluated_node": float(memb[covered].mean()) if covered.any() else 0,
             "overlapping_fraction_of_evaluated":
                 float((memb > 1).sum() / max(covered.sum(), 1))}
    json.dump(stats, open(dst / "stats.json", "w"), indent=2)
    logger.info(json.dumps(stats))


def build_features(args):
    from sentence_transformers import SentenceTransformer
    raw, out = Path(args.raw), Path(args.out)
    edges = pd.read_csv(out / "edges.tsv", sep="\t", dtype={"paper_ids": str})
    inside = [s.split(";") for s in edges.paper_ids]
    outside_map = json.load(open(out / "edge_patents_outside.json"))
    outside = [outside_map.get(str(i), []) for i in range(len(edges))]
    needed = {p for ps in inside + outside for p in ps}
    titles = {}
    for c in read(raw, "g_patent", ["patent_id", "patent_title"]):
        c = c[c.patent_id.isin(needed)]
        titles.update(zip(c.patent_id, c.patent_title))
    abstracts = {}
    for c in read(raw, "g_patent_abstract", ["patent_id", "patent_abstract"], chunksize=500_000):
        c = c[c.patent_id.isin(needed)]
        abstracts.update(zip(c.patent_id, c.patent_abstract))
    logger.info(f"{len(needed)} patents, {len(abstracts)} with abstract")
    order = sorted(needed)
    row = {p: i for i, p in enumerate(order)}
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = min(model.max_seq_length or args.max_seq_length,
                               args.max_seq_length)
    for text in args.texts:
        docs = [titles.get(p, "") if text == "title"
                else (titles.get(p, "") + ". " + abstracts.get(p, "")).strip()
                for p in order]
        emb = model.encode(docs, batch_size=args.batch_size, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        for suffix, lists in (("", inside), ("_outside", outside)):
            x = np.zeros((len(edges), emb.shape[1]), dtype=np.float32)
            for i, ps in enumerate(lists):
                if ps:
                    v = emb[[row[p] for p in ps]].mean(axis=0)
                    x[i] = v / max(np.linalg.norm(v), 1e-12)
            name = f"edge_feats_{text}{suffix}{args.tag}"
            np.save(out / f"{name}.npy", x)
            json.dump({"model": args.model, "max_seq_length": model.max_seq_length,
                       "text": text, "outside": bool(suffix),
                       "edges_with_text": int((np.abs(x).sum(1) > 0).sum())},
                      open(out / f"{name}.json", "w"))
            logger.info(f"{name}: {(np.abs(x).sum(1) > 0).mean():.3f} of edges with text")
    dst = Path(str(out) + "_assignee")
    if dst.exists():
        for path in out.glob("edge_feats_*.npy"):
            if not (dst / path.name).exists():
                os.link(path, dst / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["graph", "features"], required=True)
    parser.add_argument("--raw", default="data/raw/patents")
    parser.add_argument("--out", default="data/subsets/patents_g_2006_2010")
    parser.add_argument("--start", default="2006")
    parser.add_argument("--end", default="2010")
    parser.add_argument("--subclasses", nargs="+", default=G_MID)
    parser.add_argument("--max-inventors", type=int, default=50)
    parser.add_argument("--assignee-min-size", type=int, default=200)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="token cap (256 = all-MiniLM-L6-v2's own limit)")
    parser.add_argument("--texts", nargs="+", default=["title", "title_abstract"])
    parser.add_argument("--tag", default="",
                        help="appended to the output names, e.g. _bge (encoder ablation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    build_graph(args) if args.stage == "graph" else build_features(args)


if __name__ == "__main__":
    main()
