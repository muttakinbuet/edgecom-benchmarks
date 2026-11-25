"""
Text Edge Features

Embeds the papers behind every edge of a benchmark subset with a sentence
encoder and gives each edge the L2-normalised mean of its papers' embeddings.
Text is the DBLP title, optionally followed by the OpenAlex abstract (joined by
DOI). Venue fields are never used, since venues are the labels; with
--exclude-label-venues, papers published in the ground-truth venues are dropped
as well (leakage control).

Usage:
    python src/text_edge_features.py --subset data/subsets/cs_areas_le2006_main --text title
    python src/text_edge_features.py --subset ... --text title_abstract \
        --cache data/openalex_cache.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .openalex_enrich import load_cache
except ImportError:  # Fallback for direct script execution
    from openalex_enrich import load_cache

logger = logging.getLogger(__name__)


def load_papers(parent: Path, paper_ids: np.ndarray):
    """Rows of papers.jsonl for the given paper ids (= line numbers)."""
    wanted = set(paper_ids.tolist())
    papers = {}
    with open(parent / "papers.jsonl", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no in wanted:
                papers[line_no] = json.loads(line)
    return papers


def main():
    parser = argparse.ArgumentParser(description="Text edge features for a subset.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--text", choices=["title", "title_abstract"], default="title")
    parser.add_argument("--cache", help="OpenAlex DOI cache (for abstracts)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="token cap (256 = all-MiniLM-L6-v2's own limit)")
    parser.add_argument("--tag", default="",
                        help="appended to the output name, e.g. _bge (encoder ablation)")
    parser.add_argument("--exclude-label-venues", action="store_true",
                        help="leakage control: ignore papers published in the "
                             "subset's ground-truth venues; edges left without "
                             "papers get a zero vector")
    parser.add_argument("--with-outside", action="store_true",
                        help="write both the full and the outside variant from "
                             "one encoding of all papers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    subset = Path(args.subset)
    with open(subset / "stats.json") as f:
        parent = Path(json.load(f)["parent_dataset"])

    edges = pd.read_csv(subset / "edges.tsv", sep="\t", dtype={"paper_ids": str})
    edge_papers = [np.array(p.split(";"), dtype=np.int64) for p in edges["paper_ids"]]
    paper_ids = np.unique(np.concatenate(edge_papers))
    papers = load_papers(parent, paper_ids)
    logger.info(f"{len(edges):,} edges backed by {len(paper_ids):,} papers")

    variants = []  # (suffix, edge paper lists)
    if args.exclude_label_venues or args.with_outside:
        label_venues = set(pd.read_csv(subset / "communities.meta.tsv",
                                       sep="\t")["venue_key"])
        outside = {pid for pid in paper_ids if papers[pid]["venue_key"] not in label_venues}
        outside_papers = [np.array([p for p in ps if p in outside], dtype=np.int64)
                          for ps in edge_papers]
        logger.info(f"Outside the label venues: {len(outside):,} papers; "
                    f"{sum(len(ps) > 0 for ps in outside_papers):,} edges keep features")
        if args.exclude_label_venues and not args.with_outside:
            paper_ids = np.array(sorted(outside), dtype=np.int64)
        else:
            variants.append((args.text, edge_papers))
        variants.append((args.text + "_outside", outside_papers))
    else:
        variants.append((args.text, edge_papers))

    abstracts = {}
    if args.text == "title_abstract":
        if not args.cache:
            raise ValueError("--text title_abstract needs --cache")
        abstracts = {doi: rec.get("abstract", "") for doi, rec in
                     load_cache(Path(args.cache)).items() if rec["status"] == "found"}

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    encoder = SentenceTransformer(args.model, device="cpu")
    encoder.max_seq_length = min(encoder.max_seq_length or args.max_seq_length,
                                 args.max_seq_length)
    # SPECTER was trained on "title [SEP] abstract"
    sep = encoder.tokenizer.sep_token if "specter" in args.model.lower() else ". "

    texts, has_abstract = [], set()
    for pid in paper_ids:
        paper = papers[pid]
        text = paper["title"]
        abstract = abstracts.get(paper.get("doi", ""), "")
        if abstract:
            text = f"{text}{sep}{abstract}"
            has_abstract.add(int(pid))
        texts.append(text)

    emb = encoder.encode(texts, batch_size=args.batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True)
    logger.info(f"Encoded {len(texts):,} papers in {time.time() - t0:.0f}s "
                f"(max_seq_length {encoder.max_seq_length})")

    index = {pid: i for i, pid in enumerate(paper_ids)}
    dim = emb.shape[1]
    for suffix, lists in variants:
        feats = np.stack([emb[[index[p] for p in ps]].mean(axis=0) if len(ps)
                          else np.zeros(dim, dtype=emb.dtype) for ps in lists])
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = np.divide(feats, norms, out=np.zeros_like(feats), where=norms > 0)

        name = f"edge_feats_{suffix}{args.tag}"
        np.save(subset / f"{name}.npy", feats.astype(np.float32))
        used = {p for ps in lists for p in ps.tolist()}
        meta = {"text": args.text, "model": args.model,
                "max_seq_length": encoder.max_seq_length,
                "papers": len(used), "papers_with_abstract": len(used & has_abstract),
                "edges": int(len(edges)),
                "edges_with_features": int(sum(len(ps) > 0 for ps in lists)),
                "dim": int(feats.shape[1])}
        with open(subset / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved {name}: {meta}")


if __name__ == "__main__":
    main()
