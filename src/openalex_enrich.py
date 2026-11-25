"""
OpenAlex Enrichment

Looks up every paper of a com-DBLP build (papers.jsonl) in OpenAlex by DOI
and stores its abstract, topics, keywords and author affiliations in a cache
keyed by DOI (default: openalex_cache.jsonl next to the build folder), so
several builds share one cache and only fetch DOIs it does not have yet. Single-work lookups by DOI are free of OpenAlex's
daily usage budget; set OPENALEX_API_KEY to identify requests with your key.

The venue and other publication-location fields are deliberately not
fetched: the venue is the ground-truth label of the benchmark.

The cache is append-only, so an interrupted run resumes where it stopped.
After a --retry-errors run a DOI can appear more than once; its last record
is the current one (load_cache handles this).

Usage:
    python src/openalex_enrich.py --dataset data/com_dblp_ef_conf_le2006_venue
"""

import argparse
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works/doi:"
SELECT_FIELDS = ("id,doi,title,publication_year,language,abstract_inverted_index,"
                 "primary_topic,topics,keywords,authorships,cited_by_count")


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Rebuild abstract text from OpenAlex's word -> positions index."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def _topic(topic: Optional[Dict]) -> Optional[Dict]:
    if not topic:
        return None
    return {
        "id": topic["id"],
        "name": topic["display_name"],
        "score": topic.get("score"),
        "subfield": topic["subfield"]["display_name"],
        "field": topic["field"]["display_name"],
        "domain": topic["domain"]["display_name"],
    }


def simplify_work(work: Dict) -> Dict:
    return {
        "openalex_id": work["id"],
        "openalex_title": work.get("title"),
        "openalex_year": work.get("publication_year"),
        "language": work.get("language"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "primary_topic": _topic(work.get("primary_topic")),
        "topics": [_topic(t) for t in work.get("topics") or []],
        "keywords": [k["display_name"] for k in work.get("keywords") or []],
        "cited_by_count": work.get("cited_by_count"),
        "authorships": [{
            "position": a.get("author_position"),
            "author_id": (a.get("author") or {}).get("id"),
            "author_name": (a.get("author") or {}).get("display_name"),
            "raw_author_name": a.get("raw_author_name"),
            "institutions": [{
                "id": i.get("id"),
                "name": i.get("display_name"),
                "country": i.get("country_code"),
                "type": i.get("type"),
            } for i in a.get("institutions") or []],
        } for a in work.get("authorships") or []],
    }


class OpenAlexClient:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 6,
                 timeout: float = 30.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

    def fetch_by_doi(self, doi: str) -> Dict:
        """Return {"status": "found"|"not_found"|"error", ...} for one DOI."""
        params = {"select": SELECT_FIELDS}
        if self.api_key:
            params["api_key"] = self.api_key
        url = (API_URL + urllib.parse.quote(doi, safe="/")
               + "?" + urllib.parse.urlencode(params))

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    return {"status": "found",
                            **simplify_work(json.load(resp))}
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {"status": "not_found"}
                if e.code not in (429, 500, 502, 503, 504):
                    return {"status": "error", "error": f"HTTP {e.code}"}
                retry_after = e.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2 ** attempt
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                delay = 2 ** attempt
            time.sleep(delay + random.random())
        return {"status": "error", "error": "retries exhausted"}


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    """DOI -> latest OpenAlex record in the cache."""
    records = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["doi"]] = rec
    return records


def enrich(dataset: str, cache: Optional[str] = None, workers: int = 8,
           limit: Optional[int] = None, retry_errors: bool = False,
           api_key: Optional[str] = None):
    dataset = Path(dataset)
    out_path = Path(cache) if cache else dataset.parent / "openalex_cache.jsonl"
    done = {doi for doi, rec in load_cache(out_path).items()
            if not (retry_errors and rec["status"] == "error")}

    todo = set()
    no_doi = 0
    with open(dataset / "papers.jsonl", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            if not paper.get("doi"):
                no_doi += 1
            elif paper["doi"] not in done:
                todo.add(paper["doi"])
    todo = sorted(todo)
    if limit is not None:
        todo = todo[:limit]

    logger.info(f"{len(done):,} DOIs cached in {out_path}, {len(todo):,} to fetch, "
                f"{no_doi:,} papers without a DOI are skipped")
    if not todo:
        return

    client = OpenAlexClient(api_key=api_key)
    counts = {"found": 0, "not_found": 0, "error": 0}
    t0 = time.time()

    def task(doi):
        return {"doi": doi, **client.fetch_by_doi(doi)}

    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(task, todo), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[rec["status"]] += 1
            if i % 5000 == 0 or i == len(todo):
                out.flush()
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 3600
                logger.info(f"{i:,}/{len(todo):,} ({rate:.1f}/s, ETA {eta:.1f} h) "
                            f"{counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OpenAlex abstracts and topics for a com-DBLP build.")
    parser.add_argument("--dataset", required=True,
                        help="folder produced by src/com_dblp.py")
    parser.add_argument("--cache",
                        help="DOI-keyed cache file (default: "
                             "<dataset>/../openalex_cache.jsonl)")
    parser.add_argument("--workers", type=int, default=10,
                        help="parallel requests (OpenAlex allows <= 100/s)")
    parser.add_argument("--limit", type=int,
                        help="fetch at most this many papers (testing)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="refetch papers whose earlier lookup failed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    enrich(args.dataset, cache=args.cache, workers=args.workers, limit=args.limit,
           retry_errors=args.retry_errors,
           api_key=os.environ.get("OPENALEX_API_KEY"))


if __name__ == "__main__":
    main()
