"""
scripts/ingest_arxiv.py
─────────────────────────────────────────────────────
Downloads the Cornell ArXiv dataset from HuggingFace,
preprocesses it, and bulk-indexes it into the platform
via the /api/v1/index/bulk endpoint.

Usage:
    python scripts/ingest_arxiv.py --limit 50000 --batch-size 512
"""

import argparse
import sys
import time

import httpx
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

API_BASE = "http://localhost:8000/api/v1"
BATCH_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description="Ingest ArXiv papers into the semantic search platform")
    p.add_argument("--limit", type=int, default=50_000, help="Max papers to index (default: 50k)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Papers per API request")
    p.add_argument("--api-base", type=str, default=API_BASE, help="FastAPI base URL")
    p.add_argument("--categories", nargs="*", help="Filter to specific ArXiv categories (e.g. cs.AI cs.LG)")
    return p.parse_args()


def load_arxiv(limit: int, categories: list[str] | None = None):
    logger.info("Loading ArXiv dataset from HuggingFace (streaming)...")
    ds = load_dataset(
        "Cornell-University/arxiv",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    papers = []
    for row in tqdm(ds, desc="Fetching papers", total=limit):
        if len(papers) >= limit:
            break

        # Filter by category if specified
        if categories:
            paper_cats = row.get("categories", "").split()
            if not any(c in paper_cats for c in categories):
                continue

        abstract = (row.get("abstract") or "").strip().replace("\n", " ")
        title = (row.get("title") or "").strip().replace("\n", " ")

        if not abstract or not title or len(abstract) < 50:
            continue

        papers.append({
            "paper_id": row["id"],
            "title": title,
            "abstract": abstract,
            "authors": [a.strip() for a in (row.get("authors") or "").split(",")][:10],
            "categories": (row.get("categories") or "").split(),
            "published": row.get("update_date") or row.get("versions", [{}])[0].get("created", ""),
        })

    logger.info(f"Loaded {len(papers)} papers")
    return papers


def batch_index(papers: list[dict], batch_size: int, api_base: str) -> tuple[int, int]:
    indexed, failed = 0, 0

    with httpx.Client(timeout=120.0) as client:
        for i in tqdm(range(0, len(papers), batch_size), desc="Indexing batches"):
            batch = papers[i : i + batch_size]
            try:
                resp = client.post(
                    f"{api_base}/index/bulk",
                    json={"papers": batch},
                )
                resp.raise_for_status()
                result = resp.json()
                indexed += result["indexed"]
                failed += result["failed"]
            except Exception as e:
                logger.error(f"Batch {i//batch_size} failed: {e}")
                failed += len(batch)

            time.sleep(0.1)  # be gentle

    return indexed, failed


def main():
    args = parse_args()

    logger.info(f"Starting ingestion — limit={args.limit}, batch_size={args.batch_size}")
    t0 = time.monotonic()

    papers = load_arxiv(limit=args.limit, categories=args.categories)
    indexed, failed = batch_index(papers, batch_size=args.batch_size, api_base=args.api_base)

    elapsed = round((time.monotonic() - t0) / 60, 1)
    logger.info(f"Done — indexed={indexed}, failed={failed}, time={elapsed}min")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
