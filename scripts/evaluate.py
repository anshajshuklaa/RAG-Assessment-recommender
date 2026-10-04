"""
Evaluate the recommender on the labelled queries in data/train.csv.

Reports Mean Recall@K and MAP@K for each stage:
  bm25    - BM25 keyword scores only
  hybrid  - hybrid retriever on the raw query (one embedding call per query)
  full    - full LangGraph workflow (query enhancer + retrieval + LLM reranker)

URLs are compared on their last path segment, because the labels mix
/solutions/products/... and /products/... prefixes.

Usage:
  python scripts/evaluate.py                 # all stages
  python scripts/evaluate.py --stages bm25 hybrid
  python scripts/evaluate.py --pause 20      # seconds between queries (free-tier quotas)
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def url_key(url: str) -> str:
    return str(url).strip().rstrip("/").split("/")[-1].lower()


def recall_and_ap(predicted, relevant, k):
    predicted = [url_key(u) for u in predicted][:k]
    relevant = {url_key(u) for u in relevant}
    hits, precision_sum = 0, 0.0
    for rank, url in enumerate(predicted, 1):
        if url in relevant:
            hits += 1
            precision_sum += hits / rank
    return hits / len(relevant), precision_sum / min(len(relevant), k)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--stages", nargs="+", default=["bm25", "hybrid", "full"], choices=["bm25", "hybrid", "full"])
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between queries")
    parser.add_argument("--out", help="optional CSV path for per-query results")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    from workflow_graph import get_orchestrator

    labels = pd.read_csv(args.data)
    gold = labels.groupby("Query")["Assessment_url"].apply(list).to_dict()
    orchestrator = get_orchestrator()
    retriever = orchestrator.retriever
    urls = retriever.df["url"].tolist()

    rows = []
    for i, (query, relevant) in enumerate(gold.items(), 1):
        row = {"query": query[:60].replace("\n", " "), "n_relevant": len(relevant)}
        if "bm25" in args.stages:
            top = np.argsort(retriever._get_sparse_scores(query))[::-1][:args.k]
            row["bm25_recall"], row["bm25_ap"] = recall_and_ap([urls[j] for j in top], relevant, args.k)
        if "hybrid" in args.stages:
            top = retriever.retrieve(query, k=args.k)
            row["hybrid_recall"], row["hybrid_ap"] = recall_and_ap([urls[j] for j in top], relevant, args.k)
        if "full" in args.stages:
            start = time.time()
            output = asyncio.run(orchestrator.run(query))
            row["full_secs"] = time.time() - start
            predicted = [r.get("url", "") for r in output["results"]]
            row["full_recall"], row["full_ap"] = recall_and_ap(predicted, relevant, args.k)
        rows.append(row)
        print(f"[{i}/{len(gold)}] {row['query']}")
        if args.pause and i < len(gold):
            time.sleep(args.pause)

    results = pd.DataFrame(rows)
    if args.out:
        results.to_csv(args.out, index=False)

    print(f"\nMean over {len(results)} queries (K={args.k})")
    print("| Stage | Recall@K | MAP@K |")
    print("|---|---|---|")
    for stage in args.stages:
        print(f"| {stage} | {results[f'{stage}_recall'].mean():.3f} | {results[f'{stage}_ap'].mean():.3f} |")
    if "full" in args.stages:
        print(f"\nMean full-workflow latency: {results['full_secs'].mean():.1f}s")


if __name__ == "__main__":
    main()
