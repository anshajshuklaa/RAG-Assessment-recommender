"""
Build the FAISS index over the assessment catalogue with the configured Gemini
embedding model, and record the model in outputs/metadata.json.

Usage:  python scripts/build_index.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gemini_client import EMBEDDING_DIM, EMBEDDING_MODEL, embed, index_path  # noqa: E402

ASSESSMENTS_PATH = "outputs/assessments_processed.csv"
INDEX_PATH = index_path()
METADATA_PATH = "outputs/metadata.json"
BATCH_SIZE = 50
BATCH_PAUSE_SECS = 32  # free tier allows 100 embedding requests per minute


def main():
    df = pd.read_csv(ASSESSMENTS_PATH)
    texts = df["search_text"].fillna(df["name"]).astype(str).str[:2000].tolist()

    batches = []
    for start in range(0, len(texts), BATCH_SIZE):
        while True:
            try:
                batches.append(embed(texts[start:start + BATCH_SIZE], task_type="RETRIEVAL_DOCUMENT"))
                break
            except Exception as e:  # rate limit: wait for the quota window to reset
                print(f"  batch {start}: {str(e)[:80]}; retrying in 65s")
                time.sleep(65)
        print(f"Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
        if start + BATCH_SIZE < len(texts):
            time.sleep(BATCH_PAUSE_SECS)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    for vectors in batches:
        index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    Path(METADATA_PATH).write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "num_assessments": index.ntotal,
        "text_field": "search_text",
        "reindexed_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    print(f"Wrote {INDEX_PATH} ({index.ntotal} vectors) and {METADATA_PATH}")


if __name__ == "__main__":
    main()
