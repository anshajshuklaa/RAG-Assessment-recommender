"""
Offline retrieval experiments with open Hugging Face models (no API keys, CPU only).

For each embedding model:
  - builds outputs/faiss_<model>.index from the catalogue (skipped if present)
  - evaluates semantic-only, BM25-only and hybrid retrieval (weighted / RRF fusion,
    with and without query decomposition) on data/train.csv
  - sweeps the semantic / BM25 / specificity / quality fusion weights
Then reranks the best hybrid top-50 shortlist with open cross-encoders.

Usage:
  python scripts/hf_experiments.py                     # everything, writes outputs/experiments/*.json
  python scripts/hf_experiments.py --models BAAI/bge-base-en-v1.5 --skip-rerank

Note: data/train.csv has only 10 labelled queries (65 labels), so differences of a
few points are within noise. Results are a direction, not a verdict.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "experiments"

EMBEDDING_MODELS = {
    # model id: output dimension
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-base-v2": 768,
}
CROSS_ENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
]
SHORTLIST = 50
K = 10


def slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").replace("-", "_")


# --------------------------------------------------------------------------- worker

def run_model(model: str, dim: int, sweep: bool) -> dict:
    """Runs inside a subprocess whose env selects the embedding model."""
    import faiss
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from scripts.evaluate import recall_and_ap
    from src import gemini_client
    from src.retriever import HybridRetriever

    # e5 models expect "query: " / "passage: " prefixes
    if "e5" in model:
        original = gemini_client._embed_remote

        def e5_embed(texts, task_type):
            prefix = "query: " if task_type == "RETRIEVAL_QUERY" else "passage: "
            return np.asarray(gemini_client._local_model().encode([prefix + t for t in texts]), dtype="float32")

        gemini_client._embed_remote = e5_embed
        assert original  # keep reference for clarity

    index_file = gemini_client.index_path()
    build_secs = 0.0
    if not Path(index_file).exists():
        df = pd.read_csv("outputs/assessments_processed.csv")
        texts = df["search_text"].fillna(df["name"]).astype(str).str[:2000].tolist()
        start = time.time()
        vectors = gemini_client.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        build_secs = time.time() - start
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, index_file)

    retriever = HybridRetriever(faiss_index_path=index_file)
    urls = retriever.df["url"].tolist()
    labels = pd.read_csv("data/train.csv")
    gold = labels.groupby("Query")["Assessment_url"].apply(list).to_dict()
    queries = list(gold)

    # Pre-compute component scores once per query; every variant below reuses them
    embs = retriever._get_query_embeddings(queries)
    comps = [retriever._component_scores(q, e) for q, e in zip(queries, embs)]

    def score(rankings):
        r10, ap, r50 = [], [], []
        for ranked, q in zip(rankings, queries):
            ranked_urls = [urls[j] for j in ranked[:SHORTLIST]]
            a, b = recall_and_ap(ranked_urls, gold[q], K)
            r10.append(a)
            ap.append(b)
            r50.append(recall_and_ap(ranked_urls, gold[q], SHORTLIST)[0])
        return {"recall@10": float(np.mean(r10)), "map@10": float(np.mean(ap)), "recall@50": float(np.mean(r50))}

    top = lambda s: np.argsort(-s, kind="stable")  # noqa: E731
    results = {
        "model": model,
        "index_build_secs": round(build_secs, 1),
        "semantic_only": score([top(c["semantic"]) for c in comps]),
        "bm25_only": score([top(c["bm25"]) for c in comps]),
        "hybrid_weighted": score([top(retriever._weighted_fusion(c)) for c in comps]),
        "hybrid_rrf": score([top(retriever._rrf_fusion(c)) for c in comps]),
        # What PR 5 measured without an embedding key: semantic component zeroed
        "no_semantic_weighted": score([top(retriever._weighted_fusion({**c, "semantic": c["semantic"] * 0})) for c in comps]),
        "no_semantic_rrf": score([top(retriever._rrf_fusion({**c, "semantic": c["semantic"] * 0})) for c in comps]),
    }
    for fusion in ("weighted", "rrf"):
        rankings = [retriever.retrieve(q, k=SHORTLIST, fusion=fusion, decompose=True) for q in queries]
        results[f"hybrid_{fusion}_decompose"] = score(rankings)

    if sweep:
        grid = []
        default = dict(retriever.COMPONENT_WEIGHTS)
        steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for sem, bm, spec in itertools.product(steps, repeat=3):
            qual = round(1.0 - sem - bm - spec, 2)
            if qual < 0 or qual > 0.2:
                continue
            retriever.COMPONENT_WEIGHTS = {"semantic": sem, "bm25": bm, "specificity": spec, "quality": qual}
            for fusion, fuse in (("weighted", retriever._weighted_fusion), ("rrf", retriever._rrf_fusion)):
                if fusion == "rrf" and sem + bm + spec + qual == 0:
                    continue
                m = score([top(fuse(c)) for c in comps])
                grid.append({"weights": dict(retriever.COMPONENT_WEIGHTS), "fusion": fusion, **m})
        retriever.COMPONENT_WEIGHTS = default
        grid.sort(key=lambda g: (g["recall@10"], g["map@10"]), reverse=True)
        results["sweep_top5"] = grid[:5]
        results["sweep_size"] = len(grid)
        results["default_weights"] = default

    # Save the best default-weight shortlist for the reranking stage
    best = max(("hybrid_weighted", "hybrid_rrf"), key=lambda k: results[k]["recall@50"])
    fuse = retriever._weighted_fusion if best == "hybrid_weighted" else retriever._rrf_fusion
    results["shortlists"] = {q: [int(j) for j in top(fuse(c))[:SHORTLIST]] for q, c in zip(queries, comps)}
    results["shortlist_fusion"] = best
    return results


# --------------------------------------------------------------------------- reranking

def run_rerank(shortlists: dict, source_model: str) -> list:
    import numpy as np
    import pandas as pd
    from sentence_transformers import CrossEncoder

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from scripts.evaluate import recall_and_ap

    df = pd.read_csv("outputs/assessments_processed.csv")
    urls = df["url"].tolist()
    docs = (df["name"].astype(str) + ". " + df["description"].fillna("").astype(str).str[:500]).tolist()
    gold = pd.read_csv("data/train.csv").groupby("Query")["Assessment_url"].apply(list).to_dict()

    def metrics(order_by_query):
        r10 = [recall_and_ap([urls[j] for j in order], gold[q], K)[0] for q, order in order_by_query.items()]
        ap = [recall_and_ap([urls[j] for j in order], gold[q], K)[1] for q, order in order_by_query.items()]
        return float(np.mean(r10)), float(np.mean(ap))

    base_r, base_ap = metrics(shortlists)
    out = [{"reranker": f"none (hybrid, {source_model})", "recall@10": base_r, "map@10": base_ap, "secs_per_query": 0.0}]
    for name in CROSS_ENCODERS:
        model = CrossEncoder(name, max_length=512)
        start = time.time()
        reranked = {}
        for q, cands in shortlists.items():
            scores = model.predict([(q, docs[j]) for j in cands], batch_size=16)
            reranked[q] = [cands[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
        secs = (time.time() - start) / len(shortlists)
        r, ap = metrics(reranked)
        out.append({"reranker": name, "recall@10": r, "map@10": ap, "secs_per_query": round(secs, 2)})
        print(f"  {name}: R@10={r:.3f} MAP@10={ap:.3f} ({secs:.1f}s/query)", flush=True)
    return out


# --------------------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(EMBEDDING_MODELS))
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:  # subprocess entry point
        res = run_model(args.worker, EMBEDDING_MODELS[args.worker], sweep=not args.no_sweep)
        print("RESULT_JSON " + json.dumps(res))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model in args.models:
        print(f"== {model}", flush=True)
        env = dict(os.environ, EMBEDDING_PROVIDER="local", EMBEDDING_MODEL=model,
                   EMBEDDING_DIM=str(EMBEDDING_MODELS[model]),
                   EMBEDDING_CACHE_PATH=str(ROOT / ".cache" / f"query_embeddings_{slug(model)}.pkl"))
        cmd = [sys.executable, __file__, "--worker", model] + (["--no-sweep"] if args.no_sweep else [])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
        if line is None:
            print(proc.stderr[-3000:])
            raise SystemExit(f"worker failed for {model}")
        res = json.loads(line[len("RESULT_JSON "):])
        all_results.append(res)
        for key in ("no_semantic_weighted", "no_semantic_rrf", "semantic_only", "bm25_only", "hybrid_weighted", "hybrid_rrf",
                    "hybrid_weighted_decompose", "hybrid_rrf_decompose"):
            m = res[key]
            print(f"  {key:26s} R@10={m['recall@10']:.3f} MAP@10={m['map@10']:.3f} R@50={m['recall@50']:.3f}")

    shortlists_by_model = {r["model"]: r.pop("shortlists") for r in all_results}
    (OUT_DIR / "embedding_results.json").write_text(json.dumps(all_results, indent=2) + "\n")

    if not args.skip_rerank:
        best = max(all_results, key=lambda r: r[r["shortlist_fusion"]]["recall@50"])
        print(f"== Cross-encoder reranking on {best['model']} top-{SHORTLIST} ({best['shortlist_fusion']})", flush=True)
        rerank = run_rerank(shortlists_by_model[best["model"]], best["model"])
        (OUT_DIR / "rerank_results.json").write_text(json.dumps(rerank, indent=2) + "\n")


if __name__ == "__main__":
    main()
