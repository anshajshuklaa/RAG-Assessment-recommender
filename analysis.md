# Open-model retrieval experiments

All numbers come from `python scripts/hf_experiments.py` (CPU, no API keys), evaluated on
the 10 labelled queries / 65 labels in `data/train.csv`. URLs are matched on their last path
segment. Raw results: `outputs/experiments/embedding_results.json` and `rerank_results.json`.

> **Sample size warning.** With 10 queries, one extra hit on one query moves Recall@10 by
> roughly 1–2 points. Treat differences under ~3 points as noise. These results show a
> direction; they don't prove anything. `data/test.csv` has no labels, so nothing here is
> held out.

## 1. Does semantic search help? (default weights 0.3 / 0.2 / 0.4 / 0.1)

Baseline = the same hybrid with the semantic component switched off, which is what the
pipeline did whenever Gemini embeddings were unavailable.

| Embedding model | Dim | Index build | Semantic only R@10 | Hybrid weighted R@10 / R@50 | Hybrid RRF R@10 / MAP@10 |
|---|---|---|---|---|---|
| *(semantic off)* | – | – | – | 0.254 / 0.430 | 0.198 / 0.146 |
| BAAI/bge-small-en-v1.5 | 384 | 22 s | 0.243 | 0.210 / 0.448 | 0.234 / 0.166 |
| **BAAI/bge-base-en-v1.5** | 768 | 33 s | 0.223 | **0.283** / 0.438 | **0.276 / 0.168** |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 14 s | 0.136 | 0.224 / 0.437 | 0.224 / 0.134 |
| intfloat/e5-base-v2 | 768 | 32 s | 0.206 | 0.237 / **0.484** | 0.237 / 0.130 |

BM25 alone: R@10 0.154, R@50 0.296.

- **bge-base is the only model that clearly helps.** Weighted R@10 goes from 0.254 to 0.283. With RRF fusion it goes from 0.198 to 0.276, and that's the largest gain in the table.
- Small models (bge-small, MiniLM) **hurt** weighted R@10. A weak semantic signal adds noise to a ranking that keyword matching already drives.
- e5-base gives the best Recall@50 (0.484). That's the ceiling for any reranker, but it doesn't turn into better top-10 ranking.

## 2. Query decomposition (Experiment 3)

Splitting long job descriptions into sub-queries and fusing their rankings **does not help**.
It's flat or worse on every model, and Recall@50 usually drops. For example, with bge-base weighted, R@50 falls from 0.438 to 0.382.
Keep `RETRIEVAL_DECOMPOSE=false`.

## 3. Fusion weight sweep

The sweep tried 180+ weight combinations per model, with each weight on a 0.0–0.6 grid and quality ≤ 0.2, under both fusion methods.
Best results:

| Model | Best weights (sem / bm25 / spec / qual) | Fusion | R@10 | MAP@10 |
|---|---|---|---|---|
| bge-base | 0.5 / 0.4 / 0.1 / 0.0 | weighted | 0.293 | 0.170 |
| bge-base | 0.3 / 0.2 / 0.5 / 0.0 | rrf | 0.292 | 0.184 |
| e5-base | 0.6 / 0.0 / 0.2 / 0.2 | weighted | 0.289 | 0.123 |
| bge-small | 0.2 / 0.4 / 0.4 / 0.0 | weighted | 0.277 | 0.142 |

The best weights are only about 1 point above the default weights with bge-base (0.293 vs 0.283).
They were picked on the same 10 queries they're scored on, so they overfit. Very different
weight vectors reach almost the same score. **Don't change the default weights on this
evidence.** That needs more labelled queries and cross-validation.

## 4. Cross-encoder reranking of the top-50

These run on the e5-base hybrid shortlist, which had the highest Recall@50.

| Reranker | R@10 | MAP@10 | CPU time / query |
|---|---|---|---|
| none (hybrid order) | 0.237 | 0.104 | – |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | **0.257** | **0.129** | 1.0 s |
| BAAI/bge-reranker-base | 0.146 | 0.098 | 6.0 s |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | 0.176 | 0.068 | 6.5 s |

- Only the small MS MARCO cross-encoder helps, by about 2 points, which is within noise.
- The two larger rerankers make results **much worse**. These queries are long job descriptions, cut off at 512 tokens, and the passages are product blurbs. That's far from what the rerankers were trained on.
- Reranking adds 1–6.5 s of CPU time per query (timings vary by machine; see `rerank_results.json`).

## Recommendation

1. For local or keyless mode, use **`EMBEDDING_PROVIDER=local` with `BAAI/bge-base-en-v1.5` and `RETRIEVAL_FUSION=rrf`**.
   R@10 is 0.276 (vs 0.198 for RRF without semantic search), and it has the best MAP@10 at default weights. There's no quota, and the index builds in about 35 s.
2. Keep the default weights and keep decomposition off.
3. Don't add a cross-encoder stage yet. The only gain is within noise, and it adds CPU latency and a model download.
4. The real bottleneck is the evaluation set. Labelling about 30–50 more queries would make every comparison above trustworthy, and would let the weights be tuned by cross-validation.
