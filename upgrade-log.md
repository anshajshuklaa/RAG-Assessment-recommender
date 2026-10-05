# Upgrade log: Hugging Face embedding experiments

Builds on PR 5 (`claude/project-thread-apdr0m-pr5`), which added `EMBEDDING_PROVIDER=local|hf`.

| Step | Result |
|---|---|
| Hugging Face downloads | Work without a token once `*.hf.co` / `*.xethub.hf.co` are allowed by the network policy |
| Local mode without a Gemini key | **Fixed.** `HybridRetriever` required `GEMINI_API_KEY` even for `EMBEDDING_PROVIDER=local`; now only for `gemini` |
| `build_index.py` pacing | **Fixed.** It slept 32 s between batches for every provider; now only for Gemini's quota |
| Local BGE index | `outputs/faiss_bge_base_en_v1_5.index`: 506 vectors, 768-dim, built in 40 s on CPU |
| Experiment 2 (RRF vs weighted, semantic on) | bge-base + RRF: R@10 0.198 → 0.276 vs semantic off. Weighted: 0.254 → 0.283 |
| Experiment 3 (query decomposition) | No gain on any model; R@50 usually drops. Keep it off |
| 4 open embedding models compared | bge-base best; bge-small and MiniLM hurt R@10; e5-base best R@50 (0.484) |
| Fusion-weight sweep | Best is about 1 point above the defaults and overfits 10 queries; defaults kept |
| 3 cross-encoder rerankers on top-50 | ms-marco-MiniLM +2 pts (noise); bge-reranker-base and mxbai-xsmall hurt results; not adopted |

Reproduce: `python scripts/hf_experiments.py` (about 15 min on 4 CPUs). Details and caveats: `analysis.md`.

Open follow-ups:
- Label more queries (the main blocker for trustworthy comparisons).
- Make bge-base + RRF the documented local default in the README and `.env.example`.
- Run the full pipeline (query enhancer + LLM reranker) end to end with a Gemini key.
