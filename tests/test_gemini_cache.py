"""Repeated LLM requests are served from the cache (latency + determinism)."""

from types import SimpleNamespace

import src.gemini_client as gc


def test_generate_json_caches_responses(monkeypatch, tmp_path):
    calls = []

    def generate_content(model, contents, config):
        calls.append(contents)
        return SimpleNamespace(parsed=[3, 1, 2], text="[3, 1, 2]")

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    monkeypatch.setattr(gc, "_LLM_CACHE_PATH", tmp_path / "llm.pkl")
    monkeypatch.setattr(gc, "_llm_cache", None)

    assert gc.generate_json("rank these", schema=list[int]) == [3, 1, 2]
    assert gc.generate_json("rank these", schema=list[int]) == [3, 1, 2]
    assert len(calls) == 1

    monkeypatch.setattr(gc, "_llm_cache", None)  # fresh process: loads from disk
    assert gc.generate_json("rank these", schema=list[int]) == [3, 1, 2]
    assert len(calls) == 1


def test_index_path_is_per_model(monkeypatch):
    import src.gemini_client as gc

    monkeypatch.setattr(gc, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(gc, "EMBEDDING_MODEL", "gemini-embedding-001")
    assert gc.index_path() == "outputs/faiss_gemini_001.index"
    monkeypatch.setattr(gc, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(gc, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    assert gc.index_path() == "outputs/faiss_bge_base_en_v1_5.index"


def test_local_provider_adds_bge_query_prefix_and_checks_dim(monkeypatch, tmp_path):
    import numpy as np

    import src.gemini_client as gc

    seen = []

    class FakeModel:
        def encode(self, texts):
            seen.extend(texts)
            return np.ones((len(texts), gc.EMBEDDING_DIM))

    monkeypatch.setattr(gc, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(gc, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setattr(gc, "_local_model", lambda: FakeModel())
    monkeypatch.setattr(gc, "_CACHE_PATH", tmp_path / "e.pkl")
    monkeypatch.setattr(gc, "_cache", None)

    out = gc.embed("java developer")
    assert seen == [gc.BGE_QUERY_PREFIX + "java developer"]
    assert out.shape == (1, gc.EMBEDDING_DIM)
    assert abs(np.linalg.norm(out[0]) - 1) < 1e-4

    gc.embed(["doc text"], task_type="RETRIEVAL_DOCUMENT")
    assert seen[-1] == "doc text"  # documents get no prefix


def test_concurrent_cache_writes_do_not_fail(monkeypatch, tmp_path):
    """The API calls embed/generate_json from worker threads; a cache write must never fail the call."""
    import threading

    import numpy as np

    fake = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda model, contents, config: SimpleNamespace(parsed=[1], text="[1]")
    ))
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    monkeypatch.setattr(gc, "_embed_remote", lambda texts, task_type: np.ones((len(texts), gc.EMBEDDING_DIM), dtype="float32"))
    monkeypatch.setattr(gc, "_LLM_CACHE_PATH", tmp_path / "llm.pkl")
    monkeypatch.setattr(gc, "_CACHE_PATH", tmp_path / "emb.pkl")
    monkeypatch.setattr(gc, "_llm_cache", None)
    monkeypatch.setattr(gc, "_cache", None)

    errors = []

    def worker(i):
        for j in range(50):
            try:
                gc.embed(f"query {i} {j}")
                gc.generate_json(f"prompt {i} {j}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    monkeypatch.setattr(gc, "_llm_cache", None)  # every response reached the file on disk
    assert len(gc._load_llm_cache()) == 400
