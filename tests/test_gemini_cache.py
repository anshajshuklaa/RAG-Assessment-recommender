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
