"""API behaviour with the workflow mocked out (no Gemini calls)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api as api

RESULT = {
    "name": "Java 8 (New)", "url": "u/java", "description": "d", "duration": 18,
    "test_types": "['Knowledge']", "adaptive_irt": True, "remote_testing": True,
}


class FakeOrchestrator:
    def __init__(self, output):
        self.output = output
        self.retriever = type("R", (), {"df": pd.DataFrame({"name": ["x"]})})()

    async def run(self, query, custom_test_type_ratio=None):
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@pytest.fixture
def client(monkeypatch):
    def make(output, api_key=None, rate_limit=30):
        monkeypatch.setattr(api, "get_orchestrator", lambda: FakeOrchestrator(output))
        monkeypatch.setattr(api, "API_KEY", api_key)
        monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", rate_limit)
        api._request_log.clear()
        return TestClient(api.app, raise_server_exceptions=False)
    return make


def post(c, **headers):
    return c.post("/recommend", json={"query": "java developer"}, headers=headers)


def test_full_answer_is_not_marked_degraded(client):
    with client({"results": [RESULT], "degraded_reasons": []}) as c:
        r = post(c)
    assert r.status_code == 200
    assert r.headers["X-Degraded"] == "false"
    body = r.json()["recommended_assessments"][0]
    assert body["duration"] == 18 and body["adaptive_support"] == "Yes"


def test_fallback_answer_is_marked_degraded(client):
    with client({"results": [RESULT], "degraded_reasons": ["reranker_unavailable"]}) as c:
        r = post(c)
    assert r.status_code == 200
    assert r.headers["X-Degraded"] == "true"
    assert r.headers["X-Degraded-Reasons"] == "reranker_unavailable"


def test_no_results_returns_503(client):
    with client({"results": [], "error": "Retrieval failed: boom"}) as c:
        r = post(c)
    assert r.status_code == 503
    assert "boom" not in r.text


def test_internal_errors_are_not_leaked(client):
    with client(RuntimeError("secret internal detail")) as c:
        r = post(c)
    assert r.status_code == 500
    assert "secret" not in r.text


def test_api_key_required_when_configured(client):
    with client({"results": [RESULT]}, api_key="k") as c:
        assert post(c).status_code == 401
        assert post(c, **{"X-API-Key": "k"}).status_code == 200


def test_rate_limit(client):
    with client({"results": [RESULT]}, rate_limit=2) as c:
        codes = [post(c).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
