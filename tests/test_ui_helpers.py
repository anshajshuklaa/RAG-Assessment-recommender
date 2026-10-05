"""UI helpers: response shaping and API error handling (no Streamlit, no network)."""

import requests

import src.ui_helpers as ui

ASSESSMENT = {
    "name": "Java 8 (New)", "url": "https://example.com/java", "description": "d", "duration": 0,
    "test_type": ["Knowledge & Skills"], "remote_support": "Yes", "adaptive_support": "No",
}


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


def test_unknown_duration_is_not_published():
    assert ui.format_duration(0) == "Not published"
    assert ui.format_duration(None) == "Not published"
    assert ui.format_duration(30) == "30 min"


def test_latency_format():
    assert ui.format_latency(250) == "250 ms"
    assert ui.format_latency(2500) == "2.5 s"


def test_rows_flatten_assessments():
    row = ui.to_rows([ASSESSMENT])[0]
    assert row["Rank"] == 1 and row["Duration"] == "Not published"
    assert row["Test types"] == "Knowledge & Skills" and row["Remote"] == "Yes"


def test_degraded_reasons_come_from_headers(monkeypatch):
    resp = FakeResponse(200, {"recommended_assessments": [ASSESSMENT]},
                        {"X-Degraded": "true", "X-Degraded-Reasons": "reranker_unavailable,embedding_unavailable"})
    monkeypatch.setattr(ui.requests, "post", lambda *a, **k: resp)
    result = ui.fetch_recommendations("http://api", "java developer")
    assert result.error is None and len(result.assessments) == 1
    assert result.degraded_reasons == ["reranker_unavailable", "embedding_unavailable"]
    assert "reranking" in ui.describe_reason("reranker_unavailable")


def test_full_answer_has_no_degraded_reasons(monkeypatch):
    resp = FakeResponse(200, {"recommended_assessments": []}, {"X-Degraded": "false"})
    monkeypatch.setattr(ui.requests, "post", lambda *a, **k: resp)
    assert ui.fetch_recommendations("http://api", "java developer").degraded_reasons == []


def test_api_down_gives_friendly_error(monkeypatch):
    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(ui.requests, "post", refuse)
    result = ui.fetch_recommendations("http://api", "java developer")
    assert result.assessments == [] and "Cannot reach the API" in result.error


def test_http_errors_map_to_messages(monkeypatch):
    monkeypatch.setattr(ui.requests, "post", lambda *a, **k: FakeResponse(503))
    assert "temporarily unavailable" in ui.fetch_recommendations("http://api", "java developer").error
