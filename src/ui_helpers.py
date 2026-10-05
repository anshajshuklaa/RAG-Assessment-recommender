"""
Streamlit-free helpers for the UI: calling the API and shaping its response.

Kept separate from src/ui_simple.py so they can be unit-tested without Streamlit.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

EXAMPLE_QUERIES: Dict[str, str] = {
    "Java developer": (
        "I am hiring for Java developers who can also collaborate effectively with my business teams. "
        "Looking for an assessment that can be completed in 40 minutes."
    ),
    "Data analyst": "Data analyst with strong SQL, Python and Excel skills, plus clear written communication.",
    "Sales graduate": "Entry-level sales role for new graduates; need to assess personality and verbal ability.",
    "Customer support": "Customer service representatives for a contact centre, English comprehension and empathy.",
}

PIPELINE_STAGES: List[str] = [
    "Query enhancement: extract role, skills and duration limits (Gemini)",
    "Hybrid retrieval: FAISS semantic + BM25 + specificity + quality, fused (weighted or RRF)",
    "Test-type balancing between Knowledge and Personality tests",
    "LLM reranking of the shortlist (Gemini)",
]

# Recall@10 on the 10 labelled training queries, from analysis.md (RRF fusion)
RECALL_AT_10: Dict[str, float] = {
    "Semantic search off": 0.198,
    "BAAI/bge-base-en-v1.5": 0.276,
}

DEGRADED_REASON_LABELS: Dict[str, str] = {
    "embedding_unavailable": "Semantic search was unavailable; results use keyword matching only.",
    "query_enhancer_unavailable": "Query enhancement was unavailable; the raw query was used.",
    "reranker_unavailable": "LLM reranking was unavailable; results are in retrieval order.",
}


@dataclass
class ApiResult:
    assessments: List[dict] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


def check_health(api_url: str, timeout: float = 5) -> bool:
    try:
        return requests.get(f"{api_url}/health", timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False


def fetch_recommendations(
    api_url: str,
    query: str,
    top_k: int = 10,
    test_type_ratio: Optional[Dict[str, float]] = None,
    api_key: str = "",
    timeout: float = 90,
) -> ApiResult:
    """POST /recommend and return assessments, degradation reasons, latency or a friendly error."""
    payload: dict = {"query": query, "top_k": top_k}
    if test_type_ratio:
        payload["test_type_ratio"] = test_type_ratio
    headers = {"X-API-Key": api_key} if api_key else {}

    start = time.perf_counter()
    try:
        response = requests.post(f"{api_url}/recommend", json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return ApiResult(error=f"Cannot reach the API at {api_url}. Is it running?")
    except requests.exceptions.Timeout:
        return ApiResult(error=f"The API did not answer within {timeout:.0f} seconds. Please try again.")
    except requests.exceptions.RequestException as exc:
        return ApiResult(error=f"Request failed: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        messages = {
            401: "The API rejected the request: set API_KEY to match the server.",
            422: "The query was rejected: it must be between 5 and 5000 characters.",
            429: "Too many requests. Please wait a minute and try again.",
            503: "Recommendations are temporarily unavailable. Please try again shortly.",
        }
        return ApiResult(
            latency_ms=latency_ms,
            error=messages.get(response.status_code, f"The API returned an error ({response.status_code})."),
        )

    reasons = response.headers.get("X-Degraded-Reasons", "")
    return ApiResult(
        assessments=response.json().get("recommended_assessments", []),
        degraded_reasons=[r for r in reasons.split(",") if r],
        latency_ms=latency_ms,
    )


def format_duration(minutes) -> str:
    """The API sends 0 when the catalogue publishes no duration."""
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return "Not published"
    return f"{minutes} min" if minutes > 0 else "Not published"


def format_latency(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def describe_reason(reason: str) -> str:
    return DEGRADED_REASON_LABELS.get(reason, reason.replace("_", " "))


def to_rows(assessments: List[dict]) -> List[dict]:
    """Flatten assessments into table / CSV rows."""
    return [
        {
            "Rank": i,
            "Assessment": a.get("name", ""),
            "URL": a.get("url", ""),
            "Duration": format_duration(a.get("duration")),
            "Test types": ", ".join(a.get("test_type") or []),
            "Remote": a.get("remote_support", ""),
            "Adaptive": a.get("adaptive_support", ""),
        }
        for i, a in enumerate(assessments, 1)
    ]
