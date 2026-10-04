"""Regression tests for the retrieval bugs fixed in this module (no API calls)."""

import asyncio

import faiss
import numpy as np
import pandas as pd
import pytest

from src.retreiver import HybridRetriever, parse_duration_minutes
from workflow_graph import WorkflowOrchestrator


def _retriever_with_index(vectors: np.ndarray) -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    retriever.faiss_index = index
    retriever.embeddings = None
    retriever.df = pd.DataFrame({"name": [f"a{i}" for i in range(len(vectors))]})
    return retriever


def test_dense_scores_follow_catalogue_order():
    """Bug #1: FAISS returns scores sorted by similarity; they must map back to their rows."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    retriever = _retriever_with_index(vectors)

    for row in (0, 7, 19):
        scores = retriever._get_dense_scores(vectors[row])
        assert int(np.argmax(scores)) == row
        expected = vectors @ vectors[row]
        assert np.argsort(scores).tolist() == np.argsort(expected).tolist()


@pytest.mark.parametrize("raw, expected", [
    ("49 minutes", 49), ("30", 30), ("Untimed", None), (float("nan"), None), (None, None),
])
def test_parse_duration_minutes(raw, expected):
    """Bug #2: catalogue durations are text like '49 minutes'."""
    assert parse_duration_minutes(raw) == expected


class _FakeRetriever:
    def __init__(self):
        self.df = pd.DataFrame({
            "name": ["Java 8", "OPQ32r", "Verify G+"],
            "url": ["u/java", "u/opq", "u/verify"],
            "description": ["d1", "d2", "d3"],
            "duration_minutes": [18, 25, None],
            "test_types": ["['Knowledge']", "['Personality']", "['Ability']"],
            "adaptive_irt_support": ["No", "No", "Yes"],
            "remote_testing_support": ["Yes", "No", "Yes"],
        })

    def retrieve(self, query, k=10, return_scores=False):
        assert return_scores, "workflow must request scores"
        scores = [0.9, 0.6, 0.3]
        return [(i, {"hybrid": s}) for i, s in enumerate(scores)]


def _rag_results():
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    orchestrator.retriever = _FakeRetriever()
    state = asyncio.run(orchestrator._rag_node({"query": "java", "enhanced_query": None, "strategy": "focused"}))
    return orchestrator, state


def test_rag_node_carries_scores_duration_and_flags():
    """Bugs #2, #3, #4: scores, durations and adaptive/remote flags reach the results."""
    _, state = _rag_results()
    results = state["retrieval_results"]

    assert [r["final_score"] for r in results] == pytest.approx([1.0, 0.6 / 0.9, 0.3 / 0.9])
    assert [r["duration"] for r in results] == [18, 25, 0]
    assert [r["adaptive_irt"] for r in results] == [False, False, True]
    assert [r["remote_testing"] for r in results] == [True, False, True]


def test_retrieval_confidence_depends_on_scores():
    """Bug #3: confidence used to be a constant 0.45 because every score was 0."""
    orchestrator, _ = _rag_results()
    strong = [{"final_score": 1.0 if i < 3 else 0.5, "raw_score": 1.0} for i in range(50)]
    weak = [{"final_score": 1.0, "raw_score": 0.2} for _ in range(50)]

    strong_conf = orchestrator._calculate_retrieval_confidence(strong)
    weak_conf = orchestrator._calculate_retrieval_confidence(weak)
    assert strong_conf >= orchestrator.MIN_RETRIEVAL_CONFIDENCE > weak_conf


def test_specificity_scores_stay_normalised():
    """Bug #5: rule boosts used to be added after normalisation, pushing scores up to ~15."""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.df = pd.DataFrame({
        "name": ["Financial Accounting", "Verify Numerical", "Customer Service"],
        "description": ["finance test", "numerical reasoning", "call center"],
    })
    for query in ["senior financial analyst", "consultant", "COO executive"]:
        scores = retriever._get_specificity_scores(query)
        assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_degradation_reasons_are_collected_per_request():
    from src.degradation import report_degraded, start_request

    reasons = start_request()
    report_degraded("reranker_unavailable")
    report_degraded("reranker_unavailable")
    assert reasons == ["reranker_unavailable"]
    assert start_request() == []
