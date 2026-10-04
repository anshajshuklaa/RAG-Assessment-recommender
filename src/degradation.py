"""
Per-request record of fallbacks taken (embedding, query enhancer, reranker).

Components call report_degraded(); WorkflowOrchestrator.run() starts a fresh
record per request and returns it, so callers can tell a full answer from a
fallback answer instead of seeing an indistinguishable 200 OK.
"""

from contextvars import ContextVar
from typing import List, Optional

_reasons: ContextVar[Optional[List[str]]] = ContextVar("degraded_reasons", default=None)


def start_request() -> List[str]:
    reasons: List[str] = []
    _reasons.set(reasons)
    return reasons


def report_degraded(reason: str) -> None:
    # asyncio.to_thread copies the context, so threads append to the same list
    reasons = _reasons.get()
    if reasons is not None and reason not in reasons:
        reasons.append(reason)
