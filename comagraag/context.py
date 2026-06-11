"""Shared context helpers for CoMaGRAG."""

from typing import Any


def init_context(question: str, graph: Any, passages: list | None = None) -> dict:
    """Create the shared context object used across the pipeline."""
    return {
        "question": question,
        "graph": graph,
        "passages": passages or [],
        "subqueries": [],
        "triples": [],
        "answer": "",
        "reasoning": "",
        "score": 0.0,
        "feedback": "",
        "converged": False,
        "history": [],
        "iteration": 0,
    }

