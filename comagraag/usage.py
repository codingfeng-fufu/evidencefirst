"""Lightweight per-question usage tracking for experiment runs."""
from __future__ import annotations

import time
from typing import Any


_current: dict[str, Any] | None = None


def start_case(qid: str | None = None) -> None:
    """Start a new usage record for one evaluated question."""
    global _current
    _current = {
        "qid": qid,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "started_at": time.time(),
    }


def record_response(response: Any) -> None:
    """Record an OpenAI-compatible chat response if tracking is active."""
    if _current is None:
        return

    _current["llm_calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    input_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens

    _current["input_tokens"] += int(input_tokens or 0)
    _current["output_tokens"] += int(output_tokens or 0)
    _current["total_tokens"] += int(total_tokens or 0)


def finish_case(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finish the active record and return a serializable summary."""
    global _current
    if _current is None:
        record = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "wall_time": 0.0,
        }
    else:
        record = dict(_current)
        started_at = float(record.pop("started_at", time.time()))
        record["wall_time"] = round(time.time() - started_at, 4)

    if extra:
        record.update(extra)
    _current = None
    return record
