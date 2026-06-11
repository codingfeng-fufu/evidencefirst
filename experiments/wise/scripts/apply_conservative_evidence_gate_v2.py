#!/usr/bin/env python3
"""Apply a stricter non-gold gate to conservative adjudicator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


BAD_MARKERS = (
    "unknown",
    "insufficient",
    "not enough",
    "provided context",
    "context passage",
    "does not mention",
    "do not mention",
    "cannot determine",
    "none of",
    "no triple",
    "not specified",
    "instructions require",
    "using *only*",
    "given triples",
)


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pred_tokens = tokens(pred)
    gold_tokens = tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower()
    return (
        not normalize(value)
        or len(tokens(value)) > 14
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def qid(row: dict[str, Any], fallback: int = 0) -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rows_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    return {qid(row, i): row for i, row in enumerate(load_jsonl(path))}


def reader_only_source(source: str) -> bool:
    lowered = str(source or "").lower()
    return "reader_v1" in lowered and not any(
        trusted in lowered
        for trusted in (
            "support_fallback",
            "guarded",
            "judge",
            "kg",
            "contextual_prior",
            "prior",
        )
    )


def requested_location(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in (
            "where",
            "what city",
            "what country",
            "what region",
            "headquartered",
            "located",
        )
    )


def location_type_mismatch(question: str, proposed: str, current: str) -> bool:
    if not requested_location(question):
        return False
    if bad_surface(current):
        return False
    proposed_lower = proposed.lower()
    if any(term in proposed_lower for term in ("city", "county", "state", "province", "country", "region")):
        return False
    if "," in proposed or any(term in proposed_lower for term in ("united", "south", "north", "east", "west")):
        return False
    current_lower = current.lower()
    current_looks_location = "," in current or any(
        term in current_lower for term in ("city", "county", "state", "province", "country", "region")
    )
    return current_looks_location and len(tokens(proposed)) <= 4


def accepted(row: dict[str, Any], proposed: str, current: str) -> tuple[bool, str]:
    if normalize(proposed) == normalize(current):
        return False, "same_as_current"
    if str(row.get("adjudicator_action") or "").lower() != "switch":
        return False, "action_not_switch"
    if str(row.get("adjudicator_confidence") or "").lower() != "high":
        return False, "not_high_confidence"
    if bad_surface(proposed):
        return False, "bad_surface"
    source = str(row.get("adjudicator_source") or "")
    if reader_only_source(source):
        return False, "reader_only_source"
    if location_type_mismatch(str(row.get("question") or ""), proposed, current):
        return False, "location_type_mismatch"
    return True, "accepted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    current_by_id = rows_by_qid(args.current)
    adjudicated = load_jsonl(args.adjudicated)

    rows = []
    for i, row in enumerate(adjudicated):
        row_qid = qid(row, i)
        base = current_by_id[row_qid]
        current = str(base.get("answer") or base.get("prediction") or "")
        proposed = str(row.get("adjudicator_answer") or "").strip()
        ok, acceptance = accepted(row, proposed, current)
        selected = proposed if ok else current
        gold = str(base.get("gold") or row.get("gold") or "")
        rows.append(
            {
                **row,
                "mode": "conservative_evidence_gate_v2",
                "answer": selected,
                "prediction": selected,
                "current_answer": current,
                "selected_rule": "adjudicator_switch" if ok else "current_v3",
                "acceptance": acceptance if row.get("llm_called") else "not_target",
                "em": em(selected, gold),
                "f1": round(f1(selected, gold), 4),
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "conservative_evidence_gate_v2",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4),
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4),
        "targets": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row["selected_rule"] == "adjudicator_switch" for row in rows),
    }
    summary.update({f"acceptance_{key}": value for key, value in Counter(row.get("acceptance", "") for row in rows).items()})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"EM={summary['EM']:.4f} F1={summary['F1']:.4f} switches={summary['switches']}")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
