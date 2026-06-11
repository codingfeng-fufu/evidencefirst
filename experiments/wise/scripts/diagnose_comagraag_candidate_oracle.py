#!/usr/bin/env python3
"""Diagnose recoverable errors from existing CoMaGRAG candidate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXCLUDED_FIELDS = {
    "gold",
    "gold_answer",
    "em",
    "f1",
    "raw_response",
    "error",
    "fallback_error",
    "question",
    "mode",
    "variant",
    "type",
    "selected_source",
    "selected_choice",
    "selected_rule",
    "adjudicator_source",
    "adjudicator_confidence",
    "confidence",
    "cache_hit",
    "iterations",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "wall_time",
    "max_iter",
    "evidence_aug_enabled",
    "evidence_aug_question_type",
    "evidence_aug_candidate_count",
    "evidence_aug_passage_count",
    "evidence_aug_passages",
    "_id",
    "id",
    "qid",
}

LIST_CANDIDATE_FIELDS = {
    "deterministic_candidates",
    "reader_generated_candidates",
    "evidence_aug_candidates",
    "locator_candidates",
    "extra_candidates",
}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
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


def qid(row: dict[str, Any], fallback: int = 0) -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def is_answer_field(key: str, value: Any) -> bool:
    if key in EXCLUDED_FIELDS:
        return False
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return False
    return (
        key in {"answer", "prediction", "baseline_answer"}
        or key.endswith("_answer")
        or key.startswith("candidate_")
        or key.endswith("_candidate")
        or key in {"current_v4", "judge_v2", "contextual_prior_v2", "contextual_prior"}
    )


def split_candidate_values(key: str, value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    if key not in LIST_CANDIDATE_FIELDS and not key.endswith("_candidates"):
        return [value] if is_answer_field(key, value) else []
    values = []
    for part in re.split(r"\s+\|\s+|\s+\|\|\s+", value):
        part = re.sub(r"^[A-Za-z0-9_. -]+:\s+", "", part.strip())
        if part and part.lower() not in {"null", "none", "nan"}:
            values.append(part)
    return values


def add_candidate(
    candidates: dict[str, list[dict[str, str]]],
    qid_value: str,
    source: str,
    field: str,
    answer: str,
) -> None:
    candidates[qid_value].append({"source": source, "field": field, "answer": answer})


def classify(current_em: int, current_f1: float, best_em: int, best_f1: float) -> str:
    if current_em:
        return "current_exact"
    if best_em:
        return "selector_miss_exact_candidate"
    if best_f1 > current_f1:
        return "partial_candidate_improves_f1"
    if best_f1 > 0:
        return "partial_candidate_no_improvement"
    return "no_candidate_overlap"


def compact_sources(candidates: list[dict[str, str]], gold: str, exact_only: bool = False) -> str:
    parts = []
    seen = set()
    for candidate in candidates:
        score = em(candidate["answer"], gold)
        if exact_only and not score:
            continue
        key = (candidate["source"], candidate["field"], normalize(candidate["answer"]))
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{candidate['source']}:{candidate['field']}={candidate['answer']}")
        if len(parts) >= 5:
            break
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    current_rows = load_jsonl(args.current)
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)

    for path in args.candidate:
        source = path.stem.replace("_predictions", "")
        for i, row in enumerate(load_jsonl(path)):
            row_qid = qid(row, i)
            for field, value in row.items():
                for candidate_value in split_candidate_values(field, value):
                    add_candidate(candidates, row_qid, source, field, candidate_value)

    diagnostic_rows = []
    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        gold = str(row.get("gold") or row.get("gold_answer") or "")
        current = str(row.get("prediction") or row.get("answer") or "")
        current_em = em(current, gold)
        current_f1 = f1(current, gold)
        scored = []
        for candidate in candidates.get(row_qid, []):
            candidate_em = em(candidate["answer"], gold)
            candidate_f1 = f1(candidate["answer"], gold)
            scored.append((candidate_em, candidate_f1, candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_em = scored[0][0] if scored else 0
        best_f1 = scored[0][1] if scored else 0.0
        best_candidate = scored[0][2] if scored else {"source": "", "field": "", "answer": ""}
        exact_candidates = [item[2] for item in scored if item[0]]
        diagnostic_rows.append(
            {
                "qid": row_qid,
                "question": row.get("question", ""),
                "gold": gold,
                "current": current,
                "current_rule": row.get("selected_rule", ""),
                "current_em": current_em,
                "current_f1": f"{current_f1:.4f}",
                "best_answer": best_candidate["answer"],
                "best_source": best_candidate["source"],
                "best_field": best_candidate["field"],
                "best_em": best_em,
                "best_f1": f"{best_f1:.4f}",
                "classification": classify(current_em, current_f1, best_em, best_f1),
                "exact_candidate_sources": compact_sources(exact_candidates, gold, exact_only=True),
            }
        )

    counts = Counter(row["classification"] for row in diagnostic_rows)
    current_em_avg = sum(int(row["current_em"]) for row in diagnostic_rows) / len(diagnostic_rows)
    current_f1_avg = sum(float(row["current_f1"]) for row in diagnostic_rows) / len(diagnostic_rows)
    oracle_em_avg = sum(int(row["best_em"]) for row in diagnostic_rows) / len(diagnostic_rows)
    oracle_f1_avg = sum(float(row["best_f1"]) for row in diagnostic_rows) / len(diagnostic_rows)
    summary = {
        "n": len(diagnostic_rows),
        "current_em": f"{current_em_avg:.4f}",
        "current_f1": f"{current_f1_avg:.4f}",
        "oracle_em": f"{oracle_em_avg:.4f}",
        "oracle_f1": f"{oracle_f1_avg:.4f}",
        **dict(sorted(counts.items())),
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "qid",
            "question",
            "gold",
            "current",
            "current_rule",
            "current_em",
            "current_f1",
            "best_answer",
            "best_source",
            "best_field",
            "best_em",
            "best_f1",
            "classification",
            "exact_candidate_sources",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
