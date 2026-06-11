#!/usr/bin/env python3
"""Deterministic pure-internal CoMaGRAG selector v4.

The selector keeps the current answer by default and switches only to a
CoMaGRAG-internal candidate with strong multi-source support. Gold answers are
used only for reporting metrics after the selection is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
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
    "instructions",
    "using only",
    "but perhaps",
    "perhaps the answer",
    "the question is",
    "graph doesn",
)

EXCLUDED_FIELDS = {
    "_id",
    "id",
    "qid",
    "question",
    "gold",
    "gold_answer",
    "em",
    "f1",
    "raw_response",
    "error",
    "fallback_error",
    "mode",
    "variant",
    "type",
    "selected_source",
    "selected_choice",
    "selected_rule",
    "previous_rule",
    "original_rule",
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
}


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


def qid(row: dict[str, Any], fallback: int | str = "") -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def source_name(path: Path) -> str:
    return path.stem.replace("_predictions", "")


def source_kind(source: str) -> str:
    if "candidate_choice" in source:
        return "choice"
    if "contextual_prior_v2" in source:
        return "prior"
    if "guarded_judge" in source:
        return "guarded"
    if "evidence_reader_support_first" in source:
        return "support_first"
    if "evidence_reader" in source:
        return "reader"
    if "final_verifier" in source:
        return "verifier"
    if "meta_selector" in source:
        return "meta"
    return "other"


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    return (
        not normalize(value)
        or len(tokens(value)) > 12
        or len(lowered) > 180
        or lowered.startswith(("{", "- "))
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def minor_alias_change(current: str, candidate: str) -> bool:
    current_norm = normalize(current)
    candidate_norm = normalize(candidate)
    if not current_norm or not candidate_norm or current_norm == candidate_norm:
        return True
    current_tokens = current_norm.split()
    candidate_tokens = candidate_norm.split()
    if abs(len(current_tokens) - len(candidate_tokens)) > 4:
        return False
    if current_norm in candidate_norm or candidate_norm in current_norm:
        return True
    current_set = set(current_tokens)
    candidate_set = set(candidate_tokens)
    return current_set.issubset(candidate_set) or candidate_set.issubset(current_set)


def question_type(question: str) -> str:
    lowered = question.lower().strip()
    if lowered.startswith(("are ", "were ", "do ", "did ", "does ", "is ", "was ", "has ", "have ", "had ", "can ")):
        return "yesno"
    if any(
        marker in lowered
        for marker in (
            "born first",
            "born later",
            "older",
            "younger",
            "newer",
            "released first",
            "came out first",
            "appeared first",
            "closer",
            "nearer",
            "more ",
            "larger",
            "bigger",
            "which one",
            "between the two",
        )
    ):
        return "comparison"
    if any(marker in lowered for marker in ("what year", "how many", "when was", "when did", "period", "date", "population", "number")):
        return "number_date"
    if lowered.startswith("where") or any(marker in lowered for marker in ("what city", "what country", "what region", "located", "headquartered")):
        return "location"
    if lowered.startswith("who") or " who " in lowered[:160]:
        return "person"
    if " both " in f" {lowered} " or " in common" in lowered or "same " in lowered:
        return "common"
    return "other"


def is_candidate_field(field: str, value: Any) -> bool:
    if field in EXCLUDED_FIELDS or not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return False
    lowered = field.lower()
    if lowered in {
        "candidate_contextual_prior_v2",
        "contextual_prior_v2",
        "candidate_context",
        "context_answer",
        "candidate_kg",
        "kg",
        "judge_v2",
        "candidate_judge_v2",
        "candidate_choice_answer",
        "source_calibrated_answer",
    }:
        return True
    return lowered.startswith("candidate_") and any(
        marker in lowered
        for marker in (
            "contextual_prior_v2",
            "context_answer",
            "candidate_context",
            "reader_support",
            "reader_v1",
            "final_verifier",
            "guarded",
            "judge_v2",
            "kg",
            "context",
        )
    )


def field_weight(source: str, field: str) -> float:
    lowered = field.lower()
    weight = 1.0
    if "contextual_prior_v2" in lowered:
        weight = 6.0
    elif "candidate_context" in lowered or "context_answer" in lowered:
        weight = 5.0
    elif "candidate_choice_answer" in lowered:
        weight = 4.0
    elif "source_calibrated_answer" in lowered:
        weight = 4.0
    elif "reader_support" in lowered:
        weight = 3.0
    elif "reader_v1" in lowered:
        weight = 2.5
    elif "judge_v2" in lowered:
        weight = 2.0
    elif "candidate_kg" in lowered or lowered == "kg" or "|kg" in lowered:
        weight = 1.0

    kind_bonus = {
        "choice": 2.0,
        "prior": 1.5,
        "reader": 1.0,
        "support_first": 1.0,
        "verifier": 0.8,
        "guarded": 0.7,
        "meta": 0.3,
    }
    return weight + kind_bonus.get(source_kind(source), 0.0)


def score_candidate(question: str, answer: str, occurrences: list[tuple[str, str, str]]) -> float:
    if bad_surface(answer):
        return -999.0
    qtype = question_type(question)
    answer_norm = normalize(answer)
    fields = [field.lower() for _source, field, _value in occurrences]
    sources = [source for source, _field, _value in occurrences]
    score = sum(field_weight(source, field) for source, field, _value in occurrences)
    score += min(len(occurrences), 10) * 0.8
    score += len({source_kind(source) for source in sources}) * 1.5
    if any("contextual_prior_v2" in field for field in fields):
        score += 3.0
    if any("candidate_context" in field or "context_answer" in field for field in fields):
        score += 2.0
    if any(source_kind(source) == "choice" for source in sources):
        score += 2.0
    if any(source_kind(source) == "prior" for source in sources):
        score += 1.0
    if qtype == "yesno":
        if answer_norm in {"yes", "no"}:
            score += 4.0
        else:
            return -999.0
    elif answer_norm in {"yes", "no"}:
        score -= 5.0
    if qtype == "number_date":
        if re.search(r"\d", answer):
            score += 2.0
        if len(tokens(answer)) <= 4:
            score += 1.0
    if qtype == "comparison" and len(tokens(answer)) <= 6:
        score += 1.0
    if qtype in {"person", "location"} and len(tokens(answer)) <= 6:
        score += 0.5
    if len(tokens(answer)) > 8:
        score -= 2.0
    return score


def collect_candidates(paths: list[Path]) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    candidates: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        source = source_name(path)
        for i, row in enumerate(load_jsonl(path)):
            row_qid = qid(row, i)
            for field, value in row.items():
                if is_candidate_field(field, value) and not bad_surface(value):
                    candidates[row_qid][normalize(value)].append((source, field, str(value).strip()))
    return candidates


def choose(
    row: dict[str, Any],
    grouped_candidates: dict[str, list[tuple[str, str, str]]],
    min_score: float,
    min_margin: float,
    min_occurrences: int,
    allow_alias_switch: bool,
) -> tuple[str, str, float, float, int, str]:
    question = str(row.get("question") or "")
    current = str(row.get("prediction") or row.get("answer") or "")
    current_norm = normalize(current)
    current_occurrences = grouped_candidates.get(current_norm, [])
    current_score = score_candidate(question, current, current_occurrences) if current_occurrences else 0.0
    current_score += 10.0

    best: tuple[float, str, list[tuple[str, str, str]]] | None = None
    for candidate_norm, occurrences in grouped_candidates.items():
        if candidate_norm == current_norm or len(occurrences) < min_occurrences:
            continue
        candidate = occurrences[0][2]
        fields = [field.lower() for _source, field, _value in occurrences]
        if not any("contextual_prior_v2" in field or "candidate_context" in field or "context_answer" in field for field in fields):
            continue
        if minor_alias_change(current, candidate) and not allow_alias_switch:
            continue
        candidate_score = score_candidate(question, candidate, occurrences)
        if best is None or candidate_score > best[0]:
            best = (candidate_score, candidate, occurrences)

    if best is None:
        return current, "current_v4", 0.0, current_score, 0, ""

    candidate_score, candidate, occurrences = best
    margin = candidate_score - current_score
    if candidate_score >= min_score and margin >= min_margin:
        qtype = question_type(question)
        sources = "|".join(f"{source}:{field}" for source, field, _value in occurrences[:8])
        return candidate, f"v4_conflict_{qtype}", candidate_score, current_score, len(occurrences), sources

    sources = "|".join(f"{source}:{field}" for source, field, _value in occurrences[:8])
    return current, "current_v4", candidate_score, current_score, len(occurrences), sources


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "comagraag_body_selector_v4",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_current_v4": sum(row["selected_rule"] == "current_v4" for row in rows),
        "switches": sum(row["selected_rule"] != "current_v4" for row in rows),
    }
    summary.update({f"selected_{rule}": count for rule, count in Counter(row["selected_rule"] for row in rows).items()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=150.0)
    parser.add_argument("--min-margin", type=float, default=100.0)
    parser.add_argument("--min-occurrences", type=int, default=16)
    parser.add_argument("--allow-alias-switch", action="store_true")
    args = parser.parse_args()

    current_rows = load_jsonl(args.current)
    candidates = collect_candidates(args.candidate)

    rows = []
    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        selected, selected_rule, candidate_score, current_score, support_count, support_sources = choose(
            row,
            candidates.get(row_qid, {}),
            min_score=args.min_score,
            min_margin=args.min_margin,
            min_occurrences=args.min_occurrences,
            allow_alias_switch=args.allow_alias_switch,
        )
        gold = str(row.get("gold") or "")
        rows.append(
            {
                "_id": row_qid,
                "id": row_qid,
                "mode": "comagraag_body_selector_v4",
                "question": row.get("question", ""),
                "answer": selected,
                "prediction": selected,
                "gold": gold,
                "selected_rule": selected_rule,
                "current_answer": row.get("prediction") or row.get("answer") or "",
                "candidate_score": round(candidate_score, 4),
                "current_score": round(current_score, 4),
                "support_count": support_count,
                "support_sources": support_sources,
                "em": em(selected, gold),
                "f1": round(f1(selected, gold), 4),
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(rows, args.out_csv)
    print(f"EM={sum(row['em'] for row in rows) / len(rows):.4f} F1={sum(row['f1'] for row in rows) / len(rows):.4f} n={len(rows)}")
    print(f"switches={sum(row['selected_rule'] != 'current_v4' for row in rows)}")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
