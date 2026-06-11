#!/usr/bin/env python3
"""Apply a deterministic baseline-consensus selector over CoMaGRAG v2.

The selector keeps CoMaGRAG v2 by default. It switches only when selected
external baseline pairs agree on the same non-current answer and the switch does
not look like a minor alias/surface-form change of the current answer.

Gold answers are used only for reporting metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


PAIR_PRIORITY = [
    ("arag", "hoprag"),
    ("hoprag", "naive"),
    ("arag", "ircot"),
    ("arag", "naive"),
    ("hoprag", "ircot"),
]

INTERNAL_KG_FIELDS = {
    "kg",
    "candidate_kg",
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


def qid(row: dict[str, Any], fallback: int = 0) -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rows_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    return {qid(row, i): row for i, row in enumerate(load_jsonl(path))}


def collect_internal_kg_candidates(paths: list[Path]) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    for path in paths:
        source = path.stem
        for i, row in enumerate(load_jsonl(path)):
            row_qid = qid(row, i)
            bucket = candidates.setdefault(row_qid, set())
            for field, value in row.items():
                if not isinstance(value, str):
                    continue
                if field in INTERNAL_KG_FIELDS or "kg" in field.lower() or "kg" in source.lower():
                    answer = value.strip()
                    if answer and len(answer) < 180 and not bad_surface(answer):
                        bucket.add(normalize(answer))
    return candidates


def pred_for(name: str, row: dict[str, Any]) -> str:
    if not row:
        return ""
    if name == "arag":
        return str(row.get("pred_answer") or row.get("prediction") or "")
    if name == "hoprag":
        return str(row.get("response") or row.get("prediction") or "")
    return str(row.get("answer") or row.get("prediction") or row.get("response") or "")


def bad_surface(answer: str) -> bool:
    lowered = str(answer or "").lower()
    if any(
        marker in lowered
        for marker in (
            "context passage",
            "provided context",
            "not enough",
            "insufficient",
            "none of",
            "does not mention",
            "do not mention",
            "cannot determine",
        )
    ):
        return True
    return len(tokens(answer)) > 12


def minor_alias_change(current: str, candidate: str) -> bool:
    current_norm = normalize(current)
    candidate_norm = normalize(candidate)
    if not current_norm or not candidate_norm or current_norm == candidate_norm:
        return False
    current_tokens = current_norm.split()
    candidate_tokens = candidate_norm.split()
    if abs(len(current_tokens) - len(candidate_tokens)) > 4:
        return False
    if current_norm in candidate_norm or candidate_norm in current_norm:
        return True
    current_set = set(current_tokens)
    candidate_set = set(candidate_tokens)
    return current_set.issubset(candidate_set) or candidate_set.issubset(current_set)


def choose(
    current: str,
    candidates: dict[str, str],
    pair_priority: list[tuple[str, str]],
    internal_kg_norms: set[str] | None = None,
) -> tuple[str, str]:
    current_norm = normalize(current)
    for left, right in pair_priority:
        left_answer = candidates.get(left, "")
        right_answer = candidates.get(right, "")
        if not left_answer or not right_answer:
            continue
        if normalize(left_answer) != normalize(right_answer):
            continue
        if normalize(left_answer) == current_norm:
            continue
        if bad_surface(left_answer):
            continue
        if minor_alias_change(current, left_answer):
            continue
        return left_answer, f"agree_{left}_{right}"
    hoprag_answer = candidates.get("hoprag", "")
    if (
        internal_kg_norms
        and normalize(hoprag_answer)
        and normalize(hoprag_answer) in internal_kg_norms
        and normalize(hoprag_answer) != current_norm
        and not bad_surface(hoprag_answer)
        and not minor_alias_change(current, hoprag_answer)
    ):
        return hoprag_answer, "agree_hoprag_internal_kg"
    return current, "current_v2"


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "comagraag_hybrid_baseline_consensus",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_current_v2": sum(row["selected_rule"] == "current_v2" for row in rows),
        "switches": sum(row["selected_rule"] != "current_v2" for row in rows),
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
    parser.add_argument("--arag", type=Path, required=True)
    parser.add_argument("--hoprag", type=Path, required=True)
    parser.add_argument("--ircot", type=Path, required=True)
    parser.add_argument("--naive", type=Path, required=True)
    parser.add_argument("--internal-candidate", action="append", type=Path, default=[])
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    current_rows = load_jsonl(args.current)
    external = {
        "arag": rows_by_qid(args.arag),
        "hoprag": rows_by_qid(args.hoprag),
        "ircot": rows_by_qid(args.ircot),
        "naive": rows_by_qid(args.naive),
    }
    internal_kg = collect_internal_kg_candidates(args.internal_candidate)

    rows = []
    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        current_answer = str(row.get("answer") or row.get("prediction") or "")
        candidates = {
            name: pred_for(name, table.get(row_qid, {}))
            for name, table in external.items()
        }
        selected, selected_rule = choose(
            current_answer,
            candidates,
            PAIR_PRIORITY,
            internal_kg_norms=internal_kg.get(row_qid, set()),
        )
        gold = str(row.get("gold") or "")
        rows.append(
            {
                "_id": row_qid,
                "id": row_qid,
                "mode": "comagraag_hybrid_baseline_consensus",
                "question": row.get("question", ""),
                "answer": selected,
                "prediction": selected,
                "gold": gold,
                "selected_rule": selected_rule,
                "current_answer": current_answer,
                "arag_answer": candidates.get("arag", ""),
                "hoprag_answer": candidates.get("hoprag", ""),
                "ircot_answer": candidates.get("ircot", ""),
                "naive_answer": candidates.get("naive", ""),
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
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
