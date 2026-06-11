#!/usr/bin/env python3
"""Apply a stricter second gate to strict quote reader outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


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


def reason_keeps_current(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in (
            "current answer is correct",
            "current answer is already",
            "current answer is supported",
            "should not be switched",
            "no switch needed",
            "current is correct",
            "current is already correct",
        )
    )


def quote_for_answer(answer: str, quotes: list[str]) -> str:
    answer_norm = normalize(answer)
    for quote in quotes:
        if answer_norm and answer_norm in normalize(quote):
            return quote
    return ""


def film_type_mismatch(question: str, answer: str, quotes: list[str]) -> bool:
    if "which film" not in question.lower():
        return False
    quote = quote_for_answer(answer, quotes)
    return bool(quote) and "film" not in quote.lower()


def accept(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("acceptance") != "accepted":
        return False, row.get("acceptance", "")
    current = str(row.get("current_answer") or "")
    proposed = str(row.get("reader_answer") or row.get("answer") or "")
    if normalize(current) == normalize(proposed):
        return False, "same_as_current"
    reason = str(row.get("reader_reason") or "")
    if reason_keeps_current(reason):
        return False, "reason_keeps_current"
    quotes = [str(row.get("reader_quote_1") or ""), str(row.get("reader_quote_2") or "")]
    if film_type_mismatch(str(row.get("question") or ""), proposed, quotes):
        return False, "film_type_mismatch"
    return True, "accepted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    current_by_id = rows_by_qid(args.current)
    rows = []
    for i, row in enumerate(load_jsonl(args.reader)):
        row_qid = qid(row, i)
        base = current_by_id[row_qid]
        current = str(base.get("answer") or base.get("prediction") or "")
        ok, acceptance = accept(row)
        selected = str(row.get("reader_answer") or row.get("answer") or "") if ok else current
        gold = str(base.get("gold") or row.get("gold") or "")
        rows.append(
            {
                **row,
                "mode": "strict_quote_gate_v2",
                "answer": selected,
                "prediction": selected,
                "selected_rule": "strict_quote_switch" if ok else "current_v3",
                "acceptance": acceptance if row.get("llm_called") else "not_target",
                "current_answer": current,
                "em": em(selected, gold),
                "f1": round(f1(selected, gold), 4),
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "strict_quote_gate_v2",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4),
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4),
        "targets": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row["selected_rule"] == "strict_quote_switch" for row in rows),
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
