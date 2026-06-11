#!/usr/bin/env python3
"""Apply deterministic fallback selection to evidence-reader outputs."""

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
    "none",
    "cannot",
    "not enough",
    "no information",
    "nothing",
    "neither",
    "no triple",
    "not specified",
)


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
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("_id") or row.get("id") or row.get("qid")): row for row in rows}


def answer(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def looks_bad(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    return not lowered or any(marker in lowered for marker in BAD_MARKERS)


def looks_long(value: str, max_words: int) -> bool:
    return len(str(value or "").split()) > max_words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--max-reader-words", type=int, default=8)
    args = parser.parse_args()

    reader_rows = load_jsonl(args.reader)
    guarded = by_qid(load_jsonl(args.guarded))
    judge = by_qid(load_jsonl(args.judge))
    prior = by_qid(load_jsonl(args.prior))

    rows = []
    for reader_row in reader_rows:
        qid = str(reader_row.get("_id") or reader_row.get("id") or reader_row.get("qid"))
        reader_answer = answer(reader_row, "answer", "prediction")
        guarded_answer = answer(guarded.get(qid, {}), "answer", "prediction")
        judge_answer = answer(judge.get(qid, {}), "judge_answer", "answer")
        context_answer = answer(judge.get(qid, {}), "context")
        prior_answer = answer(prior.get(qid, {}), "answer", "prediction")

        reader_norm = normalize(reader_answer)
        agrees_with_candidate = any(
            reader_norm == normalize(candidate)
            for candidate in (guarded_answer, judge_answer, context_answer, prior_answer)
            if candidate
        )

        if not agrees_with_candidate and not looks_bad(prior_answer):
            selected = prior_answer
            source = "contextual_prior_v2"
        elif looks_bad(reader_answer) or looks_long(reader_answer, args.max_reader_words):
            selected = guarded_answer
            source = "guarded"
        else:
            selected = reader_answer
            source = "evidence_reader"

        gold = str(reader_row.get("gold") or guarded.get(qid, {}).get("gold") or "")
        row = {
            "_id": qid,
            "id": qid,
            "mode": "evidence_reader_fallback_selector",
            "question": reader_row.get("question", guarded.get(qid, {}).get("question", "")),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_source": source,
            "reader_answer": reader_answer,
            "guarded_answer": guarded_answer,
            "judge_answer": judge_answer,
            "context_answer": context_answer,
            "contextual_prior_v2": prior_answer,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        rows.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "evidence_reader_fallback_selector",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_reader": sum(row["selected_source"] == "evidence_reader" for row in rows),
        "selected_guarded": sum(row["selected_source"] == "guarded" for row in rows),
        "selected_contextual_prior": sum(row["selected_source"] == "contextual_prior_v2" for row in rows),
    }
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"EM={summary['EM']:.4f} F1={summary['F1']:.4f} n={summary['n']}")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
