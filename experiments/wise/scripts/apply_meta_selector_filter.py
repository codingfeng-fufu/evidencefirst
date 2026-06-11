#!/usr/bin/env python3
"""Apply deterministic acceptance rules over post-hoc CoMaGRAG selectors."""

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


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("_id") or row.get("id") or row.get("qid")): row for row in rows}


def answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("prediction") or "")


def choose(
    baseline_row: dict[str, Any],
    choice_row: dict[str, Any],
    source_calibrated_row: dict[str, Any],
) -> tuple[str, str]:
    """Return selected answer and the deterministic rule that accepted it."""

    baseline_answer = answer(baseline_row)
    question = str(baseline_row.get("question") or choice_row.get("question") or "").lower()

    sc_choice = str(source_calibrated_row.get("selected_choice") or "")
    if sc_choice == "contextual_prior" and not any(term in question for term in ("first", "older", "recent")):
        return answer(source_calibrated_row), "source_calibrated_contextual_prior_safe"
    if sc_choice == "reader_consensus":
        return answer(source_calibrated_row), "source_calibrated_reader_consensus"
    if sc_choice == "graph_consensus" and question.strip().startswith(("are both", "were both")):
        return answer(source_calibrated_row), "source_calibrated_graph_consensus_are_both"

    choice_source = str(choice_row.get("adjudicator_source") or "").lower()
    if (
        choice_row.get("adjudicator_confidence") == "medium"
        and ("guarded" in choice_source or "judge" in choice_source)
    ):
        return answer(choice_row), "candidate_choice_medium_guarded_or_judge"
    if "prior" in choice_source and "released first" not in question:
        return answer(choice_row), "candidate_choice_prior_not_released_first"

    return baseline_answer, "baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-choice", type=Path, required=True)
    parser.add_argument("--source-calibrated", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    choice_by_id = rows_by_qid(load_jsonl(args.candidate_choice))
    source_by_id = rows_by_qid(load_jsonl(args.source_calibrated))

    rows = []
    for baseline_row in baseline_rows:
        qid = str(baseline_row.get("_id") or baseline_row.get("id") or baseline_row.get("qid"))
        choice_row = choice_by_id.get(qid, {})
        source_row = source_by_id.get(qid, {})
        selected, rule = choose(baseline_row, choice_row, source_row)
        gold = str(baseline_row.get("gold") or choice_row.get("gold") or source_row.get("gold") or "")
        row = {
            "_id": qid,
            "id": qid,
            "mode": "meta_selector_filter",
            "question": baseline_row.get("question") or choice_row.get("question", ""),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_rule": rule,
            "baseline_answer": answer(baseline_row),
            "candidate_choice_answer": answer(choice_row),
            "source_calibrated_answer": answer(source_row),
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        rows.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "meta_selector_filter",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_baseline": sum(row["selected_rule"] == "baseline" for row in rows),
        "selected_source_calibrated_contextual_prior_safe": sum(
            row["selected_rule"] == "source_calibrated_contextual_prior_safe" for row in rows
        ),
        "selected_source_calibrated_reader_consensus": sum(
            row["selected_rule"] == "source_calibrated_reader_consensus" for row in rows
        ),
        "selected_source_calibrated_graph_consensus_are_both": sum(
            row["selected_rule"] == "source_calibrated_graph_consensus_are_both" for row in rows
        ),
        "selected_candidate_choice_medium_guarded_or_judge": sum(
            row["selected_rule"] == "candidate_choice_medium_guarded_or_judge" for row in rows
        ),
        "selected_candidate_choice_prior_not_released_first": sum(
            row["selected_rule"] == "candidate_choice_prior_not_released_first" for row in rows
        ),
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
