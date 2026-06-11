#!/usr/bin/env python3
"""Merge internal targeted replacement predictions onto a base run.

The replacement run must be produced by CoMaGRAG-internal candidates only. This
script does not decide which replacements are correct; it simply applies
non-current replacements from the replacement artifact and reports metrics.
Gold answers are used only for reporting.
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
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or "")


def is_replacement(row: dict[str, Any]) -> bool:
    current = str(row.get("current_answer") or "")
    pred = prediction(row)
    if not current:
        return str(row.get("selected_rule") or "") not in {"", "current_v2", "current", "current_v4"}
    return normalize(pred) != normalize(current)


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": rows[0]["mode"] if rows else "internal_targeted_replacements",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "replacements": sum(row["selected_rule"] == "targeted_internal_replacement" for row in rows),
    }
    summary.update({f"source_{key}": value for key, value in Counter(row.get("merged_from", "") for row in rows).items()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--mode", default="kg_gate_plus_internal_targeted_replacements")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    replacement_by_id = {qid(row, i): row for i, row in enumerate(load_jsonl(args.replacement)) if is_replacement(row)}

    rows = []
    for i, base in enumerate(base_rows):
        row_qid = qid(base, i)
        repl = replacement_by_id.get(row_qid)
        selected = prediction(repl) if repl else prediction(base)
        gold = str(base.get("gold") or (repl or {}).get("gold") or "")
        selected_rule = "targeted_internal_replacement" if repl else "base"
        rows.append(
            {
                "_id": row_qid,
                "id": row_qid,
                "mode": args.mode,
                "question": base.get("question", (repl or {}).get("question", "")),
                "answer": selected,
                "prediction": selected,
                "gold": gold,
                "selected_rule": selected_rule,
                "base_answer": prediction(base),
                "replacement_answer": prediction(repl) if repl else "",
                "replacement_rule": (repl or {}).get("selected_rule", ""),
                "merged_from": "replacement" if repl else "base",
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
    print(f"replacements={sum(row['selected_rule'] == 'targeted_internal_replacement' for row in rows)}")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
