"""Restore HotpotQA contexts by id from the original validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _qid(item: dict) -> str:
    return str(item.get("id") or item.get("_id"))


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def _support_title_missing_count(items: list[dict]) -> tuple[int, int]:
    total = 0
    missing = 0
    for item in items:
        context = item.get("context") or {}
        if isinstance(context, dict):
            titles = set(context.get("title", []))
        else:
            titles = {row[0] for row in context if row}
        support = item.get("supporting_facts") or {}
        for title in support.get("title", []):
            total += 1
            if title not in titles:
                missing += 1
    return missing, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("comagraag/data/hotpotqa_sample.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    from datasets import load_dataset

    source = _load_json(args.input)
    dataset = load_dataset("hotpot_qa", "distractor", split=args.split)
    by_id = {_qid(item): item for item in dataset}

    repaired = []
    missing_ids = []
    for item in source:
        qid = _qid(item)
        hf_item = by_id.get(qid)
        if hf_item is None:
            missing_ids.append(qid)
            repaired.append(item)
            continue
        updated = dict(item)
        updated["context"] = hf_item["context"]
        repaired.append(updated)

    before_missing, before_total = _support_title_missing_count(source)
    after_missing, after_total = _support_title_missing_count(repaired)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(repaired, f, ensure_ascii=False, indent=2)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"items={len(repaired)} missing_ids={len(missing_ids)}")
    print(f"support_title_missing_before={before_missing}/{before_total}")
    print(f"support_title_missing_after={after_missing}/{after_total}")
    if missing_ids:
        print("missing_id_sample=" + ",".join(missing_ids[:10]))


if __name__ == "__main__":
    main()
