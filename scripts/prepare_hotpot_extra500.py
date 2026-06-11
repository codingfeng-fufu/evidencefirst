import argparse
import json
import random
from pathlib import Path


def _load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset shape: {path}")


def _qid(item: dict, fallback: int = 0) -> str:
    return str(item.get("_id") or item.get("id") or item.get("qid") or fallback)


def _context_for_jsonl(item: dict) -> list[str]:
    context = item.get("context", [])
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return context
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [
            f"{title}: {' '.join(str(s).strip() for s in sents if str(s).strip())}"
            for title, sents in zip(titles, sentences)
        ]
    rows = []
    for title, sentences in context:
        if isinstance(sentences, str):
            sentences = [sentences]
        rows.append(f"{title}: {' '.join(str(s).strip() for s in sentences if str(s).strip())}")
    return rows


def _jsonl_record(item: dict, fallback: int) -> dict:
    qid = _qid(item, fallback)
    return {
        "_id": qid,
        "id": qid,
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type", ""),
        "context": _context_for_jsonl(item),
    }


def _sample_balanced(pool: list[dict], bridge: int, comparison: int, seed: int) -> list[dict]:
    bridge_rows = [row for row in pool if row.get("type") == "bridge"]
    comparison_rows = [row for row in pool if row.get("type") == "comparison"]
    if len(bridge_rows) < bridge:
        raise ValueError(f"Not enough bridge rows: need {bridge}, found {len(bridge_rows)}")
    if len(comparison_rows) < comparison:
        raise ValueError(f"Not enough comparison rows: need {comparison}, found {len(comparison_rows)}")

    rng = random.Random(seed)
    sampled_ids = {
        _qid(row, idx)
        for idx, row in enumerate(rng.sample(bridge_rows, bridge) + rng.sample(comparison_rows, comparison))
    }
    return [row for idx, row in enumerate(pool) if _qid(row, idx) in sampled_ids]


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps(_jsonl_record(row, i), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/hotpotqa_1000_test.json"))
    parser.add_argument("--existing", type=Path, default=Path("data/hotpotqa_500_test.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("data/hotpotqa_extra500_test.json"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/hotpotqa_extra500_test.jsonl"))
    parser.add_argument("--out-ids", type=Path, default=Path("data/hotpotqa_extra500_ids.txt"))
    parser.add_argument("--bridge", type=int, default=250)
    parser.add_argument("--comparison", type=int, default=250)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    pool = _load_records(args.pool)
    existing = _load_records(args.existing)
    existing_ids = {_qid(row, i) for i, row in enumerate(existing)}

    sampled = _sample_balanced(pool, args.bridge, args.comparison, args.seed)
    sampled_ids = [_qid(row, i) for i, row in enumerate(sampled)]
    overlap = set(sampled_ids) & existing_ids
    if overlap:
        raise ValueError(f"Extra set overlaps existing 500: {len(overlap)} ids, sample={sorted(overlap)[:5]}")

    _write_json(args.out_json, sampled)
    _write_jsonl(args.out_jsonl, sampled)
    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_ids.write_text("\n".join(sampled_ids) + "\n", encoding="utf-8")

    print(f"Wrote {len(sampled)} examples to {args.out_json}")
    print(f"Wrote JSONL to {args.out_jsonl}")
    print(f"Wrote ids to {args.out_ids}")
    print(f"bridge={sum(row.get('type') == 'bridge' for row in sampled)}")
    print(f"comparison={sum(row.get('type') == 'comparison' for row in sampled)}")
    print(f"overlap_with_existing={len(overlap)}")


if __name__ == "__main__":
    main()
