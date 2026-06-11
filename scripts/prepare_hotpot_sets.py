import argparse
import json
import random
from pathlib import Path


def _qid(item: dict) -> str:
    return str(item.get("id", item.get("_id")))


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _passages_from_context(context) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return context
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles = [t for t, _ in context]
        sentences = [s for _, s in context]
    return [f"{title}: {' '.join(sents)}" for title, sents in zip(titles, sentences)]


def _jsonl_record(item: dict) -> dict:
    qid = _qid(item)
    return {
        "_id": qid,
        "id": qid,
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type", ""),
        "context": _passages_from_context(item["context"]),
    }


def _write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(_jsonl_record(item), ensure_ascii=False) + "\n")


def _sample_balanced(ds, exclude_ids: set[str], n_bridge: int, n_comparison: int, seed: int) -> list[dict]:
    bridge = [dict(x) for x in ds if x.get("type") == "bridge" and _qid(x) not in exclude_ids]
    comparison = [dict(x) for x in ds if x.get("type") == "comparison" and _qid(x) not in exclude_ids]

    if len(bridge) < n_bridge:
        raise ValueError(f"Not enough bridge examples: need {n_bridge}, found {len(bridge)}")
    if len(comparison) < n_comparison:
        raise ValueError(f"Not enough comparison examples: need {n_comparison}, found {len(comparison)}")

    rng = random.Random(seed)
    sampled = rng.sample(bridge, n_bridge) + rng.sample(comparison, n_comparison)
    rng.shuffle(sampled)
    return sampled


def _assert_no_overlap(name: str, left: list[dict], right: list[dict]) -> None:
    overlap = {_qid(x) for x in left} & {_qid(x) for x in right}
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"{name} overlap is not zero: {len(overlap)} ids, sample={sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-test", type=Path, default=Path("comagraag/data/hotpotqa_sample.json"))
    parser.add_argument("--existing-val", type=Path, default=Path("comagraag/data/hotpotqa_val.json"))
    parser.add_argument("--out-test-json", type=Path, default=Path("data/hotpotqa_1000_test.json"))
    parser.add_argument("--out-test-jsonl", type=Path, default=Path("data/hotpotqa_1000_test.jsonl"))
    parser.add_argument("--out-existing-jsonl", type=Path, default=Path("data/hotpotqa_500_test.jsonl"))
    parser.add_argument("--out-ids", type=Path, default=Path("data/hotpotqa_1000_ids.txt"))
    parser.add_argument("--bridge", type=int, default=500)
    parser.add_argument("--comparison", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-existing", action="store_true", help="Only export the existing 500-question JSONL")
    args = parser.parse_args()

    existing_test = _load_json(args.existing_test)
    existing_val = _load_json(args.existing_val)
    _write_jsonl(args.out_existing_jsonl, existing_test)

    if args.only_existing:
        print(f"Wrote existing 500 JSONL to {args.out_existing_jsonl}")
        return

    exclude_ids = {_qid(x) for x in existing_test} | {_qid(x) for x in existing_val}

    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)
    sampled = _sample_balanced(ds, exclude_ids, args.bridge, args.comparison, args.seed)

    _assert_no_overlap("1000 vs existing test", sampled, existing_test)
    _assert_no_overlap("1000 vs existing val", sampled, existing_val)

    _write_json(args.out_test_json, sampled)
    _write_jsonl(args.out_test_jsonl, sampled)

    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_ids, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(_qid(item) + "\n")

    print(f"Wrote {len(sampled)} examples to {args.out_test_json}")
    print(f"Wrote JSONL to {args.out_test_jsonl}")
    print(f"Wrote existing 500 JSONL to {args.out_existing_jsonl}")
    print(f"bridge={sum(x.get('type') == 'bridge' for x in sampled)} comparison={sum(x.get('type') == 'comparison' for x in sampled)}")


if __name__ == "__main__":
    main()
