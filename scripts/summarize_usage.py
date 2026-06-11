import argparse
import csv
import json
import sys
from pathlib import Path


def _load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return [data]
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input extension: {path}")


def _num(row: dict, *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _has_source_usage(row: dict) -> bool:
    return any(str(key).startswith("source_") for key in row)


def _source_num(row: dict, key: str) -> float:
    source_key = f"source_{key}"
    if source_key in row:
        return _num(row, source_key)
    return _num(row, key)


def _end_to_end_num(row: dict, key: str) -> float:
    current = _num(row, key)
    if _has_source_usage(row):
        return current + _num(row, f"source_{key}")
    return current


def _llm_calls(row: dict) -> float:
    explicit = _num(row, "llm_calls")
    if explicit:
        return explicit
    loops = _num(row, "loops")
    if not loops:
        return 0.0
    postprocess = 1.0 if _num(row, "answer_postprocess_cost") or row.get("raw_pred_answer") else 0.0
    forced = 1.0 if row.get("max_loops_exceeded") or row.get("token_budget_exceeded") else 0.0
    return loops + postprocess + forced


def _source_llm_calls(row: dict) -> float:
    if "source_llm_calls" in row:
        return _num(row, "source_llm_calls")
    return _llm_calls(row)


def _end_to_end_llm_calls(row: dict) -> float:
    if _has_source_usage(row):
        return _llm_calls(row) + _num(row, "source_llm_calls")
    return _llm_calls(row)


def summarize(path: Path) -> dict:
    rows = _load_records(path)
    fresh = [r for r in rows if not str(r.get("cache_hit", "")).lower() == "true"]
    source = fresh or rows
    n = len(source)
    if n == 0:
        return {"file": str(path), "n": 0}

    return {
        "file": str(path),
        "n": n,
        "avg_llm_calls": round(sum(_llm_calls(r) for r in source) / n, 3),
        "avg_input_tokens": round(sum(_num(r, "input_tokens", "prompt_tokens") for r in source) / n, 1),
        "avg_output_tokens": round(sum(_num(r, "output_tokens", "completion_tokens") for r in source) / n, 1),
        "avg_total_tokens": round(sum(_num(r, "total_tokens") for r in source) / n, 1),
        "avg_wall_time_s": round(sum(_num(r, "wall_time", "wall_time_s") for r in source) / n, 3),
        "avg_iter_or_loops": round(sum(_num(r, "iterations", "loops") for r in source) / n, 3),
        "avg_retrieved_tokens": round(sum(_num(r, "total_retrieved_tokens") for r in source) / n, 1),
        "avg_total_cost": round(sum(_num(r, "total_cost") for r in source) / n, 6),
        "avg_source_llm_calls": round(sum(_source_llm_calls(r) for r in source) / n, 3),
        "avg_source_input_tokens": round(sum(_source_num(r, "input_tokens") for r in source) / n, 1),
        "avg_source_output_tokens": round(sum(_source_num(r, "output_tokens") for r in source) / n, 1),
        "avg_source_total_tokens": round(sum(_source_num(r, "total_tokens") for r in source) / n, 1),
        "avg_source_wall_time_s": round(sum(_source_num(r, "wall_time") for r in source) / n, 3),
        "avg_end_to_end_llm_calls": round(sum(_end_to_end_llm_calls(r) for r in source) / n, 3),
        "avg_end_to_end_total_tokens": round(sum(_end_to_end_num(r, "total_tokens") for r in source) / n, 1),
        "avg_end_to_end_wall_time_s": round(sum(_end_to_end_num(r, "wall_time") for r in source) / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.paths]
    fieldnames = [
        "file",
        "n",
        "avg_llm_calls",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "avg_wall_time_s",
        "avg_iter_or_loops",
        "avg_retrieved_tokens",
        "avg_total_cost",
        "avg_source_llm_calls",
        "avg_source_input_tokens",
        "avg_source_output_tokens",
        "avg_source_total_tokens",
        "avg_source_wall_time_s",
        "avg_end_to_end_llm_calls",
        "avg_end_to_end_total_tokens",
        "avg_end_to_end_wall_time_s",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            out = csv.DictWriter(f, fieldnames=fieldnames)
            out.writeheader()
            out.writerows(rows)


if __name__ == "__main__":
    main()
