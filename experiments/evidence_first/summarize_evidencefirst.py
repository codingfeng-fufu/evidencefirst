import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_summary_csv(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _rate(count: int, total: int) -> str:
    return f"{count / total:.4f}" if total else "0.0000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="EvidenceFirst per-example JSONL")
    parser.add_argument("--csv", type=Path, default=None, help="Optional run summary CSV")
    args = parser.parse_args()

    rows = _load_jsonl(args.jsonl)
    summary = _load_summary_csv(args.csv)
    n = len(rows)

    complete = sum(1 for row in rows if row.get("evidence_first_chain_complete"))
    missing_total = sum(int(row.get("evidence_first_missing_entity_count") or 0) for row in rows)
    disconnected_total = sum(int(row.get("evidence_first_disconnected_pair_count") or 0) for row in rows)
    repair_attempts = sum(1 for row in rows if row.get("evidence_first_repair_steps"))
    fallback_cases = sum(1 for row in rows if row.get("evidence_first_fallback_steps"))

    print("metric,value")
    if summary:
        for key in ("EM", "F1", "avg_llm_calls", "avg_input_tokens", "avg_output_tokens", "avg_latency_s"):
            print(f"{key},{summary.get(key, '')}")
    print(f"n,{n}")
    print(f"chain_complete_count,{complete}")
    print(f"chain_complete_rate,{_rate(complete, n)}")
    print(f"repair_attempt_count,{repair_attempts}")
    print(f"repair_attempt_rate,{_rate(repair_attempts, n)}")
    print(f"fallback_count,{fallback_cases}")
    print(f"fallback_rate,{_rate(fallback_cases, n)}")
    print(f"missing_entity_total,{missing_total}")
    print(f"disconnected_pair_total,{disconnected_total}")

    for key in (
        "evidence_first_answer_type",
        "evidence_first_gap_type",
        "evidence_first_repair_steps",
    ):
        counts = Counter(str(row.get(key, "")) for row in rows)
        for value, count in sorted(counts.items()):
            label = value.replace(",", ";") if value else "<empty>"
            print(f"{key}:{label},{count}")


if __name__ == "__main__":
    main()
