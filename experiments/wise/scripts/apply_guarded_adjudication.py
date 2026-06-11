import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


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


def looks_bad(answer: str) -> bool:
    lowered = str(answer or "").lower().strip()
    return not lowered or any(marker in lowered for marker in BAD_MARKERS)


def subset_shorter(current: str, candidate: str) -> bool:
    current_tokens = set(normalize(current).split())
    candidate_tokens = set(normalize(candidate).split())
    return bool(candidate_tokens and candidate_tokens < current_tokens and not looks_bad(current))


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(row.get("_id") or row.get("id") or row.get("qid")): row for row in rows}


def current_answer(row: dict) -> str:
    return str(row.get("answer") or row.get("prediction") or "")


def select_answer(current_row: dict, judge_row: dict, prior_row: dict) -> tuple[str, str]:
    current = current_answer(current_row)
    judge = str(judge_row.get("judge_answer") or judge_row.get("answer") or "")
    prior = str(judge_row.get("prior") or "")

    use_judge = looks_bad(current) or (
        prior
        and normalize(prior) == normalize(judge)
        and normalize(judge) != normalize(current)
    )
    if use_judge and not subset_shorter(current, judge):
        selected = judge
        source = "judge_v2"
    else:
        selected = current
        source = "current_v4"

    contextual_prior = str(prior_row.get("answer") or prior_row.get("prediction") or "")
    if looks_bad(selected) and not looks_bad(contextual_prior):
        selected = contextual_prior
        source = "contextual_prior_v2"

    return selected, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--contextual-prior", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    current_rows = load_jsonl(args.current)
    judge_rows = load_jsonl(args.judge)
    prior_rows = load_jsonl(args.contextual_prior)

    rows = []
    for qid, current_row in current_rows.items():
        judge_row = judge_rows.get(qid, {})
        prior_row = prior_rows.get(qid, {})
        selected, source = select_answer(current_row, judge_row, prior_row)
        gold = str(current_row.get("gold") or judge_row.get("gold") or prior_row.get("gold") or "")
        row = {
            "_id": qid,
            "id": qid,
            "mode": "guarded_judge_contextual_prior_v2",
            "question": current_row.get("question", judge_row.get("question", "")),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "current_v4": current_answer(current_row),
            "judge_v2": judge_row.get("judge_answer", judge_row.get("answer", "")),
            "contextual_prior_v2": prior_row.get("answer", prior_row.get("prediction", "")),
            "selected_source": source,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        rows.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "guarded_judge_contextual_prior_v2",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "current_v4_EM": round(
            sum(em(row["current_v4"], row["gold"]) for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "judge_v2_EM": round(
            sum(em(row["judge_v2"], row["gold"]) for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "contextual_prior_v2_EM": round(
            sum(em(row["contextual_prior_v2"], row["gold"]) for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "selected_current": sum(row["selected_source"] == "current_v4" for row in rows),
        "selected_judge": sum(row["selected_source"] == "judge_v2" for row in rows),
        "selected_contextual_prior": sum(
            row["selected_source"] == "contextual_prior_v2" for row in rows
        ),
    }
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"EM={summary['EM']:.4f} F1={summary['F1']:.4f} n={summary['n']}")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
