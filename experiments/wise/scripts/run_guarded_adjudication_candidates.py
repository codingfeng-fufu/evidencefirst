#!/usr/bin/env python3
"""Generate guarded adjudication candidates for completed CoMaGRAG predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comagraag"))

import config  # noqa: E402


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

client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)


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


def current_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("prediction") or "")


def select_answer(current_row: dict[str, Any], judge_row: dict[str, Any], prior_row: dict[str, Any]) -> tuple[str, str]:
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


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("_id") or row.get("id") or row.get("qid")): row
        for row in rows
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_candidates(cache_entry: dict[str, Any], current: str) -> tuple[str, str]:
    history = cache_entry.get("history") or []
    kg_entries = [row for row in history if row.get("iteration") != "context_fallback"]
    context_entry = next((row for row in history if row.get("iteration") == "context_fallback"), {})
    kg_answer = str((kg_entries[-1] if kg_entries else {}).get("answer") or current or "")
    context_answer = str(context_entry.get("answer") or "")
    return kg_answer, context_answer


def usage_from_response(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
    total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def llm_text(prompt: str, max_tokens: int, retries: int = 4) -> tuple[str, dict[str, int]]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            return text.strip(), usage_from_response(resp)
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2 ** attempt)
    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def parse_json_obj(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def prior_prompt(question: str) -> str:
    return f"""Answer this multi-hop QA question from your parametric knowledge only.

Rules:
- Return only the concise final answer.
- Prefer exact names, dates, numbers, and yes/no answers.
- If you are not confident, return "unknown".
- Do not explain.

Question: {question}
Answer:"""


def judge_prompt(question: str, current: str, kg: str, context: str) -> str:
    return f"""You are adjudicating answer candidates for a multi-hop QA benchmark.

Question:
{question}

Candidate answers:
- current_v4: {current}
- kg: {kg}
- context: {context}

Tasks:
1. Give a question-only prior answer only if you are confident; otherwise use an empty string.
2. Choose the best final answer. Prefer specific, concise answers. Avoid refusals such as "unknown" when another candidate is usable.

Return strict JSON only:
{{"prior": "...", "judge_answer": "..."}}
"""


def write_guarded_outputs(
    current_rows: list[dict[str, Any]],
    judge_rows: dict[str, dict[str, Any]],
    prior_rows: dict[str, dict[str, Any]],
    out_jsonl: Path,
    out_csv: Path,
) -> None:
    rows = []
    for current_row in current_rows:
        qid = str(current_row.get("_id") or current_row.get("id") or current_row.get("qid"))
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

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "guarded_judge_contextual_prior_v2",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "current_v4_EM": round(sum(em(row["current_v4"], row["gold"]) for row in rows) / len(rows), 4) if rows else 0,
        "judge_v2_EM": round(sum(em(row["judge_v2"], row["gold"]) for row in rows) / len(rows), 4) if rows else 0,
        "contextual_prior_v2_EM": round(
            sum(em(row["contextual_prior_v2"], row["gold"]) for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "selected_current": sum(row["selected_source"] == "current_v4" for row in rows),
        "selected_judge": sum(row["selected_source"] == "judge_v2" for row in rows),
        "selected_contextual_prior": sum(row["selected_source"] == "contextual_prior_v2" for row in rows),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-judge-jsonl", type=Path, required=True)
    parser.add_argument("--out-prior-jsonl", type=Path, required=True)
    parser.add_argument("--out-guarded-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--expected-n", type=int, default=None)
    args = parser.parse_args()

    current_rows = load_jsonl_rows(args.current)
    if args.expected_n is not None and len(current_rows) < args.expected_n:
        raise SystemExit(f"current predictions incomplete: {len(current_rows)}/{args.expected_n}")

    cache = load_cache(args.cache)
    judge_rows = rows_by_qid(load_jsonl_rows(args.out_judge_jsonl))
    prior_rows = rows_by_qid(load_jsonl_rows(args.out_prior_jsonl))

    for index, current_row in enumerate(current_rows, start=1):
        qid = str(current_row.get("_id") or current_row.get("id") or current_row.get("qid"))
        question = str(current_row.get("question") or "")
        gold = str(current_row.get("gold") or "")
        current = current_answer(current_row)
        kg_answer, context_answer = extract_candidates(cache.get(qid, {}), current)

        usage_record = {
            "_id": qid,
            "mode": "guarded_judge_contextual_prior_v2",
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_hit": False,
        }
        started = time.time()

        if qid not in prior_rows:
            prior_error = None
            try:
                prior_answer, prior_usage = llm_text(prior_prompt(question), max_tokens=80)
            except Exception as exc:
                prior_answer = "unknown"
                prior_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                prior_error = str(exc)
            prior_row = {
                "_id": qid,
                "question": question,
                "gold": gold,
                "answer": prior_answer,
                "em": em(prior_answer, gold),
                "f1": round(f1(prior_answer, gold), 4),
                "error": prior_error,
            }
            append_jsonl(args.out_prior_jsonl, prior_row)
            prior_rows[qid] = prior_row
            usage_record["llm_calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                usage_record[key] += prior_usage[key]

        if qid not in judge_rows:
            judge_error = None
            try:
                raw_judge, judge_usage = llm_text(judge_prompt(question, current, kg_answer, context_answer), max_tokens=160)
                obj = parse_json_obj(raw_judge)
                prior = str(obj.get("prior") or "")
                judge_answer = str(obj.get("judge_answer") or obj.get("answer") or raw_judge).strip()
            except Exception as exc:
                raw_judge = ""
                judge_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                prior = ""
                judge_answer = current
                judge_error = str(exc)
            judge_row = {
                "_id": qid,
                "question": question,
                "gold": gold,
                "current_v4": current,
                "kg": kg_answer,
                "context": context_answer,
                "prior": prior,
                "judge_answer": judge_answer,
                "em": em(judge_answer, gold),
                "f1": round(f1(judge_answer, gold), 4),
                "error": judge_error,
            }
            append_jsonl(args.out_judge_jsonl, judge_row)
            judge_rows[qid] = judge_row
            usage_record["llm_calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                usage_record[key] += judge_usage[key]

        usage_record["wall_time"] = round(time.time() - started, 4)
        if args.usage_log and usage_record["llm_calls"]:
            append_jsonl(args.usage_log, usage_record)

        if index % 25 == 0:
            print(f"[{index}/{len(current_rows)}] guarded candidates generated", flush=True)

    write_guarded_outputs(
        current_rows=current_rows,
        judge_rows=judge_rows,
        prior_rows=prior_rows,
        out_jsonl=args.out_guarded_jsonl,
        out_csv=args.out_csv,
    )
    print(f"Wrote {args.out_guarded_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
