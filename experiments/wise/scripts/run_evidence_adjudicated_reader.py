#!/usr/bin/env python3
"""Evidence-grounded final reader over CoMaGRAG candidate answers."""

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_data(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported data shape: {path}")
    return data


def qid(row: dict[str, Any], fallback: str = "") -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {qid(row): row for row in rows}


def passages_from_context(context: Any) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return [str(p) for p in context if p]
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles = [title for title, _sentences in context]
        sentences = [sentences for _title, sentences in context]
    passages = []
    for title, sent_list in zip(titles, sentences):
        body = sent_list if isinstance(sent_list, str) else " ".join(str(s) for s in sent_list)
        if body:
            passages.append(f"{title}: {body}")
    return passages


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def parse_answer(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return str(obj.get("answer") or obj.get("final_answer") or "").strip()
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return str(obj.get("answer") or obj.get("final_answer") or "").strip()
        except Exception:
            pass
    for prefix in ("FINAL_ANSWER:", "Answer:", "answer:"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned.strip().strip('"')


def llm_answer(prompt: str, retries: int = 4) -> tuple[str, dict[str, int]]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=120,
            )
            return parse_answer(resp.choices[0].message.content or ""), usage_from_response(resp)
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2 ** attempt)
    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def build_prompt(question: str, candidates: dict[str, str], passages: list[str], prompt_version: str) -> str:
    candidate_lines = "\n".join(
        f"- {name}: {value}" for name, value in candidates.items() if str(value or "").strip()
    )
    passage_lines = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    if prompt_version == "support_first":
        return f"""You are a HotpotQA-style multi-hop question answering expert.

Use only the evidence passages below. Candidate answers are hints from other modules; they may be wrong.

Work internally as follows, but do not show your work:
1. Identify the relevant entities in the question.
2. Find the two supporting facts or the comparison facts in the passages.
3. Resolve the bridge/comparison.
4. Output the shortest exact answer expected by an EM/F1 evaluator.

Question:
{question}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Answer style rules:
- Return only the final answer, not a sentence.
- For yes/no questions, return only "yes" or "no".
- For countries, prefer the common country name used by QA datasets when the evidence has a formal name.
- For "what do X and Y have in common", return the shared type/category, not a full explanatory sentence.
- If evidence and candidates conflict, trust the passages.
- If the passages are insufficient, use the best concise candidate that is not a refusal; otherwise return "unknown".

Return strict JSON only:
{{"answer": "..."}}
"""

    return f"""You are an exact-answer reader for a multi-hop QA benchmark.

Use only the evidence passages below. Candidate answers are hints from other modules; ignore any candidate that is unsupported or contradicted by the passages.

Question:
{question}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Rules:
- Return the shortest answer that directly answers the question.
- For yes/no questions, return only "yes" or "no".
- For comparison questions, return the chosen entity or "yes"/"no" as appropriate.
- Prefer the surface form used in the evidence when it is equivalent to a candidate.
- Do not include citations, explanations, markdown, or prefixes.
- If the evidence is insufficient, return the best non-refusal candidate; if none is usable, return "unknown".

Return strict JSON only:
{{"answer": "..."}}
"""


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "evidence_adjudicated_reader",
        "n": len(rows),
        "EM": round(sum(float(row["em"]) for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(float(row["f1"]) for row in rows) / len(rows), 4) if rows else 0,
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-version", choices=("default", "support_first"), default="default")
    parser.add_argument(
        "--extra-candidate",
        action="append",
        default=[],
        help="Extra candidate JSONL as label:path. Row answer/prediction is used.",
    )
    args = parser.parse_args()

    data_by_id = {qid(row, str(i)): row for i, row in enumerate(load_data(args.data))}
    current_rows = load_jsonl(args.current)
    judge_by_id = rows_by_qid(load_jsonl(args.judge))
    prior_by_id = rows_by_qid(load_jsonl(args.prior))
    guarded_by_id = rows_by_qid(load_jsonl(args.guarded))
    extra_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.extra_candidate:
        if ":" not in spec:
            raise ValueError(f"Invalid --extra-candidate value: {spec}")
        label, path = spec.split(":", 1)
        extra_candidates[label] = rows_by_qid(load_jsonl(Path(path)))
    existing = rows_by_qid(load_jsonl(args.out_jsonl)) if args.out_jsonl.exists() else {}

    selected_rows = current_rows[args.start:]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    for index, current in enumerate(selected_rows, start=1):
        row_qid = qid(current)
        if row_qid in existing:
            continue

        item = data_by_id.get(row_qid, {})
        judge = judge_by_id.get(row_qid, {})
        prior = prior_by_id.get(row_qid, {})
        guarded = guarded_by_id.get(row_qid, {})
        question = str(current.get("question") or item.get("question") or "")
        gold = str(current.get("gold") or item.get("answer") or "")
        candidates = {
            "guarded": str(guarded.get("answer") or guarded.get("prediction") or ""),
            "current_v4": str(current.get("answer") or current.get("prediction") or ""),
            "kg": str(judge.get("kg") or ""),
            "context": str(judge.get("context") or ""),
            "judge_v2": str(judge.get("judge_answer") or judge.get("answer") or ""),
            "contextual_prior_v2": str(prior.get("answer") or prior.get("prediction") or ""),
        }
        for label, rows in extra_candidates.items():
            extra_row = rows.get(row_qid, {})
            candidates[label] = str(extra_row.get("answer") or extra_row.get("prediction") or "")
        passages = passages_from_context(item.get("context", []))
        prompt = build_prompt(question, candidates, passages, args.prompt_version)
        started = time.time()
        try:
            answer, usage = llm_answer(prompt)
            error = None
        except Exception as exc:
            answer = candidates["guarded"] or candidates["current_v4"]
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            error = str(exc)

        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "evidence_adjudicated_reader",
            "question": question,
            "answer": answer,
            "prediction": answer,
            "gold": gold,
            "em": em(answer, gold),
            "f1": round(f1(answer, gold), 4),
            "fallback_error": error,
            **{f"candidate_{name}": value for name, value in candidates.items()},
        }
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if args.usage_log:
            append_jsonl(args.usage_log, {
                "_id": row_qid,
                "mode": "evidence_adjudicated_reader",
                "llm_calls": 0 if error else 1,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "wall_time": round(time.time() - started, 4),
                "error": error,
            })

        if index % 25 == 0:
            print(f"[{index}/{len(selected_rows)}] evidence adjudicated", flush=True)

    final_rows = [existing[qid(row)] for row in selected_rows if qid(row) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
