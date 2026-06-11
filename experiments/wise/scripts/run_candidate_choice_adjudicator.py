#!/usr/bin/env python3
"""Candidate-choice adjudicator over existing CoMaGRAG HotpotQA outputs."""

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
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_data(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported data shape: {path}")


def qid(row: dict[str, Any], fallback: str = "") -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {qid(row): row for row in rows}


def answer(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


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


def looks_bad(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    return not lowered or any(marker in lowered for marker in BAD_MARKERS)


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


def llm_json(prompt: str, retries: int = 4) -> tuple[dict[str, Any], dict[str, int], str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=180,
            )
            raw = resp.choices[0].message.content or ""
            return parse_json_obj(raw), usage_from_response(resp), raw
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2**attempt)
    return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, ""


def unique_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: dict[str, tuple[str, str]] = {}
    for label, value in candidates:
        value = str(value or "").strip()
        if not value:
            continue
        key = normalize(value)
        if not key:
            continue
        if key in seen:
            old_label, old_value = seen[key]
            seen[key] = (f"{old_label}|{label}", old_value)
        else:
            seen[key] = (label, value)
    return list(seen.values())


def build_prompt(question: str, candidates: list[tuple[str, str]], passages: list[str]) -> str:
    candidate_lines = "\n".join(
        f"{i}. [{label}] {value}" for i, (label, value) in enumerate(candidates, start=1)
    )
    passage_lines = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"""You are selecting the final answer for a HotpotQA-style multi-hop QA benchmark.

Use only the evidence passages below. Candidate answers are hints from other modules; any of them may be wrong.

Question:
{question}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Selection rules:
- Prefer an exact candidate answer when it is fully supported and directly answers the question.
- You may make a minimal surface edit to a candidate only to remove extra explanation, add a clearly stated canonical word, or use the concise name/date/number from the passages.
- Return the shortest answer that would be accepted by an exact-match QA evaluator, not a full sentence.
- For yes/no questions, return only "yes" or "no".
- For comparison questions, return the selected entity or yes/no answer, not the reasoning.
- For "both/same/common" questions, answer the shared category, location, country, occupation, or property asked for.
- If the passages are insufficient, keep the strongest non-refusal candidate.

Return strict JSON only:
{{"answer": "...", "source": "candidate label or minimal_edit", "confidence": "high|medium|low"}}
"""


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "candidate_choice_adjudicator",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_baseline": sum(row["selected_source"] == "baseline" for row in rows),
        "selected_adjudicator": sum(row["selected_source"] == "adjudicator" for row in rows),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--reader-v1", type=Path, required=True)
    parser.add_argument("--reader-v1-fallback", type=Path, required=True)
    parser.add_argument("--reader-support", type=Path, required=True)
    parser.add_argument("--final-verifier", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-disagreements", action="store_true")
    args = parser.parse_args()

    data_rows = load_data(args.data)
    data_by_id = {qid(row, str(i)): row for i, row in enumerate(data_rows)}
    baseline_rows = load_jsonl(args.baseline)
    baseline_by_id = rows_by_qid(baseline_rows)
    full_by_id = rows_by_qid(load_jsonl(args.full))
    guarded_by_id = rows_by_qid(load_jsonl(args.guarded))
    judge_by_id = rows_by_qid(load_jsonl(args.judge))
    prior_by_id = rows_by_qid(load_jsonl(args.prior))
    reader_v1_by_id = rows_by_qid(load_jsonl(args.reader_v1))
    reader_v1_fallback_by_id = rows_by_qid(load_jsonl(args.reader_v1_fallback))
    reader_support_by_id = rows_by_qid(load_jsonl(args.reader_support))
    final_verifier_by_id = rows_by_qid(load_jsonl(args.final_verifier))
    existing = rows_by_qid(load_jsonl(args.out_jsonl))

    selected_rows = baseline_rows[args.start:]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    for index, baseline_row in enumerate(selected_rows, start=1):
        row_qid = qid(baseline_row)
        if row_qid in existing:
            continue
        item = data_by_id.get(row_qid, {})
        gold = str(baseline_row.get("gold") or item.get("answer") or "")
        baseline_answer = answer(baseline_row, "answer", "prediction")
        judge_row = judge_by_id.get(row_qid, {})
        candidates = unique_candidates([
            ("support_fallback", baseline_answer),
            ("reader_support", answer(reader_support_by_id.get(row_qid, {}), "answer", "prediction")),
            ("reader_v1", answer(reader_v1_by_id.get(row_qid, {}), "answer", "prediction")),
            ("reader_v1_fallback", answer(reader_v1_fallback_by_id.get(row_qid, {}), "answer", "prediction")),
            ("final_verifier", answer(final_verifier_by_id.get(row_qid, {}), "answer", "prediction")),
            ("guarded", answer(guarded_by_id.get(row_qid, {}), "answer", "prediction")),
            ("full", answer(full_by_id.get(row_qid, {}), "answer", "prediction")),
            ("judge_v2", answer(judge_row, "judge_answer", "answer")),
            ("kg", answer(judge_row, "kg")),
            ("context", answer(judge_row, "context")),
            ("contextual_prior_v2", answer(prior_by_id.get(row_qid, {}), "answer", "prediction")),
        ])

        non_bad = [normalize(value) for _label, value in candidates if not looks_bad(value)]
        has_disagreement = len(set(non_bad)) > 1
        if args.only_disagreements and not has_disagreement:
            selected = baseline_answer
            raw = ""
            obj: dict[str, Any] = {}
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            selected_source = "baseline"
            error = None
        else:
            prompt = build_prompt(
                question=str(baseline_row.get("question") or item.get("question") or ""),
                candidates=candidates,
                passages=passages_from_context(item.get("context", [])),
            )
            started = time.time()
            try:
                obj, usage, raw = llm_json(prompt)
                selected = str(obj.get("answer") or "").strip() or baseline_answer
                selected_source = "adjudicator"
                error = None
            except Exception as exc:
                selected = baseline_answer
                obj = {}
                raw = ""
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                selected_source = "baseline"
                error = str(exc)
            if args.usage_log:
                append_jsonl(args.usage_log, {
                    "_id": row_qid,
                    "mode": "candidate_choice_adjudicator",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                })

        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "candidate_choice_adjudicator",
            "question": baseline_row.get("question") or item.get("question", ""),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_source": selected_source,
            "adjudicator_source": obj.get("source", ""),
            "adjudicator_confidence": obj.get("confidence", ""),
            "baseline_answer": baseline_answer,
            "raw_response": raw,
            "error": error,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        for label, value in candidates:
            out_row[f"candidate_{label}"] = value
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if index % 25 == 0:
            print(f"[{index}/{len(selected_rows)}] candidate choice adjudicated", flush=True)

    final_rows = [existing[qid(row)] for row in selected_rows if qid(row) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
