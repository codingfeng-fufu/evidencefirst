#!/usr/bin/env python3
"""Source-calibrated selector over strong CoMaGRAG candidate families."""

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


def looks_bad(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    return not lowered or any(marker in lowered for marker in BAD_MARKERS)


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
                max_tokens=160,
            )
            raw = resp.choices[0].message.content or ""
            return parse_json_obj(raw), usage_from_response(resp), raw
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2**attempt)
    return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, ""


def consensus(values: list[str]) -> str:
    counts = Counter(normalize(value) for value in values if value and not looks_bad(value))
    if not counts:
        return ""
    key, count = counts.most_common(1)[0]
    if count < 2:
        return ""
    for value in values:
        if normalize(value) == key:
            return value
    return ""


def build_alternatives(
    qid_value: str,
    baseline: str,
    full: dict[str, dict[str, Any]],
    guarded: dict[str, dict[str, Any]],
    judge: dict[str, dict[str, Any]],
    prior: dict[str, dict[str, Any]],
    reader_v1: dict[str, dict[str, Any]],
    reader_support: dict[str, dict[str, Any]],
    final_verifier: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    judge_row = judge.get(qid_value, {})
    graph = consensus([
        answer(full.get(qid_value, {}), "answer", "prediction"),
        answer(guarded.get(qid_value, {}), "answer", "prediction"),
        answer(judge_row, "judge_answer", "answer"),
    ])
    reader = consensus([
        answer(reader_v1.get(qid_value, {}), "answer", "prediction"),
        answer(reader_support.get(qid_value, {}), "answer", "prediction"),
        answer(final_verifier.get(qid_value, {}), "answer", "prediction"),
    ])
    prior_answer = answer(prior.get(qid_value, {}), "answer", "prediction")
    values = [
        ("baseline", baseline),
        ("contextual_prior", prior_answer),
        ("graph_consensus", graph),
        ("reader_consensus", reader),
    ]
    out = []
    seen = set()
    for label, value in values:
        value = str(value or "").strip()
        key = normalize(value)
        if not key or key in seen or looks_bad(value):
            continue
        seen.add(key)
        out.append((label, value))
    return out


def build_prompt(question: str, alternatives: list[tuple[str, str]], passages: list[str]) -> str:
    alt_lines = "\n".join(f"- {label}: {value}" for label, value in alternatives)
    passage_lines = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"""You are a source-calibrated final-answer selector for HotpotQA.

Use only the evidence passages to decide the answer. The candidate labels describe how each answer was produced:
- baseline: strongest current CoMaGRAG answer, usually good but can over-answer or follow the wrong entity.
- contextual_prior: question-level canonical answer, useful for common names, broad categories, countries, cities, and option selection, but can be wrong on temporal/comparison facts.
- graph_consensus: strict graph answer, useful when multiple graph modules agree, especially yes/no and explicit comparison facts, but can miss context.
- reader_consensus: evidence-reader answer agreed by reader variants, useful when baseline fell back to a prior.

Question:
{question}

Candidates:
{alt_lines}

Evidence passages:
{passage_lines}

Rules:
- Choose the answer that directly answers the question, not merely a related entity.
- Keep the exact surface form concise: entity, date, number, location, country, category, occupation, or yes/no.
- For yes/no questions, return only "yes" or "no".
- For "both/same/common" questions, choose the shared property at the granularity requested by the question.
- For temporal or numeric comparisons, trust explicit dates/numbers in the passages over source labels.
- If the evidence does not decide between candidates, keep baseline.

Return strict JSON only:
{{"choice": "baseline|contextual_prior|graph_consensus|reader_consensus", "answer": "...", "confidence": "high|medium|low"}}
"""


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "source_calibrated_selector",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_baseline": sum(row["selected_choice"] == "baseline" for row in rows),
        "selected_contextual_prior": sum(row["selected_choice"] == "contextual_prior" for row in rows),
        "selected_graph_consensus": sum(row["selected_choice"] == "graph_consensus" for row in rows),
        "selected_reader_consensus": sum(row["selected_choice"] == "reader_consensus" for row in rows),
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
    parser.add_argument("--reader-support", type=Path, required=True)
    parser.add_argument("--final-verifier", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_rows = load_data(args.data)
    data_by_id = {qid(row, str(i)): row for i, row in enumerate(data_rows)}
    baseline_rows = load_jsonl(args.baseline)
    full = rows_by_qid(load_jsonl(args.full))
    guarded = rows_by_qid(load_jsonl(args.guarded))
    judge = rows_by_qid(load_jsonl(args.judge))
    prior = rows_by_qid(load_jsonl(args.prior))
    reader_v1 = rows_by_qid(load_jsonl(args.reader_v1))
    reader_support = rows_by_qid(load_jsonl(args.reader_support))
    final_verifier = rows_by_qid(load_jsonl(args.final_verifier))
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
        alternatives = build_alternatives(
            row_qid,
            baseline_answer,
            full=full,
            guarded=guarded,
            judge=judge,
            prior=prior,
            reader_v1=reader_v1,
            reader_support=reader_support,
            final_verifier=final_verifier,
        )
        if len(alternatives) <= 1:
            selected = baseline_answer
            choice = "baseline"
            confidence = ""
            raw = ""
            error = None
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        else:
            prompt = build_prompt(
                question=str(baseline_row.get("question") or item.get("question") or ""),
                alternatives=alternatives,
                passages=passages_from_context(item.get("context", [])),
            )
            started = time.time()
            try:
                obj, usage, raw = llm_json(prompt)
                choice = str(obj.get("choice") or "baseline").strip()
                by_label = {label: value for label, value in alternatives}
                selected = str(obj.get("answer") or "").strip() or by_label.get(choice, baseline_answer)
                if choice not in by_label:
                    choice = "baseline"
                    selected = baseline_answer
                confidence = str(obj.get("confidence") or "")
                error = None
            except Exception as exc:
                selected = baseline_answer
                choice = "baseline"
                confidence = ""
                raw = ""
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                error = str(exc)
            if args.usage_log:
                append_jsonl(args.usage_log, {
                    "_id": row_qid,
                    "mode": "source_calibrated_selector",
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
            "mode": "source_calibrated_selector",
            "question": baseline_row.get("question") or item.get("question", ""),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_choice": choice,
            "confidence": confidence,
            "baseline_answer": baseline_answer,
            "raw_response": raw,
            "error": error,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        for label, value in alternatives:
            out_row[f"candidate_{label}"] = value
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if index % 25 == 0:
            print(f"[{index}/{len(selected_rows)}] source-calibrated selected", flush=True)

    final_rows = [existing[qid(row)] for row in selected_rows if qid(row) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
