#!/usr/bin/env python3
"""Strict quote-gated passage reader for comparison/common HotpotQA cases.

This experiment does not rebuild the graph. It asks for exact supporting
quotes from the original passages and accepts a switch only when the quotes are
verbatim substrings of the evidence and the proposed answer is short and
grounded.
"""

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
    "not enough",
    "provided context",
    "context passage",
    "does not mention",
    "do not mention",
    "cannot determine",
    "none of",
    "no triple",
    "not specified",
    "instructions",
    "using only",
)


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pred_tokens = tokens(pred)
    gold_tokens = tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qid(row: dict[str, Any], fallback: int | str = "") -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rows_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    return {qid(row, i): row for i, row in enumerate(load_jsonl(path))}


def answer(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, "") and str(value).strip().lower() not in {"null", "nan"}:
            return str(value).strip()
    return ""


def question_type(question: str) -> str:
    lowered = question.lower().strip()
    if " both " in f" {lowered} " or " in common" in lowered or "same " in lowered:
        return "common"
    if any(
        marker in lowered
        for marker in (
            "born first",
            "older",
            "newer",
            "released first",
            "came out first",
            "which was developed first",
            "opened first",
            "more ",
            "larger",
            "bigger",
            "closer",
            "nearer",
            "higher population",
        )
    ):
        return "comparison"
    return "other"


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower()
    return (
        not normalize(value)
        or len(tokens(value)) > 10
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def passages_from_context(context: Any) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return [str(p) for p in context if p]
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    elif isinstance(context, list):
        titles = [title for title, _sentences in context]
        sentences = [sentences for _title, sentences in context]
    else:
        return []
    passages = []
    for title, sent_list in zip(titles, sentences):
        body = sent_list if isinstance(sent_list, str) else " ".join(str(s) for s in sent_list)
        if body:
            passages.append(f"{title}: {body}")
    return passages


def compact_passages(passages: list[str], max_chars: int = 7000) -> list[str]:
    kept = []
    used = 0
    for passage in passages:
        passage = re.sub(r"\s+", " ", passage).strip()
        if not passage:
            continue
        room = max_chars - used
        if room <= 0:
            break
        if len(passage) > room:
            passage = passage[:room].rsplit(" ", 1)[0]
        kept.append(passage)
        used += len(passage)
    return kept


def target_rows(current_rows: list[dict[str, Any]], limit: int | None, target_qtype: str | None) -> set[str]:
    ranked = []
    for i, row in enumerate(current_rows):
        qtype = question_type(str(row.get("question") or ""))
        if target_qtype and qtype != target_qtype:
            continue
        if not target_qtype and qtype not in {"common", "comparison"}:
            continue
        current = answer(row, "answer", "prediction")
        priority = 0 if qtype == "common" else 1
        if bad_surface(current):
            priority -= 2
        if str(row.get("previous_rule") or "") in {
            "baseline",
            "source_calibrated_reader_consensus",
            "candidate_choice_prior_not_released_first",
        }:
            priority -= 1
        ranked.append((priority, i, qid(row, i)))
    ranked.sort()
    if limit is not None:
        ranked = ranked[:limit]
    return {row_qid for _priority, _i, row_qid in ranked}


def build_prompt(question: str, qtype: str, current: str, passages: list[str]) -> str:
    passage_lines = "\n".join(f"[{i + 1}] {passage}" for i, passage in enumerate(passages))
    if qtype == "comparison":
        task_rules = """- Identify the two compared entities from the question.
- Quote evidence for the relevant attribute of each entity.
- Return one of the compared entities, or yes/no only if the question asks yes/no."""
    else:
        task_rules = """- Identify both entities from the question.
- Quote evidence showing the shared category, property, location, occupation, or type.
- Return only that shared answer, not a full sentence."""
    return f"""You are a strict HotpotQA evidence reader.

Use only the evidence passages. Do not use outside knowledge. The current answer may be correct; switch only when exact quotes prove a different answer.

Question type: {qtype}
Question:
{question}

Current answer:
{current}

Evidence passages:
{passage_lines}

Task-specific rules:
{task_rules}

Output rules:
- quote_1 and quote_2 must be exact copied substrings from the Evidence passages.
- If you cannot provide exact quotes, set should_switch to false and keep current.
- The answer must be concise: entity, yes/no, date, number, location, or category.
- Do not put reasoning in the answer.

Return strict JSON only:
{{"answer": "...", "confidence": "high|medium|low", "should_switch": true, "quote_1": "...", "quote_2": "...", "reason": "short"}}
"""


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


def llm_json(prompt: str, model: str, retries: int = 3) -> tuple[dict[str, Any], dict[str, int], str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=220,
            )
            raw = resp.choices[0].message.content or ""
            return parse_json_obj(raw), usage_from_response(resp), raw
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2**attempt)
    return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, ""


def quote_in_passages(quote: str, passages: list[str]) -> bool:
    quote_compact = compact(quote)
    if len(quote_compact) < 16:
        return False
    evidence = compact(" ".join(passages))
    return quote_compact in evidence


def answer_grounded_in_quotes(answer_text: str, quotes: list[str], question: str, qtype: str) -> bool:
    answer_norm = normalize(answer_text)
    if not answer_norm:
        return False
    quote_norm = normalize(" ".join(quotes))
    if answer_norm in quote_norm:
        return True
    if qtype == "comparison" and answer_norm in normalize(question):
        return True
    answer_tokens = tokens(answer_text)
    if len(answer_tokens) <= 2:
        return all(token in quote_norm for token in answer_tokens)
    overlap = sum(1 for token in answer_tokens if token in quote_norm)
    return overlap / len(answer_tokens) >= 0.75


def should_accept(
    current: str,
    proposed: str,
    obj: dict[str, Any],
    question: str,
    qtype: str,
    passages: list[str],
) -> tuple[bool, str]:
    proposed = str(proposed or "").strip()
    if normalize(proposed) == normalize(current):
        return False, "same_as_current"
    if str(obj.get("confidence") or "").lower() != "high":
        return False, "not_high_confidence"
    if not bool(obj.get("should_switch", False)):
        return False, "should_switch_false"
    if bad_surface(proposed):
        return False, "bad_surface"
    quotes = [str(obj.get("quote_1") or "").strip(), str(obj.get("quote_2") or "").strip()]
    if not all(quote_in_passages(quote, passages) for quote in quotes):
        return False, "quote_not_verbatim"
    if not answer_grounded_in_quotes(proposed, quotes, question, qtype):
        return False, "answer_not_quote_grounded"
    if qtype == "comparison" and normalize(proposed) not in normalize(question) and normalize(proposed) not in {"yes", "no"}:
        return False, "comparison_not_option"
    return True, "accepted"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "strict_quote_reader",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "targets": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row.get("selected_rule") == "strict_quote_switch" for row in rows),
    }
    summary.update({f"acceptance_{key}": value for key, value in Counter(row.get("acceptance", "") for row in rows).items()})
    summary.update({f"target_{key}": value for key, value in Counter(row.get("question_type", "") for row in rows if row.get("llm_called")).items()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--target-qtype", choices=("common", "comparison"), default=None)
    args = parser.parse_args()

    data_by_id = rows_by_qid(args.data)
    current_rows = load_jsonl(args.current)
    targets = target_rows(current_rows, args.limit, args.target_qtype)
    existing = rows_by_qid(args.out_jsonl) if args.out_jsonl.exists() else {}

    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        if row_qid in existing:
            continue
        item = data_by_id.get(row_qid, {})
        question = str(row.get("question") or item.get("question") or "")
        qtype = question_type(question)
        current = answer(row, "answer", "prediction")
        gold = answer(row, "gold") or answer(item, "answer")
        passages = compact_passages(passages_from_context(item.get("context", [])))
        should_call = row_qid in targets
        selected = current
        selected_rule = "current_v3"
        obj: dict[str, Any] = {}
        raw = ""
        error = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        acceptance = "not_target"
        started = time.time()
        if should_call:
            try:
                obj, usage, raw = llm_json(build_prompt(question, qtype, current, passages), args.model)
                proposed = str(obj.get("answer") or "").strip()
                ok, acceptance = should_accept(current, proposed, obj, question, qtype, passages)
                if ok:
                    selected = proposed
                    selected_rule = "strict_quote_switch"
            except Exception as exc:
                error = str(exc)
                acceptance = "error"

        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "strict_quote_reader",
            "question": question,
            "question_type": qtype,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_rule": selected_rule,
            "previous_rule": row.get("previous_rule", ""),
            "current_answer": current,
            "llm_called": int(should_call),
            "reader_answer": obj.get("answer", ""),
            "reader_confidence": obj.get("confidence", ""),
            "reader_should_switch": obj.get("should_switch", ""),
            "reader_quote_1": obj.get("quote_1", ""),
            "reader_quote_2": obj.get("quote_2", ""),
            "reader_reason": obj.get("reason", ""),
            "acceptance": acceptance,
            "raw_response": raw,
            "error": error,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row
        if args.usage_log and should_call:
            append_jsonl(
                args.usage_log,
                {
                    "_id": row_qid,
                    "mode": "strict_quote_reader",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                },
            )
        if should_call and len([r for r in existing.values() if r.get("llm_called")]) % 20 == 0:
            print(f"strict quote reader {len([r for r in existing.values() if r.get('llm_called')])}/{len(targets)}", flush=True)

    final_rows = [existing[qid(row, i)] for i, row in enumerate(current_rows) if qid(row, i) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
