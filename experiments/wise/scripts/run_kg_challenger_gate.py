#!/usr/bin/env python3
"""KG challenger with passage verification.

KG-derived answers are used only as challengers. A switch is accepted only when
the model selects a KG challenger with high confidence and supplies verbatim
passage quotes that support it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
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


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower()
    return (
        not normalize(value)
        or len(tokens(value)) > 12
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def question_type(question: str) -> str:
    lowered = question.lower().strip()
    if lowered.startswith(("are ", "were ", "do ", "did ", "does ", "is ", "was ", "has ", "have ", "had ", "can ")):
        return "yesno"
    if any(term in lowered for term in ("what year", "how many", "when was", "when did", "period", "date", "population", "number")):
        return "number_date"
    if lowered.startswith("where") or any(term in lowered for term in ("what city", "what country", "what region", "located", "headquartered")):
        return "location"
    if lowered.startswith("who") or " who " in lowered[:140]:
        return "person"
    if " both " in f" {lowered} " or " in common" in lowered or "same " in lowered:
        return "common"
    if any(term in lowered for term in ("born first", "older", "newer", "released first", "came out first", "closer", "nearer", "more ", "larger", "bigger")):
        return "comparison"
    return "other"


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


def kg_tables() -> list[tuple[str, dict[str, dict[str, Any]]]]:
    paths = [
        "results/wise/hotpot_old500_comagraag_candidate_choice_predictions.jsonl",
        "results/wise/hotpot_extra500_comagraag_candidate_choice_predictions.jsonl",
        "results/wise/hotpot_old500_comagraag_candidate_judge_v2_all_predictions.jsonl",
        "results/wise/hotpot_extra500_comagraag_candidate_judge_v2_all_predictions.jsonl",
        "results/wise/hotpot_old500_comagraag_evidence_reader_predictions.jsonl",
        "results/wise/hotpot_extra500_comagraag_evidence_reader_predictions.jsonl",
        "results/wise/hotpot_old500_comagraag_evidence_reader_fallback_predictions.jsonl",
        "results/wise/hotpot_extra500_comagraag_evidence_reader_fallback_predictions.jsonl",
        "results/wise/hotpot_old500_comagraag_final_verifier_predictions.jsonl",
        "results/wise/hotpot_extra500_comagraag_final_verifier_predictions.jsonl",
    ]
    return [(Path(path).stem.replace("_predictions", ""), rows_by_qid(ROOT / path)) for path in paths]


def kg_candidates(row_qid: str, tables: list[tuple[str, dict[str, dict[str, Any]]]]) -> list[tuple[str, str]]:
    candidates = []
    for source, table in tables:
        row = table.get(row_qid, {})
        for field, value in row.items():
            if not isinstance(value, str):
                continue
            if field == "kg" or "kg" in field.lower():
                value = value.strip()
                if value and not bad_surface(value):
                    candidates.append((f"{source}:{field}", value))
    seen: dict[str, tuple[str, str]] = {}
    for label, value in candidates:
        key = normalize(value)
        if key in seen:
            old_label, old_value = seen[key]
            seen[key] = (f"{old_label}|{label}", old_value)
        else:
            seen[key] = (label, value)
    return list(seen.values())[:8]


def rank_targets(current_rows: list[dict[str, Any]], tables: list[tuple[str, dict[str, dict[str, Any]]]], limit: int | None) -> set[str]:
    priority_by_type = {
        "yesno": 0,
        "comparison": 1,
        "number_date": 2,
        "person": 3,
        "common": 4,
        "location": 5,
        "other": 6,
    }
    ranked = []
    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        current = answer(row, "answer", "prediction")
        cands = [(label, value) for label, value in kg_candidates(row_qid, tables) if normalize(value) != normalize(current)]
        if not cands:
            continue
        qtype = question_type(str(row.get("question") or ""))
        priority = priority_by_type[qtype]
        if bad_surface(current):
            priority -= 2
        if any(normalize(value) in {"yes", "no"} for _label, value in cands):
            priority -= 1
        ranked.append((priority, i, row_qid))
    ranked.sort()
    if limit is not None:
        ranked = ranked[:limit]
    return {row_qid for _priority, _i, row_qid in ranked}


def build_prompt(question: str, qtype: str, current: str, candidates: list[tuple[str, str]], passages: list[str]) -> str:
    candidate_lines = "\n".join(f"{i}. [{label}] {value}" for i, (label, value) in enumerate(candidates, start=1))
    passage_lines = "\n".join(f"[{i + 1}] {passage}" for i, passage in enumerate(passages))
    return f"""You are verifying whether a knowledge-graph challenger should replace the current HotpotQA answer.

Use only the evidence passages. The KG challengers may be wrong because graph extraction can miss context. Keep current unless a KG challenger is directly supported.

Question type: {qtype}
Question:
{question}

Current answer:
{current}

KG challenger answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Rules:
- Switch only to one of the KG challenger answers.
- quote_1 and quote_2 must be exact copied substrings from the evidence passages.
- For yes/no questions, the answer must be exactly "yes" or "no".
- For comparison questions, quote evidence for both compared entities.
- If the evidence is ambiguous, or if current is already supported, keep current.

Return strict JSON only:
{{"action": "keep|switch", "answer": "...", "confidence": "high|medium|low", "kg_source": "candidate label", "quote_1": "...", "quote_2": "...", "reason": "short"}}
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
    return quote_compact in compact(" ".join(passages))


def matches_kg_candidate(proposed: str, candidates: list[tuple[str, str]]) -> bool:
    return any(normalize(proposed) == normalize(value) for _label, value in candidates)


def reason_keeps_current(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in ("current answer is correct", "current is correct", "keep current", "current already"))


def should_accept(current: str, proposed: str, obj: dict[str, Any], candidates: list[tuple[str, str]], passages: list[str]) -> tuple[bool, str]:
    proposed = str(proposed or "").strip()
    if normalize(proposed) == normalize(current):
        return False, "same_as_current"
    if str(obj.get("action") or "").lower() != "switch":
        return False, "action_not_switch"
    if str(obj.get("confidence") or "").lower() != "high":
        return False, "not_high_confidence"
    if reason_keeps_current(str(obj.get("reason") or "")):
        return False, "reason_keeps_current"
    if bad_surface(proposed):
        return False, "bad_surface"
    if not matches_kg_candidate(proposed, candidates):
        return False, "not_kg_candidate"
    quotes = [str(obj.get("quote_1") or ""), str(obj.get("quote_2") or "")]
    if not all(quote_in_passages(quote, passages) for quote in quotes):
        return False, "quote_not_verbatim"
    return True, "accepted"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "kg_challenger_gate",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4),
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4),
        "targets": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row["selected_rule"] == "kg_challenger_switch" for row in rows),
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
    args = parser.parse_args()

    data_by_id = rows_by_qid(args.data)
    current_rows = load_jsonl(args.current)
    tables = kg_tables()
    targets = rank_targets(current_rows, tables, args.limit)
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
        cands = [(label, value) for label, value in kg_candidates(row_qid, tables) if normalize(value) != normalize(current)]
        should_call = row_qid in targets and bool(cands)
        selected = current
        selected_rule = "current"
        obj: dict[str, Any] = {}
        raw = ""
        error = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        acceptance = "not_target"
        started = time.time()
        if should_call:
            try:
                obj, usage, raw = llm_json(build_prompt(question, qtype, current, cands, passages), args.model)
                proposed = str(obj.get("answer") or "").strip()
                ok, acceptance = should_accept(current, proposed, obj, cands, passages)
                if ok:
                    selected = proposed
                    selected_rule = "kg_challenger_switch"
            except Exception as exc:
                error = str(exc)
                acceptance = "error"

        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "kg_challenger_gate",
            "question": question,
            "question_type": qtype,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_rule": selected_rule,
            "current_answer": current,
            "llm_called": int(should_call),
            "kg_candidates": " | ".join(f"{label}={value}" for label, value in cands),
            "kg_answer": obj.get("answer", ""),
            "kg_source": obj.get("kg_source", ""),
            "kg_confidence": obj.get("confidence", ""),
            "kg_action": obj.get("action", ""),
            "kg_quote_1": obj.get("quote_1", ""),
            "kg_quote_2": obj.get("quote_2", ""),
            "kg_reason": obj.get("reason", ""),
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
                    "mode": "kg_challenger_gate",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                },
            )
        if should_call and len([r for r in existing.values() if r.get("llm_called")]) % 20 == 0:
            print(f"kg challenger {len([r for r in existing.values() if r.get('llm_called')])}/{len(targets)}", flush=True)

    final_rows = [existing[qid(row, i)] for i, row in enumerate(current_rows) if qid(row, i) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
