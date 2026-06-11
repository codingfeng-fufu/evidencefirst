#!/usr/bin/env python3
"""Conservative evidence adjudicator for targeted CoMaGRAG body-v3 cases.

This script is intentionally post-hoc and non-gold: gold answers are used only
for metric reporting. The adjudicator is called only on deterministic high-risk
cases, then a local acceptance gate keeps the current answer unless the LLM
returns a short, high-confidence answer that is one of the presented candidates
or a small evidence-backed surface edit.
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

LOW_RULES = {
    "candidate_choice_prior_not_released_first",
    "source_calibrated_reader_consensus",
    "source_calibrated_contextual_prior_safe",
    "candidate_choice_medium_guarded_or_judge",
}

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
    "instructions require",
    "using *only*",
    "given triples",
)


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


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


def load_data(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {qid(row, i): row for i, row in enumerate(rows)}


def answer(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, "") and str(value).strip().lower() not in {"null", "none", "nan"}:
            return str(value).strip()
    return ""


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower()
    return (
        not normalize(value)
        or len(tokens(value)) > 14
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def is_yes_no_question(question: str) -> bool:
    return question.strip().lower().startswith(
        ("are ", "were ", "do ", "did ", "does ", "is ", "was ", "has ", "have ", "had ", "can ")
    )


def is_comparison_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in (
            "which ",
            "who is older",
            "born first",
            "released first",
            "came out first",
            "newer",
            "older",
            "closer",
            "more ",
            "larger",
            "bigger",
        )
    )


def target_reasons(row: dict[str, Any]) -> list[str]:
    question = str(row.get("question") or "")
    current = answer(row, "answer", "prediction")
    reasons = []
    if str(row.get("previous_rule") or "") in LOW_RULES:
        reasons.append("low_rule")
    if bad_surface(current):
        reasons.append("bad_current")
    if is_yes_no_question(question) and normalize(current) not in {"yes", "no"}:
        reasons.append("yesno_non_yesno")
    if is_comparison_question(question) and len(tokens(current)) > 8:
        reasons.append("comparison_long")
    return reasons


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


def compact_passages(passages: list[str], max_chars: int = 5500) -> list[str]:
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


def add_candidate(candidates: list[tuple[str, str]], label: str, value: str) -> None:
    value = str(value or "").strip()
    if not value or bad_surface(value):
        return
    candidates.append((label, value))


def unique_candidates(candidates: list[tuple[str, str]], max_n: int = 14) -> list[tuple[str, str]]:
    seen: dict[str, tuple[str, str]] = {}
    for label, value in candidates:
        key = normalize(value)
        if not key:
            continue
        if key in seen:
            old_label, old_value = seen[key]
            seen[key] = (f"{old_label}|{label}", old_value)
        else:
            seen[key] = (label, value)
    return list(seen.values())[:max_n]


def collect_candidates(row_qid: str, current: str, tables: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    add_candidate(candidates, "current", current)
    for side in ("old", "extra"):
        choice = tables[f"choice_{side}"].get(row_qid, {})
        add_candidate(candidates, f"choice_{side}", answer(choice, "answer", "prediction"))
        add_candidate(candidates, f"choice_prior_{side}", answer(choice, "candidate_contextual_prior_v2"))
        add_candidate(candidates, f"choice_kg_{side}", answer(choice, "candidate_kg", "candidate_full|kg"))
        for field, value in choice.items():
            if field.startswith("candidate_") and any(key in field for key in ("context", "judge", "guarded", "kg")):
                add_candidate(candidates, f"{field}_{side}", str(value))

        source = tables[f"source_{side}"].get(row_qid, {})
        add_candidate(candidates, f"source_{side}", answer(source, "answer", "prediction"))
        add_candidate(candidates, f"source_prior_{side}", answer(source, "candidate_contextual_prior"))

        for prefix in ("reader", "readerfb", "support", "supportfb", "final", "guarded", "judge", "prior", "meta"):
            row = tables[f"{prefix}_{side}"].get(row_qid, {})
            for field in (
                "answer",
                "prediction",
                "context_answer",
                "candidate_context",
                "candidate_contextual_prior_v2",
                "contextual_prior_v2",
                "guarded_answer",
                "judge_answer",
                "current_v4",
                "kg",
                "context",
                "baseline_answer",
                "candidate_choice_answer",
                "source_calibrated_answer",
            ):
                add_candidate(candidates, f"{prefix}_{field}_{side}", answer(row, field))
    return unique_candidates(candidates)


def build_prompt(question: str, current: str, candidates: list[tuple[str, str]], passages: list[str]) -> str:
    candidate_lines = "\n".join(f"{i}. [{label}] {value}" for i, (label, value) in enumerate(candidates, start=1))
    passage_lines = "\n".join(f"[{i + 1}] {passage}" for i, passage in enumerate(passages))
    return f"""You are a conservative final-answer adjudicator for a HotpotQA-style QA benchmark.

Use only the evidence passages. The current answer is usually correct; change it only when the evidence clearly supports another candidate.

Question:
{question}

Current answer:
{current}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Rules:
- Default to keeping the current answer unless another answer is clearly supported by the passages.
- Choose only a concise final answer: entity, date, number, location, category, occupation, or yes/no.
- For yes/no questions, return only "yes" or "no".
- For comparison questions, return the selected entity or yes/no answer, not reasoning.
- Do not use outside knowledge. Do not output a full sentence.
- If evidence is ambiguous, keep current.

Return strict JSON only:
{{"action": "keep|switch", "answer": "...", "confidence": "high|medium|low", "source": "candidate label or current", "why": "short evidence reason"}}
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
                max_tokens=180,
            )
            raw = resp.choices[0].message.content or ""
            return parse_json_obj(raw), usage_from_response(resp), raw
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2**attempt)
    return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, ""


def source_matches(answer_text: str, source: str, candidates: list[tuple[str, str]]) -> bool:
    answer_norm = normalize(answer_text)
    source = str(source or "").lower()
    if not answer_norm:
        return False
    for label, value in candidates:
        if answer_norm == normalize(value) and (not source or label.lower() in source or source in label.lower()):
            return True
    return any(answer_norm == normalize(value) for _label, value in candidates)


def acceptable_switch(current: str, proposed: str, obj: dict[str, Any], candidates: list[tuple[str, str]]) -> tuple[bool, str]:
    proposed = str(proposed or "").strip()
    if normalize(proposed) == normalize(current):
        return False, "same_as_current"
    if str(obj.get("action") or "").lower() != "switch":
        return False, "action_not_switch"
    if str(obj.get("confidence") or "").lower() != "high":
        return False, "not_high_confidence"
    if bad_surface(proposed):
        return False, "bad_surface"
    if not source_matches(proposed, str(obj.get("source") or ""), candidates):
        return False, "not_candidate"
    return True, "accepted"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "conservative_evidence_adjudicator",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "targets": sum(bool(row.get("target_reasons")) for row in rows),
        "llm_calls": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row.get("selected_rule") == "adjudicator_switch" for row in rows),
        "accepted_switches": sum(row.get("acceptance") == "accepted" for row in rows),
    }
    summary.update({f"acceptance_{key}": value for key, value in Counter(row.get("acceptance", "") for row in rows).items()})
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=config.LLM_MODEL)
    args = parser.parse_args()

    current_rows = load_jsonl(args.current)
    data = load_data(args.data)
    tables = {
        "choice_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_candidate_choice_predictions.jsonl"),
        "choice_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_candidate_choice_predictions.jsonl"),
        "source_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_source_calibrated_predictions.jsonl"),
        "source_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_source_calibrated_predictions.jsonl"),
        "reader_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_evidence_reader_predictions.jsonl"),
        "reader_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_evidence_reader_predictions.jsonl"),
        "readerfb_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_evidence_reader_fallback_predictions.jsonl"),
        "readerfb_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_evidence_reader_fallback_predictions.jsonl"),
        "support_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_evidence_reader_support_first_predictions.jsonl"),
        "support_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_evidence_reader_support_first_predictions.jsonl"),
        "supportfb_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_evidence_reader_support_first_fallback_predictions.jsonl"),
        "supportfb_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_evidence_reader_support_first_fallback_predictions.jsonl"),
        "final_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_final_verifier_predictions.jsonl"),
        "final_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_final_verifier_predictions.jsonl"),
        "guarded_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_guarded_judge_contextual_prior_v2_predictions.jsonl"),
        "guarded_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_guarded_judge_contextual_prior_v2_predictions.jsonl"),
        "judge_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_candidate_judge_v2_all_predictions.jsonl"),
        "judge_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_candidate_judge_v2_all_predictions.jsonl"),
        "prior_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_contextual_prior_v2_predictions.jsonl"),
        "prior_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_contextual_prior_v2_predictions.jsonl"),
        "meta_old": rows_by_qid(ROOT / "results/wise/hotpot_old500_comagraag_meta_selector_v2_predictions.jsonl"),
        "meta_extra": rows_by_qid(ROOT / "results/wise/hotpot_extra500_comagraag_meta_selector_v2_predictions.jsonl"),
    }

    target_ids = [qid(row, i) for i, row in enumerate(current_rows) if target_reasons(row)]
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    target_set = set(target_ids)
    existing = rows_by_qid(args.out_jsonl)

    for i, row in enumerate(current_rows, start=1):
        row_qid = qid(row, i - 1)
        if row_qid in existing:
            continue
        current = answer(row, "answer", "prediction")
        gold = answer(row, "gold")
        reasons = target_reasons(row)
        should_call = row_qid in target_set
        candidates = collect_candidates(row_qid, current, tables) if should_call else []
        selected = current
        selected_rule = "current_v3"
        obj: dict[str, Any] = {}
        raw = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        acceptance = "not_target"
        error = ""
        started = time.time()
        if should_call:
            item = data.get(row_qid, {})
            prompt = build_prompt(
                question=str(row.get("question") or item.get("question") or ""),
                current=current,
                candidates=candidates,
                passages=compact_passages(passages_from_context(item.get("context", []))),
            )
            try:
                obj, usage, raw = llm_json(prompt, args.model)
                proposed = str(obj.get("answer") or "").strip()
                ok, acceptance = acceptable_switch(current, proposed, obj, candidates)
                if ok:
                    selected = proposed
                    selected_rule = "adjudicator_switch"
            except Exception as exc:
                error = str(exc)
                acceptance = "error"
        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "conservative_evidence_adjudicator",
            "question": row.get("question", ""),
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_rule": selected_rule,
            "previous_rule": row.get("previous_rule", ""),
            "current_answer": current,
            "target_reasons": "|".join(reasons) if should_call else "",
            "llm_called": int(should_call),
            "adjudicator_action": obj.get("action", ""),
            "adjudicator_answer": obj.get("answer", ""),
            "adjudicator_confidence": obj.get("confidence", ""),
            "adjudicator_source": obj.get("source", ""),
            "adjudicator_why": obj.get("why", ""),
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
                    "mode": "conservative_evidence_adjudicator",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                },
            )
        if should_call and len([r for r in existing.values() if r.get("llm_called")]) % 20 == 0:
            print(f"adjudicated {len([r for r in existing.values() if r.get('llm_called')])}/{len(target_set)} targets", flush=True)

    final_rows = [existing[qid(row, i)] for i, row in enumerate(current_rows) if qid(row, i) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
