#!/usr/bin/env python3
"""Targeted pure-internal conflict adjudicator for CoMaGRAG.

This script ranks CoMaGRAG-internal answer conflicts, asks an LLM to choose
among the current answer and a small set of internal candidates, and reports
metrics. External baseline predictions are not used. Gold answers are used only
for reporting after selection.
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
    "do not mention",
    "cannot determine",
    "none of",
    "no triple",
    "not specified",
    "instructions",
    "using only",
    "but perhaps",
    "perhaps the answer",
    "the question is",
    "graph doesn",
)

EXCLUDED_FIELDS = {
    "_id",
    "id",
    "qid",
    "question",
    "gold",
    "gold_answer",
    "em",
    "f1",
    "raw_response",
    "error",
    "fallback_error",
    "mode",
    "variant",
    "type",
    "selected_source",
    "selected_choice",
    "selected_rule",
    "previous_rule",
    "original_rule",
    "adjudicator_source",
    "adjudicator_confidence",
    "confidence",
    "cache_hit",
    "iterations",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "wall_time",
    "max_iter",
}

CONFIDENCE_ORDER = {"low": 1, "medium": 2, "high": 3}

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "did",
    "does",
    "do",
    "which",
    "what",
    "who",
    "where",
    "when",
    "both",
}


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


def load_data(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
    else:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and "data" in data:
            rows = data["data"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported data shape: {path}")
    return {qid(row, i): row for i, row in enumerate(rows)}


def source_name(path: Path) -> str:
    return path.stem.replace("_predictions", "")


def source_kind(source: str) -> str:
    if "candidate_choice" in source:
        return "choice"
    if "contextual_prior_v2" in source:
        return "prior"
    if "guarded_judge" in source:
        return "guarded"
    if "evidence_reader_support_first" in source:
        return "support_first"
    if "evidence_reader" in source:
        return "reader"
    if "final_verifier" in source:
        return "verifier"
    if "meta_selector" in source:
        return "meta"
    return "other"


def bad_surface(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    return (
        not normalize(value)
        or len(tokens(value)) > 14
        or len(lowered) > 220
        or lowered.startswith(("{", "- "))
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def minor_alias_change(current: str, candidate: str) -> bool:
    current_norm = normalize(current)
    candidate_norm = normalize(candidate)
    if not current_norm or not candidate_norm or current_norm == candidate_norm:
        return False
    current_tokens = current_norm.split()
    candidate_tokens = candidate_norm.split()
    if abs(len(current_tokens) - len(candidate_tokens)) > 4:
        return False
    if current_norm in candidate_norm or candidate_norm in current_norm:
        return True
    current_set = set(current_tokens)
    candidate_set = set(candidate_tokens)
    return current_set.issubset(candidate_set) or candidate_set.issubset(current_set)


def question_type(question: str) -> str:
    lowered = question.lower().strip()
    if any(
        marker in lowered
        for marker in (
            "born first",
            "born later",
            "older",
            "younger",
            "newer",
            "released first",
            "came out first",
            "appeared first",
            "closer",
            "nearer",
            "further north",
            "further south",
            "further east",
            "further west",
            "farther north",
            "farther south",
            "farther east",
            "farther west",
            "more north",
            "more south",
            "more east",
            "more west",
            "more ",
            "larger",
            "bigger",
            "which one",
            "between the two",
        )
    ):
        return "comparison"
    if lowered.startswith(("are ", "were ", "do ", "did ", "does ", "is ", "was ", "has ", "have ", "had ", "can ")):
        return "yesno"
    if any(marker in lowered for marker in ("what year", "how many", "when was", "when did", "period", "date", "population", "number")):
        return "number_date"
    if lowered.startswith("where") or any(marker in lowered for marker in ("what city", "what country", "what region", "located", "headquartered")):
        return "location"
    if lowered.startswith("who") or " who " in lowered[:160]:
        return "person"
    if " both " in f" {lowered} " or " in common" in lowered or "same " in lowered:
        return "common"
    return "other"


def is_candidate_field(field: str, value: Any) -> bool:
    if field in EXCLUDED_FIELDS or not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return False
    lowered = field.lower()
    if lowered in {
        "answer",
        "prediction",
        "baseline_answer",
        "candidate_contextual_prior_v2",
        "contextual_prior_v2",
        "candidate_context",
        "context_answer",
        "candidate_kg",
        "kg",
        "judge_v2",
        "candidate_judge_v2",
        "candidate_choice_answer",
        "source_calibrated_answer",
    }:
        return True
    return lowered.startswith("candidate_") or lowered.endswith("_answer")


def field_weight(source: str, field: str) -> float:
    lowered = field.lower()
    if "contextual_prior_v2" in lowered:
        weight = 6.0
    elif "candidate_context" in lowered or "context_answer" in lowered:
        weight = 5.0
    elif "candidate_choice_answer" in lowered:
        weight = 4.0
    elif "source_calibrated_answer" in lowered:
        weight = 4.0
    elif "reader_support" in lowered:
        weight = 3.0
    elif "reader_v1" in lowered:
        weight = 2.5
    elif "judge_v2" in lowered:
        weight = 2.0
    elif "candidate_kg" in lowered or lowered == "kg" or "|kg" in lowered:
        weight = 1.2
    elif lowered in {"answer", "prediction", "baseline_answer"}:
        weight = 0.8
    else:
        weight = 1.0
    return weight + {
        "choice": 2.0,
        "prior": 1.5,
        "reader": 1.0,
        "support_first": 1.0,
        "verifier": 0.8,
        "guarded": 0.7,
        "meta": 0.3,
    }.get(source_kind(source), 0.0)


def candidate_score(question: str, answer: str, occurrences: list[dict[str, str]]) -> float:
    if bad_surface(answer):
        return -999.0
    qtype = question_type(question)
    answer_norm = normalize(answer)
    fields = [occurrence["field"].lower() for occurrence in occurrences]
    sources = [occurrence["source"] for occurrence in occurrences]
    score = sum(field_weight(occurrence["source"], occurrence["field"]) for occurrence in occurrences)
    score += min(len(occurrences), 10) * 0.8
    score += len({source_kind(source) for source in sources}) * 1.5
    if any("contextual_prior_v2" in field for field in fields):
        score += 3.0
    if any("candidate_context" in field or "context_answer" in field for field in fields):
        score += 2.0
    if any(source_kind(source) == "choice" for source in sources):
        score += 2.0
    if qtype == "yesno":
        if answer_norm in {"yes", "no"}:
            score += 4.0
        else:
            return -999.0
    elif answer_norm in {"yes", "no"}:
        score -= 5.0
    if qtype == "number_date" and re.search(r"\d", answer):
        score += 2.0
    if qtype in {"number_date", "comparison"} and len(tokens(answer)) <= 5:
        score += 1.0
    return score


def collect_candidates(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        source = source_name(path)
        for i, row in enumerate(load_jsonl(path)):
            row_qid = qid(row, i)
            for field, value in row.items():
                if is_candidate_field(field, value) and not bad_surface(value):
                    candidates[row_qid].append(
                        {"source": source, "field": field, "answer": str(value).strip()}
                    )
    return candidates


def grouped_candidates(
    question: str,
    current: str,
    candidates: list[dict[str, str]],
    max_candidates: int,
    allow_alias_candidates: bool,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    def add(answer: str, source: str, field: str) -> None:
        if bad_surface(answer):
            return
        key = normalize(answer)
        if not key:
            return
        group = groups.setdefault(
            key,
            {
                "answer": answer,
                "norm": key,
                "sources": [],
                "fields": [],
                "occurrences": [],
                "score": 0.0,
                "is_current": False,
                "minor_alias": False,
            },
        )
        if len(answer) < len(group["answer"]):
            group["answer"] = answer
        group["sources"].append(source)
        group["fields"].append(field)
        group["occurrences"].append({"source": source, "field": field, "answer": answer})
        if source == "current":
            group["is_current"] = True

    add(current, "current", "answer")
    for candidate in candidates:
        add(candidate["answer"], candidate["source"], candidate["field"])

    for group in groups.values():
        group["minor_alias"] = minor_alias_change(current, group["answer"])
        group["score"] = candidate_score(question, group["answer"], group["occurrences"])

    current_norm = normalize(current)
    challengers = [
        group
        for norm_key, group in groups.items()
        if norm_key != current_norm
        and (allow_alias_candidates or not group["minor_alias"])
        and group["score"] > -100
    ]
    challengers.sort(key=lambda item: (item["score"], len(item["sources"])), reverse=True)
    current_group = groups.get(current_norm) or {
        "answer": current,
        "norm": current_norm,
        "sources": ["current"],
        "fields": ["answer"],
        "occurrences": [{"source": "current", "field": "answer", "answer": current}],
        "score": 0.0,
        "is_current": True,
        "minor_alias": False,
    }
    return [current_group] + challengers[: max_candidates - 1]


def rank_value(question: str, groups: list[dict[str, Any]]) -> float:
    if len(groups) <= 1:
        return -999.0
    qtype = question_type(question)
    current_score = groups[0]["score"] + 10.0
    top = groups[1]
    value = top["score"] - current_score
    value += min(len(top["sources"]), 12) * 0.8
    if qtype in {"yesno", "comparison", "number_date", "person"}:
        value += 4.0
    elif qtype == "other":
        value -= 1.0
    if top["minor_alias"]:
        value -= 8.0
    return value


def bm25_terms(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in STOPWORDS]


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


def focused_passages(question: str, candidate_answers: list[str], context: Any, max_passages: int, max_chars: int) -> list[str]:
    passages = passages_from_context(context)
    query_terms = set(bm25_terms(" ".join([question] + candidate_answers)))
    scored = []
    for passage in passages:
        terms = bm25_terms(passage)
        score = sum(1 for term in terms if term in query_terms)
        scored.append((score, passage))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    used = 0
    for _score, passage in scored:
        passage = re.sub(r"\s+", " ", passage).strip()
        if not passage:
            continue
        room = max_chars - used
        if room <= 0:
            break
        if len(passage) > room:
            passage = passage[:room].rsplit(" ", 1)[0]
        out.append(passage)
        used += len(passage)
        if len(out) >= max_passages:
            break
    return out


def format_candidates(groups: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    label_map = {}
    lines = []
    for index, group in enumerate(groups, start=1):
        label = f"C{index}"
        label_map[label] = group
        source_counts = Counter(group["sources"])
        support = ", ".join(f"{source}x{count}" for source, count in source_counts.most_common(5))
        current = " current" if index == 1 else ""
        lines.append(f"{label}.{current} {group['answer']}  [support: {support}]")
    return "\n".join(lines), label_map


def build_prompt(question: str, groups: list[dict[str, Any]], passages: list[str], aggressive: bool) -> tuple[str, dict[str, dict[str, Any]]]:
    candidate_lines, label_map = format_candidates(groups)
    passage_lines = "\n".join(f"[{i + 1}] {passage}" for i, passage in enumerate(passages))
    tie_rule = (
        "If evidence is close, choose the candidate that best matches the question wording and support list."
        if aggressive
        else "If evidence is ambiguous or tied, choose C1."
    )
    prompt = f"""You are choosing the final short answer for a HotpotQA exact-match benchmark.

Use only the evidence passages and the listed CoMaGRAG-internal candidate answers. Candidate C1 is the current answer. Other candidates are challengers produced by CoMaGRAG modules.

Question:
{question}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Rules:
- Return only a candidate label, not a rewritten answer.
- For yes/no questions, choose a yes/no candidate when the evidence supports it.
- For comparison questions, choose the entity or yes/no candidate supported by both compared facts.
- For common/shared-property questions, choose the candidate at the requested granularity.
- Do not use outside knowledge.
- {tie_rule}

Return strict JSON only:
{{"choice": "C1", "confidence": "high|medium|low", "why": "short evidence reason"}}
"""
    return prompt, label_map


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


def confidence_allows(confidence: str, threshold: str) -> bool:
    return CONFIDENCE_ORDER.get(confidence.lower(), 0) >= CONFIDENCE_ORDER[threshold]


def choose_from_response(
    current: str,
    obj: dict[str, Any],
    label_map: dict[str, dict[str, Any]],
    switch_confidence: str,
    allow_alias_switch: bool,
    reject_absence_switch: bool,
) -> tuple[str, str, str, str]:
    choice = str(obj.get("choice") or "C1").strip().upper()
    confidence = str(obj.get("confidence") or "").strip().lower()
    if choice not in label_map:
        return current, "C1", confidence, "invalid_choice"
    if choice == "C1":
        return current, choice, confidence, "chose_current"
    selected = str(label_map[choice]["answer"] or "").strip()
    if not confidence_allows(confidence, switch_confidence):
        return current, choice, confidence, "low_confidence"
    if bad_surface(selected):
        return current, choice, confidence, "bad_surface"
    if minor_alias_change(current, selected) and not allow_alias_switch:
        return current, choice, confidence, "minor_alias"
    if reject_absence_switch:
        why = str(obj.get("why") or "").lower()
        absence_markers = (
            "no evidence",
            "not enough evidence",
            "none of the evidence",
            "does not mention",
            "do not mention",
            "doesn't mention",
            "no passage mentions",
            "no passage states",
            "absence of",
            "ambiguous",
            "cannot determine",
            "not provided",
            "not specified",
        )
        if any(marker in why for marker in absence_markers):
            return current, choice, confidence, "absence_reason"
    return selected, choice, confidence, "accepted"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "internal_conflict_adjudicator",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "targets": sum(int(row.get("targeted", 0)) for row in rows),
        "llm_calls": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(row.get("selected_rule") == "internal_adjudicator_switch" for row in rows),
    }
    summary.update({f"acceptance_{key}": value for key, value in Counter(row.get("acceptance", "") for row in rows).items()})
    summary.update({f"qtype_{key}": value for key, value in Counter(row.get("question_type", "") for row in rows if row.get("targeted")).items()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, default=None)
    parser.add_argument("--rank-limit", type=int, default=100)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-passages", type=int, default=10)
    parser.add_argument("--max-passage-chars", type=int, default=6500)
    parser.add_argument("--switch-confidence", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--allow-alias-candidates", action="store_true")
    parser.add_argument("--allow-alias-switch", action="store_true")
    parser.add_argument(
        "--target-question-type",
        action="append",
        choices=("comparison", "number_date", "person", "location", "other", "yesno", "common"),
        help="Question type to target. Repeat for multiple. Defaults to comparison/number_date/person/location/other.",
    )
    parser.add_argument(
        "--allow-absence-switch",
        action="store_true",
        help="Allow switches justified by missing/ambiguous evidence. Disabled by default.",
    )
    args = parser.parse_args()
    target_question_types = set(
        args.target_question_type
        or ["comparison", "number_date", "person", "location", "other"]
    )

    data_by_id = load_data(args.data)
    current_rows = load_jsonl(args.current)
    candidate_by_id = collect_candidates(args.candidate)

    ranked = []
    groups_by_id: dict[str, list[dict[str, Any]]] = {}
    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        question = str(row.get("question") or data_by_id.get(row_qid, {}).get("question") or "")
        if question_type(question) not in target_question_types:
            continue
        current = str(row.get("prediction") or row.get("answer") or "")
        groups = grouped_candidates(
            question,
            current,
            candidate_by_id.get(row_qid, []),
            max_candidates=args.max_candidates,
            allow_alias_candidates=args.allow_alias_candidates,
        )
        groups_by_id[row_qid] = groups
        value = rank_value(question, groups)
        if value > -100 and len(groups) > 1:
            ranked.append((value, i, row_qid))
    ranked.sort(reverse=True)
    target_ids = {row_qid for _value, _i, row_qid in ranked[: args.rank_limit]}
    if args.max_llm_calls is not None:
        target_ids = {row_qid for _value, _i, row_qid in ranked[: args.max_llm_calls]}

    existing = {qid(row, i): row for i, row in enumerate(load_jsonl(args.out_jsonl))} if args.out_jsonl.exists() else {}
    llm_calls = sum(int(row.get("llm_called", 0)) for row in existing.values())

    for i, row in enumerate(current_rows):
        row_qid = qid(row, i)
        if row_qid in existing:
            continue
        item = data_by_id.get(row_qid, {})
        question = str(row.get("question") or item.get("question") or "")
        current = str(row.get("prediction") or row.get("answer") or "")
        gold = str(row.get("gold") or item.get("answer") or "")
        qtype = question_type(question)
        groups = groups_by_id.get(row_qid, [])
        target = row_qid in target_ids
        selected = current
        selected_rule = "current"
        obj: dict[str, Any] = {}
        raw = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        choice = "C1"
        confidence = ""
        acceptance = "not_target"
        error = ""
        started = time.time()
        should_call = target and (args.max_llm_calls is None or llm_calls < args.max_llm_calls)
        if should_call:
            try:
                passages = focused_passages(
                    question,
                    [str(group.get("answer") or "") for group in groups],
                    item.get("context", []),
                    max_passages=args.max_passages,
                    max_chars=args.max_passage_chars,
                )
                prompt, label_map = build_prompt(question, groups, passages, aggressive=args.aggressive)
                obj, usage, raw = llm_json(prompt, args.model)
                selected, choice, confidence, acceptance = choose_from_response(
                    current,
                    obj,
                    label_map,
                    switch_confidence=args.switch_confidence,
                    allow_alias_switch=args.allow_alias_switch,
                    reject_absence_switch=not args.allow_absence_switch,
                )
                if normalize(selected) != normalize(current):
                    selected_rule = "internal_adjudicator_switch"
                llm_calls += 1
            except Exception as exc:
                error = str(exc)
                acceptance = "error"

        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "internal_conflict_adjudicator",
            "question": question,
            "question_type": qtype,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "selected_rule": selected_rule,
            "current_answer": current,
            "targeted": int(target),
            "llm_called": int(should_call),
            "rank_value": round(rank_value(question, groups), 4),
            "candidate_labels": " | ".join(f"C{idx + 1}={group['answer']}" for idx, group in enumerate(groups)),
            "adjudicator_choice": choice,
            "adjudicator_confidence": confidence,
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
                    "mode": "internal_conflict_adjudicator",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                },
            )
        if should_call and llm_calls % 20 == 0:
            print(f"internal adjudicator calls={llm_calls}/{len(target_ids)}", flush=True)

    final_rows = [existing[qid(row, i)] for i, row in enumerate(current_rows) if qid(row, i) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
