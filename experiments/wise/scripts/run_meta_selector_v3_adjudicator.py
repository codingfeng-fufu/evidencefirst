#!/usr/bin/env python3
"""Disagreement-only v3 adjudicator over existing CoMaGRAG candidates.

This keeps the v2 meta-selector output as the current answer and only asks an
LLM to adjudicate questions whose candidate pool contains a competing answer.
Gold answers are used only for reporting metrics after the selection is made.
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
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comagraag"))

import config  # noqa: E402


client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)

BAD_MARKERS = {
    "",
    "null",
    "none",
    "nan",
    "unknown",
    "insufficient information",
    "not enough information",
}

BAD_SURFACE_MARKERS = (
    "context passages",
    "provided context",
    "do not provide",
    "does not provide",
    "does not mention",
    "do not mention",
    "not mentioned",
    "not enough information",
    "insufficient information",
    "neither ",
    "none of ",
    "cannot determine",
    "the answer is",
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

ANSWER_FIELDS = {
    "answer",
    "prediction",
    "baseline_answer",
    "candidate_choice_answer",
    "source_calibrated_answer",
    "current_v4",
    "kg",
    "context",
    "prior",
    "judge_answer",
    "judge_v2",
    "contextual_prior",
    "contextual_prior_v2",
    "context_answer",
    "reader_answer",
    "guarded_answer",
    "candidate_baseline",
    "candidate_context",
    "candidate_contextual_prior",
    "candidate_contextual_prior_v2",
    "candidate_current_v4",
    "candidate_guarded",
    "candidate_judge_v2",
    "candidate_kg",
    "candidate_reader_support",
    "candidate_reader_support_fallback",
    "candidate_reader_v1",
    "candidate_reader_v1_fallback",
}

SOURCE_PRIORITIES = {
    "current_v2": 8.0,
    "candidate_choice": 5.0,
    "meta_selector": 4.5,
    "source_calibrated": 4.0,
    "evidence_reader_fallback": 3.5,
    "evidence_reader_support_first_fallback": 3.5,
    "evidence_reader": 3.0,
    "evidence_reader_support_first": 3.0,
    "final_verifier": 3.0,
    "full": 2.5,
    "guarded": 2.5,
    "candidate_judge": 2.0,
    "contextual_prior": 1.5,
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


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
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


def clean(value: Any) -> str:
    text = str(value or "").strip()
    lowered = normalize(text)
    if lowered in BAD_MARKERS:
        return ""
    if len(text) > 180:
        return ""
    return text


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def is_bad_surface(value: str) -> bool:
    lowered = str(value or "").lower().strip()
    if not lowered:
        return True
    if any(marker in lowered for marker in BAD_SURFACE_MARKERS):
        return True
    if lowered.startswith(("the context", "the provided", "none of the")):
        return True
    if len(tokens(value)) > 12 and not re.fullmatch(r"[\w .,'&()/+-]+", value):
        return True
    return False


def is_minor_alias_change(current: str, selected: str) -> bool:
    current_norm = normalize(current)
    selected_norm = normalize(selected)
    if not current_norm or not selected_norm or current_norm == selected_norm:
        return False
    current_tokens = current_norm.split()
    selected_tokens = selected_norm.split()
    token_delta = abs(len(current_tokens) - len(selected_tokens))
    if token_delta > 4:
        return False
    if current_norm in selected_norm or selected_norm in current_norm:
        return True
    current_set = set(current_tokens)
    selected_set = set(selected_tokens)
    return selected_set.issubset(current_set) or current_set.issubset(selected_set)


def qid(row: dict[str, Any], fallback: str = "") -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def source_name(path: Path) -> str:
    name = path.stem.replace("_predictions", "")
    name = re.sub(r"^hotpot(?:1000|_old500|_extra500)_comagraag_", "", name)
    return name


def source_weight(source: str) -> float:
    for needle, weight in SOURCE_PRIORITIES.items():
        if needle in source:
            return weight
    return 1.0


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


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {qid(row, str(i)): row for i, row in enumerate(rows)}


def passages_from_context(context: Any, max_passage_chars: int) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        passages = [str(p) for p in context if p]
    elif isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        passages = []
        for title, sent_list in zip(titles, sentences):
            body = sent_list if isinstance(sent_list, str) else " ".join(str(s) for s in sent_list)
            if body:
                passages.append(f"{title}: {body}")
    else:
        passages = []
        for title, sent_list in context or []:
            body = sent_list if isinstance(sent_list, str) else " ".join(str(s) for s in sent_list)
            if body:
                passages.append(f"{title}: {body}")
    if max_passage_chars <= 0:
        return passages
    return [p[:max_passage_chars] for p in passages]


def sentence_corpus_from_context(context: Any) -> list[str]:
    passages = passages_from_context(context, max_passage_chars=0)
    corpus = []
    for passage in passages:
        title = ""
        body = passage
        if ":" in passage:
            title, body = passage.split(":", 1)
            title = title.strip()
            body = body.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = sentence.strip()
            if not sentence:
                continue
            corpus.append(f"{title}: {sentence}" if title else sentence)
    return corpus


def bm25_terms(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in STOPWORDS]


def top_bm25(corpus: list[str], query: str, top_k: int) -> list[str]:
    if not corpus:
        return []
    tokenized = [bm25_terms(item) for item in corpus]
    query_terms = bm25_terms(query)
    if not query_terms:
        return corpus[:top_k]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_terms)
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    out = []
    for idx in ranked:
        if scores[idx] <= 0 and out:
            break
        out.append(corpus[idx])
        if len(out) >= top_k:
            break
    return out


def arag_trace_chunks(row: dict[str, Any], max_chunk_chars: int = 600) -> list[str]:
    chunks = []
    for step in row.get("trajectory") or []:
        tool_result = str(step.get("tool_result") or "")
        if not tool_result:
            continue
        parts = re.split(r"\n\s*\n(?=Chunk ID:)", tool_result)
        for part in parts:
            part = " ".join(part.split())
            if part:
                chunks.append(part[:max_chunk_chars])
    return chunks


def focused_evidence(
    question: str,
    candidates: list[dict[str, Any]],
    item: dict[str, Any],
    arag_row: dict[str, Any] | None,
    max_snippets: int,
    max_snippet_chars: int,
) -> list[str]:
    corpus = sentence_corpus_from_context(item.get("context", []))
    snippets = []
    snippets.extend(top_bm25(corpus, question, top_k=6))
    for candidate in candidates:
        snippets.extend(top_bm25(corpus, f"{question} {candidate['answer']}", top_k=2))

    if arag_row:
        chunks = arag_trace_chunks(arag_row)
        query = " ".join([question] + [candidate["answer"] for candidate in candidates])
        snippets.extend(top_bm25(chunks, query, top_k=6))

    out = []
    seen = set()
    for snippet in snippets:
        snippet = snippet[:max_snippet_chars].strip()
        key = normalize(snippet)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(snippet)
        if len(out) >= max_snippets:
            break
    return out


def is_answer_field(key: str, value: Any) -> bool:
    if key in EXCLUDED_FIELDS or not isinstance(value, str):
        return False
    return (
        key in ANSWER_FIELDS
        or key.startswith("candidate_")
        or key.endswith("_answer")
    )


def collect_candidates(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        source = source_name(path)
        for i, row in enumerate(load_jsonl(path)):
            row_qid = qid(row, str(i))
            for field, value in row.items():
                if not is_answer_field(field, value):
                    continue
                answer = clean(value)
                if answer:
                    candidates[row_qid].append(
                        {"source": source, "field": field, "answer": answer}
                    )
    return candidates


def grouped_candidates(
    current_answer: str,
    candidates: list[dict[str, str]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    def add(answer: str, source: str, field: str) -> None:
        answer = clean(answer)
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
                "score": 0.0,
                "is_current": False,
            },
        )
        if len(answer) < len(group["answer"]):
            group["answer"] = answer
        group["sources"].append(source)
        group["fields"].append(field)
        group["score"] += source_weight(source)
        if source == "current_v2":
            group["is_current"] = True

    add(current_answer, "current_v2", "answer")
    for candidate in candidates:
        add(candidate["answer"], candidate["source"], candidate["field"])

    ordered = sorted(
        groups.values(),
        key=lambda item: (item["is_current"], item["score"], len(item["sources"])),
        reverse=True,
    )
    return ordered[:max_candidates]


def format_candidates(
    candidates: list[dict[str, Any]],
    mark_current: bool = True,
) -> tuple[str, dict[str, dict[str, Any]]]:
    label_map: dict[str, dict[str, Any]] = {}
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        label = f"C{index}"
        label_map[label] = candidate
        source_counts = Counter(candidate["sources"])
        top_sources = ", ".join(f"{name}x{count}" for name, count in source_counts.most_common(4))
        current = " current" if mark_current and candidate["is_current"] else ""
        lines.append(f"{label}.{current} {candidate['answer']}  [support: {top_sources}]")
    return "\n".join(lines), label_map


def build_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    passages: list[str],
    neutral: bool,
    label_only: bool,
    independent_answer: bool,
    aggressive_targeted: bool,
) -> tuple[str, dict[str, dict[str, Any]]]:
    candidate_lines, label_map = format_candidates(candidates, mark_current=not neutral)
    passage_lines = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    current_instruction = (
        "Candidate C1 is the current CoMaGRAG v2 answer; it is usually strong. "
        "Switch away from C1 only when the passages clearly support another candidate as the direct answer."
        if not neutral
        else (
            "Candidate C1 is one candidate from the current system, but it is not privileged. "
            "Decide from the evidence and candidate support."
        )
    )
    tie_instruction = (
        "If evidence is ambiguous, choose the candidate with the strongest support list; keep C1 only when candidates are tied."
        if aggressive_targeted
        else "If evidence does not clearly decide, keep C1."
    )
    if independent_answer:
        answer_instruction = (
            "Answer the question from the evidence. Candidate answers are hints only; "
            "any of them may be wrong. Return the concise answer string, not a label."
        )
    else:
        answer_instruction = (
        'Return the selected candidate label only; do not rewrite, shorten, expand, or paraphrase candidate text.'
        if label_only
        else "Prefer exact candidate wording when it is supported. A minimal surface edit is allowed only to remove extra explanation or choose a canonical alias explicitly present in evidence."
        )
    json_shape = (
        '{"answer": "...", "confidence": "high|medium|low"}'
        if independent_answer
        else (
        '{"choice": "C1", "confidence": "high|medium|low"}'
        if label_only
        else '{"choice": "C1", "answer": "...", "confidence": "high|medium|low"}'
        )
    )
    prompt = f"""You are adjudicating the final short answer for a HotpotQA exact-match benchmark.

Use only the evidence passages. {current_instruction}

Question:
{question}

Candidate answers:
{candidate_lines}

Evidence passages:
{passage_lines}

Decision rules:
- Return the shortest canonical answer that directly answers the question.
- For yes/no questions, return only "yes" or "no".
- For comparison questions, return the selected entity or yes/no answer, not the reasoning.
- For "both", "same", "shared", or "common" questions, answer the shared property at the requested granularity.
- {answer_instruction}
- {tie_instruction}

Return strict JSON only:
{json_shape}
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def confidence_allows_switch(confidence: str, threshold: str) -> bool:
    return CONFIDENCE_ORDER.get(confidence.lower(), 0) >= CONFIDENCE_ORDER[threshold]


def choose_answer(
    current_answer: str,
    obj: dict[str, Any],
    label_map: dict[str, dict[str, Any]],
    switch_confidence: str,
    label_only: bool,
    independent_answer: bool,
) -> tuple[str, str, str]:
    confidence = str(obj.get("confidence") or "").strip().lower()
    if independent_answer:
        answer_norm = normalize(clean(obj.get("answer")))
        choice = "C1"
        adjudicated_answer = current_answer
        for label, candidate in label_map.items():
            if answer_norm and answer_norm == normalize(candidate.get("answer", "")):
                choice = label
                adjudicated_answer = candidate["answer"]
                break
    else:
        choice = str(obj.get("choice") or "C1").strip().upper()
        if label_only:
            adjudicated_answer = label_map.get(choice, {}).get("answer", "")
        else:
            adjudicated_answer = clean(obj.get("answer")) or label_map.get(choice, {}).get("answer", "")
    if choice not in label_map:
        choice = "C1"
        adjudicated_answer = current_answer
    if choice == "C1" or not confidence_allows_switch(confidence, switch_confidence):
        return current_answer, choice, confidence
    if is_bad_surface(adjudicated_answer):
        return current_answer, choice, confidence
    if is_minor_alias_change(current_answer, adjudicated_answer) and not is_bad_surface(current_answer):
        return current_answer, choice, confidence
    return adjudicated_answer or label_map[choice]["answer"], choice, confidence


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    summary = {
        "mode": "meta_selector_v3_adjudicator",
        "n": len(rows),
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "selected_current_v2": sum(row["selected_source"] == "current_v2" for row in rows),
        "selected_adjudicator": sum(row["selected_source"] == "adjudicator" for row in rows),
        "llm_calls": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": sum(int(row.get("switched", 0)) for row in rows),
    }
    summary.update({f"confidence_{key}": value for key, value in Counter(row.get("adjudicator_confidence", "") for row in rows).items()})
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-passage-chars", type=int, default=1000)
    parser.add_argument("--switch-confidence", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--neutral-prompt", action="store_true")
    parser.add_argument("--label-only", action="store_true")
    parser.add_argument("--independent-answer", action="store_true")
    parser.add_argument("--aggressive-targeted", action="store_true")
    parser.add_argument("--focused-evidence", action="store_true")
    parser.add_argument("--arag-trace", action="append", type=Path, default=[])
    parser.add_argument("--max-snippets", type=int, default=14)
    parser.add_argument("--max-snippet-chars", type=int, default=700)
    args = parser.parse_args()

    data_by_id = rows_by_qid(load_data(args.data))
    current_rows = load_jsonl(args.current)
    candidate_by_id = collect_candidates(args.candidate)
    arag_trace_by_id: dict[str, dict[str, Any]] = {}
    for trace_path in args.arag_trace:
        arag_trace_by_id.update(rows_by_qid(load_jsonl(trace_path)))
    existing = rows_by_qid(load_jsonl(args.out_jsonl))

    selected_rows = current_rows[args.start:]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    llm_calls = 0
    for index, current_row in enumerate(selected_rows, start=1):
        row_qid = qid(current_row, str(index - 1 + args.start))
        if row_qid in existing:
            continue

        item = data_by_id.get(row_qid, {})
        question = str(current_row.get("question") or item.get("question") or "")
        gold = str(current_row.get("gold") or item.get("answer") or "")
        current_answer = str(current_row.get("answer") or current_row.get("prediction") or "")
        candidates = grouped_candidates(
            current_answer,
            candidate_by_id.get(row_qid, []),
            max_candidates=args.max_candidates,
        )
        has_disagreement = len({candidate["norm"] for candidate in candidates}) > 1

        obj: dict[str, Any] = {}
        raw = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        error = None
        selected = current_answer
        choice = "C1"
        confidence = ""
        llm_called = 0
        adjudicated_answer = current_answer

        if has_disagreement and (
            args.max_llm_calls is None or llm_calls < args.max_llm_calls
        ):
            prompt, label_map = build_prompt(
                question=question,
                candidates=candidates,
                passages=(
                    focused_evidence(
                        question=question,
                        candidates=candidates,
                        item=item,
                        arag_row=arag_trace_by_id.get(row_qid),
                        max_snippets=args.max_snippets,
                        max_snippet_chars=args.max_snippet_chars,
                    )
                    if args.focused_evidence
                    else passages_from_context(item.get("context", []), args.max_passage_chars)
                ),
                neutral=args.neutral_prompt,
                label_only=args.label_only,
                independent_answer=args.independent_answer,
                aggressive_targeted=args.aggressive_targeted,
            )
            started = time.time()
            try:
                obj, usage, raw = llm_json(prompt)
                selected, choice, confidence = choose_answer(
                    current_answer=current_answer,
                    obj=obj,
                    label_map=label_map,
                    switch_confidence=args.switch_confidence,
                    label_only=args.label_only,
                    independent_answer=args.independent_answer,
                )
                adjudicated_answer = (
                    label_map.get(choice, {}).get("answer", current_answer)
                    if args.label_only or args.independent_answer
                    else clean(obj.get("answer")) or label_map.get(choice, {}).get("answer", current_answer)
                )
                llm_calls += 1
                llm_called = 1
            except Exception as exc:
                error = str(exc)
                selected = current_answer
                choice = "C1"
                confidence = ""
            if args.usage_log:
                append_jsonl(
                    args.usage_log,
                    {
                        "_id": row_qid,
                        "mode": "meta_selector_v3_adjudicator",
                        "llm_calls": llm_called,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "wall_time": round(time.time() - started, 4),
                        "error": error,
                    },
                )

        selected_source = "adjudicator" if normalize(selected) != normalize(current_answer) else "current_v2"
        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "meta_selector_v3_adjudicator",
            "question": question,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "current_answer": current_answer,
            "adjudicated_answer": adjudicated_answer,
            "adjudicator_choice": choice,
            "adjudicator_confidence": confidence,
            "selected_source": selected_source,
            "llm_called": llm_called,
            "switched": int(selected_source == "adjudicator"),
            "candidate_count": len(candidates),
            "candidate_answers": [
                {
                    "label": f"C{i}",
                    "answer": candidate["answer"],
                    "is_current": candidate["is_current"],
                    "sources": candidate["sources"],
                    "fields": candidate["fields"],
                }
                for i, candidate in enumerate(candidates, start=1)
            ],
            "raw_response": raw,
            "error": error,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
        }
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if index % 25 == 0:
            print(
                f"[{index}/{len(selected_rows)}] v3 processed, llm_calls={llm_calls}",
                flush=True,
            )

    final_rows = [existing[qid(row, str(i + args.start))] for i, row in enumerate(selected_rows) if qid(row, str(i + args.start)) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
