#!/usr/bin/env python3
"""Two-stage evidence locator and reader for HotpotQA.

Stage 1 locates the passages/snippets needed to answer the question and proposes
passage-grounded candidates. Stage 2 answers using only the located evidence.
Gold answers are used only for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "comagraag"))

import config  # noqa: E402
import run_type_aware_candidate_reader as base  # noqa: E402


client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)

DEFAULT_TARGET_QTYPES = (
    "comparison",
    "yesno",
    "number_date",
    "person",
    "location",
    "common",
    "disjunctive_fact",
    "other",
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {base.qid(row, i): row for i, row in enumerate(rows)}


def bm25_terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


class GlobalDocRetriever:
    def __init__(self, docs_dir: Path, max_doc_chars: int = 1400) -> None:
        self.docs: list[dict[str, str]] = []
        for path in sorted(docs_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            title = path.stem
            passage = f"{title}: {text[:max_doc_chars].rsplit(' ', 1)[0] if len(text) > max_doc_chars else text}"
            self.docs.append({"title": title, "passage": passage})
        self.bm25 = BM25Okapi([bm25_terms(f"{doc['title']} {doc['passage']}") for doc in self.docs]) if self.docs else None

    def retrieve(self, query: str, top_k: int, min_score: float = 0.0) -> list[dict[str, Any]]:
        if not self.docs or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(bm25_terms(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        hits = []
        for index in ranked[: max(top_k * 4, top_k)]:
            score = float(scores[index])
            if score <= min_score:
                continue
            doc = self.docs[index]
            hits.append({"title": doc["title"], "passage": doc["passage"], "score": round(score, 4)})
            if len(hits) >= top_k:
                break
        return hits


def build_global_query(question: str, current: str, candidates: list[str]) -> str:
    parts = [question, current]
    parts.extend(candidates[:8])
    parts.extend(quoted_constraints(question))
    return " ".join(part for part in parts if part)


def merge_passages(local_passages: list[str], global_hits: list[dict[str, Any]]) -> list[str]:
    merged = []
    seen = set()
    for passage in [*local_passages, *(hit["passage"] for hit in global_hits)]:
        key = base.normalize(passage[:220])
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(passage)
    return merged


def format_global_hits(global_hits: list[dict[str, Any]]) -> str:
    return " | ".join(f"{hit['title']}:{hit['score']}" for hit in global_hits)


def parse_json_obj(text: str) -> dict[str, Any]:
    obj = base.parse_json_obj(text)
    if obj:
        return obj
    cleaned = re.sub(r"```(?:json)?", "", str(text or "")).strip().rstrip("```").strip()
    out: dict[str, Any] = {}
    for key in ("sufficient", "should_switch"):
        match = re.search(rf'"{key}"\s*:\s*(true|false)', cleaned, flags=re.I)
        if match:
            out[key] = match.group(1).lower() == "true"
    for key in ("answer", "confidence", "evidence", "reason"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', cleaned, flags=re.S)
        if match:
            out[key] = match.group(1).strip()
    return out


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


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(left.get("input_tokens", 0)) + int(right.get("input_tokens", 0)),
        "output_tokens": int(left.get("output_tokens", 0)) + int(right.get("output_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
    }


def llm_json(prompt: str, model: str, max_tokens: int, retries: int = 3) -> tuple[dict[str, Any], dict[str, int], str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            return parse_json_obj(raw), usage_from_response(resp), raw
        except Exception:
            if attempt >= retries - 1:
                raise
            time.sleep(2**attempt)
    return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, ""


def format_passages(passages: list[str], max_chars: int, per_passage_chars: int) -> str:
    lines = []
    used = 0
    for index, passage in enumerate(passages, start=1):
        passage = re.sub(r"\s+", " ", str(passage)).strip()
        if not passage:
            continue
        if len(passage) > per_passage_chars:
            passage = passage[:per_passage_chars].rsplit(" ", 1)[0]
        line = f"[{index}] {passage}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def coerce_indices(value: Any, n_passages: int) -> list[int]:
    if not isinstance(value, list):
        return []
    indices = []
    for item in value:
        try:
            index = int(str(item).strip())
        except ValueError:
            continue
        if 1 <= index <= n_passages and index not in indices:
            indices.append(index)
    return indices[:4]


def coerce_string_list(value: Any, limit: int, max_words: int = 28) -> list[str]:
    if not isinstance(value, list):
        return []
    kept = []
    for item in value:
        text = str(item or "").strip()
        if not text or len(base.tokens(text)) > max_words:
            continue
        kept.append(text)
    return base.unique(kept, limit=limit)


def build_locator_prompt(question: str, qtype: str, current: str, candidates: list[str], passages: list[str]) -> str:
    candidate_lines = "\n".join(f"- {candidate}" for candidate in candidates if candidate)
    passage_lines = format_passages(passages, max_chars=9000, per_passage_chars=950)
    return f"""You are locating evidence for a HotpotQA multi-hop question.

Use only the listed passages. Your job is not to answer from memory. Select the smallest set of passages/snippets that jointly prove the final answer.

Question type: {qtype}
Question:
{question}

Current answer:
{current}

Existing deterministic candidates:
{candidate_lines}

Passages:
{passage_lines}

Evidence-location rules:
- For bridge questions, select one passage for the bridge entity/fact and one passage for the final requested answer.
- For comparison questions, select evidence for both compared entities and the compared attribute.
- For yes/no questions, select evidence for every required entity/fact; do not default to yes.
- For number/date/how-many questions, locate the sentence containing the final requested number/date, not just an intermediate year or entity.
- If a passage matches the final predicate but not the bridge entity, treat it as a distractor.
- If the listed passages do not contain enough evidence, set sufficient=false and keep generated_candidates empty.

Return strict JSON only:
{{"passage_indices": [1, 2], "evidence_snippets": ["short quote or paraphrase", "..."], "generated_candidates": ["..."], "sufficient": true, "reason": "under 20 words"}}
"""


def build_reader_prompt(
    question: str,
    qtype: str,
    current: str,
    candidates: list[str],
    locator: dict[str, Any],
    selected_passages: list[str],
) -> str:
    candidate_lines = "\n".join(f"- {candidate}" for candidate in candidates if candidate)
    snippets = coerce_string_list(locator.get("evidence_snippets"), limit=6, max_words=40)
    snippet_lines = "\n".join(f"- {snippet}" for snippet in snippets)
    passage_lines = format_passages(selected_passages, max_chars=4500, per_passage_chars=1200)
    sufficient = bool(locator.get("sufficient", False))
    return f"""You are a HotpotQA exact-answer reader.

Use only the located evidence below. Do not use outside knowledge or unlisted passages. If the located evidence is insufficient, keep the current answer.

Question type: {qtype}
Question:
{question}

Current answer:
{current}

Candidate answers:
{candidate_lines}

Locator sufficient: {str(sufficient).lower()}

Located evidence snippets:
{snippet_lines}

Located passages:
{passage_lines}

Answer rules:
- comparison: return the chosen entity from the question.
- yesno: return only "yes" or "no".
- disjunctive_fact: return the passage-supported entity fact in dataset wording, not bare yes/no.
- number_date: return the final requested number, year, date, rank, count, or short period. Do not return an intermediate bridge year for a how-many question.
- location/common/person: return the requested location, shared property, or person at the asked granularity.
- If the evidence does not prove a different answer, keep the current answer and set should_switch=false.

Return strict JSON only:
{{"answer": "...", "confidence": "high|medium|low", "should_switch": true, "evidence": "under 20 words"}}
"""


def selected_passages(locator: dict[str, Any], passages: list[str], fallback: list[str]) -> list[str]:
    indices = coerce_indices(locator.get("passage_indices"), len(passages))
    selected = [passages[index - 1] for index in indices]
    if selected:
        return selected
    return fallback[:3]


def qtype_answer_ok(qtype: str, question: str, proposed: str) -> tuple[bool, str]:
    ok, reason = base.qtype_answer_ok(qtype, question, proposed)
    if not ok:
        return ok, reason
    question_lower = question.lower()
    proposed_norm = base.normalize(proposed)
    if "which film" in question_lower and proposed_norm not in {base.normalize(x) for x in base.compared_entities(question)}:
        return False, "film_comparison_not_option"
    if "which director" in question_lower and proposed_norm not in {base.normalize(x) for x in base.compared_entities(question)}:
        return False, "director_comparison_not_option"
    return True, "qtype_ok"


def quoted_constraints(question: str) -> list[str]:
    constraints = []
    for match in re.finditer(r'"([^"]{3,80})"', question):
        value = match.group(1).strip()
        if value and len(base.tokens(value)) <= 6:
            constraints.append(value)
    return base.unique(constraints, limit=4)


def evidence_covers_quoted_constraints(question: str, passages: list[str]) -> bool:
    constraints = quoted_constraints(question)
    if not constraints:
        return True
    evidence_norm = base.normalize(" ".join(passages))
    return all(base.normalize(constraint) in evidence_norm for constraint in constraints)


def should_accept(
    current: str,
    proposed: str,
    obj: dict[str, Any],
    locator: dict[str, Any],
    question: str,
    qtype: str,
    passages: list[str],
    min_confidence: str,
) -> tuple[bool, str]:
    proposed = base.clean_candidate(proposed)
    if not bool(locator.get("sufficient", False)):
        return False, "locator_insufficient"
    if not evidence_covers_quoted_constraints(question, passages):
        return False, "missing_quoted_constraint"
    if base.normalize(proposed) == base.normalize(current):
        return False, "same_as_current"
    if base.looks_bad(proposed):
        return False, "bad_surface"
    if not bool(obj.get("should_switch", False)):
        return False, "should_switch_false"
    if not base.min_confidence_ok(str(obj.get("confidence") or ""), min_confidence):
        return False, "low_confidence"
    ok, reason = qtype_answer_ok(qtype, question, proposed)
    if not ok:
        return False, reason
    if not base.appears_grounded(proposed, question, passages):
        return False, "not_grounded"
    return True, "accepted"


def load_diagnosis_targets(path: Path | None, classes: set[str]) -> set[str] | None:
    if path is None:
        return None
    targets = set()
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("classification") in classes:
                targets.add(str(row.get("qid") or row.get("_id") or row.get("id") or ""))
    return targets


def write_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    switches = [row for row in rows if row.get("selected_rule") == "two_stage_switch"]
    called = [row for row in rows if row.get("llm_called")]
    summary = {
        "mode": "two_stage_evidence_reader",
        "n": len(rows),
        "current_EM": round(sum(row["current_em"] for row in rows) / len(rows), 4) if rows else 0,
        "current_F1": round(sum(row["current_f1"] for row in rows) / len(rows), 4) if rows else 0,
        "EM": round(sum(row["em"] for row in rows) / len(rows), 4) if rows else 0,
        "F1": round(sum(row["f1"] for row in rows) / len(rows), 4) if rows else 0,
        "targets": len(called),
        "llm_calls": sum(int(row.get("llm_calls", 0)) for row in rows),
        "switches": len(switches),
        "switch_wins": sum(1 for row in switches if row["em"] > row["current_em"]),
        "switch_losses": sum(1 for row in switches if row["em"] < row["current_em"]),
        "avg_input_tokens": round(sum(row.get("input_tokens", 0) for row in called) / len(called), 2) if called else 0,
        "avg_output_tokens": round(sum(row.get("output_tokens", 0) for row in called) / len(called), 2) if called else 0,
        "avg_wall_time": round(sum(row.get("wall_time", 0.0) for row in called) / len(called), 3) if called else 0,
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
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of targeted rows to run")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--locator-max-tokens", type=int, default=500)
    parser.add_argument("--reader-max-tokens", type=int, default=220)
    parser.add_argument("--min-confidence", choices=("high", "medium", "low"), default="high")
    parser.add_argument("--target-qtype", action="append", choices=DEFAULT_TARGET_QTYPES)
    parser.add_argument("--diagnosis", type=Path, default=None)
    parser.add_argument("--diagnosis-class", action="append", default=["no_candidate_overlap"])
    parser.add_argument("--global-docs", type=Path, default=None, help="Optional local txt-doc directory for BM25 evidence augmentation")
    parser.add_argument("--global-top-k", type=int, default=4)
    parser.add_argument("--global-min-score", type=float, default=0.0)
    parser.add_argument("--global-max-doc-chars", type=int, default=1400)
    args = parser.parse_args()

    data_by_id = rows_by_qid(base.load_data(args.data))
    current_rows = base.load_jsonl(args.current)[args.start:]
    target_qtypes = set(args.target_qtype or DEFAULT_TARGET_QTYPES)
    diagnosis_targets = load_diagnosis_targets(args.diagnosis, set(args.diagnosis_class))
    existing = rows_by_qid(base.load_jsonl(args.out_jsonl)) if args.out_jsonl.exists() else {}
    global_retriever = GlobalDocRetriever(args.global_docs, args.global_max_doc_chars) if args.global_docs else None
    if global_retriever is not None:
        print(f"Loaded {len(global_retriever.docs)} global docs from {args.global_docs}", flush=True)

    targeted_calls = 0
    for index, row in enumerate(current_rows, start=1):
        row_qid = base.qid(row, index - 1 + args.start)
        if row_qid in existing:
            continue

        item = data_by_id.get(row_qid, {})
        question = str(row.get("question") or item.get("question") or "")
        qtype = base.question_type(question)
        current = base.answer(row, "answer", "prediction")
        gold = base.answer(row, "gold") or base.answer(item, "answer")
        passages = base.passages_from_context(item.get("context", []))
        candidates = base.deterministic_candidates(question, qtype, current, passages)

        is_target = qtype in target_qtypes and (diagnosis_targets is None or row_qid in diagnosis_targets)
        should_call = is_target and (args.limit is None or targeted_calls < args.limit)
        global_hits: list[dict[str, Any]] = []
        evidence_pool = passages
        if should_call and global_retriever is not None:
            global_query = build_global_query(question, current, candidates)
            global_hits = global_retriever.retrieve(global_query, top_k=args.global_top_k, min_score=args.global_min_score)
            evidence_pool = merge_passages(passages, global_hits)
        fallback_passages = base.compact_passages(evidence_pool, question, candidates, max_chars=3500)

        selected = current
        selected_rule = "current"
        acceptance = "not_target" if not is_target else "limit_not_called"
        locator: dict[str, Any] = {}
        reader: dict[str, Any] = {}
        locator_raw = ""
        reader_raw = ""
        error = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        llm_calls = 0
        started = time.time()

        if should_call:
            targeted_calls += 1
            try:
                locator_prompt = build_locator_prompt(question, qtype, current, candidates, evidence_pool)
                locator, locator_usage, locator_raw = llm_json(locator_prompt, args.model, args.locator_max_tokens)
                usage = add_usage(usage, locator_usage)
                llm_calls += 1

                locator_candidates = coerce_string_list(locator.get("generated_candidates"), limit=8, max_words=18)
                combined_candidates = base.unique([*candidates, *locator_candidates], limit=28)
                evidence_passages = selected_passages(locator, evidence_pool, fallback_passages)

                reader_prompt = build_reader_prompt(
                    question,
                    qtype,
                    current,
                    combined_candidates,
                    locator,
                    evidence_passages,
                )
                reader, reader_usage, reader_raw = llm_json(reader_prompt, args.model, args.reader_max_tokens)
                usage = add_usage(usage, reader_usage)
                llm_calls += 1

                proposed = base.clean_candidate(str(reader.get("answer") or ""))
                ok, acceptance = should_accept(
                    current=current,
                    proposed=proposed,
                    obj=reader,
                    locator=locator,
                    question=question,
                    qtype=qtype,
                    passages=evidence_passages,
                    min_confidence=args.min_confidence,
                )
                if ok:
                    selected = proposed
                    selected_rule = "two_stage_switch"
                candidates = combined_candidates
            except Exception as exc:
                error = str(exc)
                acceptance = "error"

        current_em = base.em(current, gold)
        current_f1 = base.f1(current, gold)
        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "two_stage_evidence_reader",
            "question": question,
            "question_type": qtype,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "em": base.em(selected, gold),
            "f1": round(base.f1(selected, gold), 4),
            "current_answer": current,
            "current_em": current_em,
            "current_f1": round(current_f1, 4),
            "selected_rule": selected_rule,
            "llm_called": int(should_call),
            "llm_calls": llm_calls,
            "acceptance": acceptance,
            "locator_sufficient": locator.get("sufficient", ""),
            "locator_passage_indices": " | ".join(str(x) for x in coerce_indices(locator.get("passage_indices"), len(evidence_pool))),
            "locator_candidates": " | ".join(coerce_string_list(locator.get("generated_candidates"), limit=8, max_words=18)),
            "locator_snippets": " | ".join(coerce_string_list(locator.get("evidence_snippets"), limit=6, max_words=40)),
            "locator_reason": locator.get("reason", ""),
            "global_hits": format_global_hits(global_hits),
            "global_hit_count": len(global_hits),
            "reader_answer": base.clean_candidate(str(reader.get("answer") or "")),
            "reader_confidence": reader.get("confidence", ""),
            "reader_should_switch": reader.get("should_switch", ""),
            "reader_evidence": reader.get("evidence", ""),
            "deterministic_candidates": " | ".join(candidates),
            "locator_raw": locator_raw,
            "reader_raw": reader_raw,
            "error": error,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "wall_time": round(time.time() - started, 4) if should_call else 0.0,
        }
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if args.usage_log and should_call:
            append_jsonl(
                args.usage_log,
                {
                    "_id": row_qid,
                    "mode": "two_stage_evidence_reader",
                    "llm_calls": llm_calls,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": out_row["wall_time"],
                    "error": error,
                },
            )

        if should_call and targeted_calls % 10 == 0:
            print(f"two-stage reader {targeted_calls} targeted rows", flush=True)

    final_rows = [existing[base.qid(row, i + args.start)] for i, row in enumerate(current_rows) if base.qid(row, i + args.start) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
