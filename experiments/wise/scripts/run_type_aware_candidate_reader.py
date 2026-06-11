#!/usr/bin/env python3
"""Type-aware candidate generator and evidence reader for HotpotQA.

This is a no-rebuild experiment: it uses the original per-question context to
generate missing candidates for hard question types before selecting a concise
answer. Gold answers are used only for reporting metrics.
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
    "cannot",
    "cannot determine",
    "does not mention",
    "do not mention",
    "insufficient",
    "no information",
    "not enough",
    "not specified",
    "provided context",
    "unknown",
)

DEFAULT_TARGET_QTYPES = ("comparison", "yesno", "number_date", "common", "location", "disjunctive_fact")
MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)
NUMBER_WORD_RE = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
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
    return {qid(row, i): row for i, row in enumerate(rows)}


def answer(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, "") and str(value).strip().lower() not in {"nan", "none", "null"}:
            return str(value).strip()
    return ""


def passages_from_context(context: Any) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return [str(p) for p in context if str(p).strip()]
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


def compact_passages(passages: list[str], question: str, candidates: list[str], max_chars: int) -> list[str]:
    if not passages:
        return []
    query_terms = set(tokens(" ".join([question] + candidates)))
    scored = []
    for index, passage in enumerate(passages):
        passage = re.sub(r"\s+", " ", passage).strip()
        passage_terms = set(tokens(passage))
        score = len(query_terms & passage_terms)
        if any(normalize(candidate) and normalize(candidate) in normalize(passage) for candidate in candidates):
            score += 3
        scored.append((score, -index, passage))
    scored.sort(reverse=True)

    kept = []
    used = 0
    for _score, _neg_index, passage in scored:
        room = max_chars - used
        if room <= 0:
            break
        if len(passage) > room:
            passage = passage[:room].rsplit(" ", 1)[0]
        if passage:
            kept.append(passage)
            used += len(passage)
    return kept


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
            "released earlier",
            "released later",
            "came out first",
            "came out earlier",
            "came out later",
            "appeared first",
            "opened first",
            "opened earlier",
            "opened later",
            "earlier",
            "later",
            "lived longer",
            "longer middle name",
            "longer ",
            "which was developed first",
            "higher numbered",
            "higher population",
            "more species",
            "most number",
            "closer",
            "nearer",
            "larger",
            "bigger",
            "more ",
            "further north",
            "further south",
            "further east",
            "further west",
            "farther north",
            "farther south",
            "farther east",
            "farther west",
            "between the two",
        )
    ):
        return "comparison"
    if (
        lowered.startswith(("did ", "does ", "do ", "has ", "have ", "had "))
        and " or " in f" {lowered} "
        and " both " not in f" {lowered} "
    ):
        return "disjunctive_fact"
    if lowered.startswith(("are ", "were ", "do ", "did ", "does ", "is ", "was ", "has ", "have ", "had ", "can ")):
        return "yesno"
    if any(
        marker in lowered
        for marker in (
            "what year",
            "which year",
            "how many",
            "when was",
            "when did",
            "when were",
            "what date",
            "which president",
            "what president",
            "what number",
            "which number",
            "opened on what date",
            "founded on what date",
            "period",
            "population",
            "number",
            "date?",
        )
    ):
        return "number_date"
    if lowered.startswith("where") or any(
        marker in lowered
        for marker in ("what city", "what country", "what region", "what county", "located", "headquartered")
    ):
        return "location"
    if " both " in f" {lowered} " or " in common" in lowered or "same " in lowered:
        return "common"
    if lowered.startswith("who") or " who " in lowered[:160]:
        return "person"
    return "other"


def looks_bad(value: str, max_words: int = 14) -> bool:
    lowered = str(value or "").lower()
    return (
        not normalize(value)
        or len(tokens(value)) > max_words
        or len(str(value)) > 220
        or lowered.startswith(("{", "- "))
        or any(marker in lowered for marker in BAD_MARKERS)
    )


def clean_candidate(value: str) -> str:
    value = str(value or "").strip().strip("\"'`")
    value = re.sub(r"^(final answer|answer)\s*:\s*", "", value, flags=re.I).strip()
    value = re.sub(r"^(the answer is|therefore|thus|so)\s+", "", value, flags=re.I).strip()
    return value.strip(" .;")


def unique(values: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    kept = []
    for value in values:
        value = clean_candidate(value)
        key = normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(value)
        if limit is not None and len(kept) >= limit:
            break
    return kept


def compared_entities(question: str) -> list[str]:
    patterns = [
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:,|\?|$)",
        r"\b(?:which|who|what)\b.+?,\s*(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+or\s+(.+?)(?:\?|$)",
    ]
    found = []
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.I)
        if not match:
            continue
        for part in match.groups():
            part = re.sub(r"^(which|who|what|where)\b", "", part, flags=re.I).strip(" ,.?")
            part = re.sub(r"^(did|does|do|has|have|had|is|are|was|were|can)\s+", "", part, flags=re.I).strip(" ,.?")
            part = re.sub(r"\b(has|have|had|is|are|was|were|located|opened|released).*$", "", part, flags=re.I).strip()
            if 1 <= len(tokens(part)) <= 8:
                found.append(part)
        if found:
            break
    return unique(found, limit=4)


def leading_name(text: str) -> str:
    match = re.match(
        r"^([A-Z][A-Za-z0-9'&.\-]*(?:\s+[A-Z][A-Za-z0-9'&.\-]*){0,8})"
        r"(?:\s+\(|\s+is\b|\s+are\b|\s+was\b|\s+were\b|\s*,|$)",
        text.strip(),
    )
    return match.group(1).strip() if match else ""


def number_date_candidates(passages: list[str]) -> list[str]:
    text = "\n".join(passages)
    candidates = []
    candidates.extend(re.findall(rf"\b(?:{MONTH_RE})\s+\d{{1,2}},?\s+\d{{4}}\b", text, flags=re.I))
    candidates.extend(re.findall(rf"\b\d{{1,2}}\s+(?:{MONTH_RE})\s+\d{{4}}\b", text, flags=re.I))
    candidates.extend(re.findall(r"\b\d+(?:st|nd|rd|th)\b", text, flags=re.I))
    candidates.extend(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text))
    candidates.extend(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", text))
    candidates.extend(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    candidates.extend(re.findall(rf"\b(?:{NUMBER_WORD_RE})(?:\s+(?:{NUMBER_WORD_RE}))*\b", text, flags=re.I))
    return unique(candidates, limit=20)


def president_ordinal_candidates(question: str, passages: list[str]) -> list[str]:
    lowered = question.lower()
    if "president of the united states" not in lowered and "which president" not in lowered:
        return []
    text = "\n".join(passages)
    patterns = [
        r"\b(\d{1,2}(?:st|nd|rd|th))\s+President of the United States\b",
        r"\bPresident of the United States,\s*(\d{1,2}(?:st|nd|rd|th))\b",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text, flags=re.I))
    return unique(candidates, limit=8)


def sentence_candidates(question: str, passages: list[str], limit: int = 8) -> list[str]:
    query_terms = set(tokens(question))
    scored = []
    for passage in passages:
        _title, _sep, body = passage.partition(":")
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence or len(tokens(sentence)) > 22:
                continue
            sentence_terms = set(tokens(sentence))
            score = len(query_terms & sentence_terms)
            if re.search(r"\d", sentence):
                score += 2
            if score > 1:
                scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return unique([sentence for _score, sentence in scored], limit=limit)


def location_candidates(passages: list[str]) -> list[str]:
    candidates = []
    for passage in passages:
        for pattern in (
            r"\b(?:located|based|headquartered|situated)\s+in\s+([A-Z][A-Za-z .'-]{2,80})",
            r"\bin\s+the\s+(?:city|town|village|county|region|province|state|country)\s+of\s+([A-Z][A-Za-z .'-]{2,80})",
            r"\b(?:city|town|village|county|region|province|state|country)\s+of\s+([A-Z][A-Za-z .'-]{2,80})",
        ):
            for match in re.finditer(pattern, passage):
                value = re.split(r"[.;,()]|\s+and\s+", match.group(1), maxsplit=1)[0].strip()
                if value:
                    candidates.append(value)
    return unique(candidates, limit=12)


def person_candidates(passages: list[str]) -> list[str]:
    candidates = []
    for passage in passages:
        _title, _sep, body = passage.partition(":")
        name = leading_name(body)
        if name:
            candidates.append(name)
        candidates.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z'.-]+){1,4}\b", passage))
    return unique(candidates, limit=16)


def deterministic_candidates(question: str, qtype: str, current: str, passages: list[str]) -> list[str]:
    candidates = [current]
    if qtype == "yesno":
        candidates.extend(["yes", "no"])
    if qtype in {"comparison", "disjunctive_fact"}:
        candidates.extend(compared_entities(question))
    if qtype == "number_date":
        candidates.extend(number_date_candidates(passages))
        candidates.extend(president_ordinal_candidates(question, passages))
    if qtype == "disjunctive_fact":
        candidates.extend(sentence_candidates(question, passages))
    if qtype == "location":
        candidates.extend(location_candidates(passages))
    if qtype == "common":
        candidates.extend(location_candidates(passages))
    if qtype in {"person", "common", "comparison"}:
        candidates.extend(person_candidates(passages))
    return unique([candidate for candidate in candidates if not looks_bad(candidate, max_words=18)], limit=24)


def parse_json_obj(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", str(text or "")).strip().rstrip("```").strip()
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
            pass

    # Some providers truncate long JSON after the leading fields. Keep usable
    # answer/confidence gates rather than treating the whole call as empty.
    obj: dict[str, Any] = {}
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned, flags=re.S)
    if answer_match:
        obj["answer"] = answer_match.group(1).strip()
    confidence_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', cleaned, flags=re.I)
    if confidence_match:
        obj["confidence"] = confidence_match.group(1).lower()
    switch_match = re.search(r'"should_switch"\s*:\s*(true|false)', cleaned, flags=re.I)
    if switch_match:
        obj["should_switch"] = switch_match.group(1).lower() == "true"
    evidence_match = re.search(r'"evidence"\s*:\s*"([^"]*)"', cleaned, flags=re.S)
    if evidence_match:
        obj["evidence"] = evidence_match.group(1).strip()
    return obj


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


def build_prompt(question: str, qtype: str, current: str, candidates: list[str], passages: list[str]) -> str:
    candidate_lines = "\n".join(f"- {candidate}" for candidate in candidates if candidate)
    passage_lines = "\n".join(f"[{index + 1}] {passage}" for index, passage in enumerate(passages))
    return f"""You are a HotpotQA evidence reader and candidate generator.

Use only the evidence passages. Existing candidate answers are hints and may be wrong. First generate passage-grounded candidate answers for the question type, then select the concise final answer.

Question type: {qtype}
Question:
{question}

Current answer:
{current}

Existing and deterministic candidates:
{candidate_lines}

Evidence passages:
{passage_lines}

Type-specific rules:
- comparison: extract the relevant attribute for both compared entities, compare them, and return the selected entity. Do not return an explanation.
- disjunctive_fact: when the question asks whether X or Y has a stated property, return the passage-supported entity fact in the dataset's wording, not a bare yes/no.
- yesno: verify each required entity/fact separately; do not default to yes just because one fact is true. Return only "yes" or "no".
- number_date: return the exact number, year, date, rank, count, or short period requested.
- location: return the requested city, country, county, region, or place, not an organization unless the question asks for one.
- common: return the shared category, occupation, nationality, location, or property at the granularity asked.
- person: return the person's name, not their role or work.

Selection rules:
- The final answer must be directly supported by the passages.
- If the current answer is wrong or incomplete, replace it with a generated candidate.
- If the passages do not support a better answer, keep the current answer.
- Do not use outside knowledge. If a compared entity or attribute is missing from the passages, keep the current answer.
- Keep the final answer short enough for exact-match QA evaluation.
- Keep "evidence" under 25 words.

Return strict JSON only:
{{"answer": "...", "confidence": "high|medium|low", "should_switch": true, "generated_candidates": ["..."], "evidence": "brief evidence reason"}}
"""


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


def appears_grounded(value: str, question: str, passages: list[str]) -> bool:
    value_norm = normalize(value)
    if not value_norm:
        return False
    if value_norm in {"yes", "no"}:
        return True
    evidence_norm = normalize(" ".join(passages))
    if value_norm in evidence_norm:
        return True
    value_tokens = tokens(value)
    if len(value_tokens) <= 2:
        return all(token in evidence_norm for token in value_tokens)
    overlap = sum(1 for token in value_tokens if token in evidence_norm)
    return overlap / len(value_tokens) >= 0.75


def min_confidence_ok(actual: str, required: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2}
    return order.get(str(actual or "").lower(), -1) >= order.get(required, 2)


def qtype_answer_ok(qtype: str, question: str, proposed: str) -> tuple[bool, str]:
    question_lower = question.lower()
    proposed_norm = normalize(proposed)
    if qtype == "yesno" and proposed_norm not in {"yes", "no"}:
        return False, "yesno_not_binary"
    if qtype == "comparison":
        options = compared_entities(question)
        option_norms = {normalize(option) for option in options}
        if options and proposed_norm not in option_norms and proposed_norm not in {"yes", "no"} and len(tokens(proposed)) > 6:
            return False, "comparison_not_option"
    if qtype == "disjunctive_fact" and proposed_norm in {"yes", "no"}:
        return False, "disjunctive_fact_binary"
    if question_lower.startswith("how many") and re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", proposed.strip()):
        return False, "how_many_year"
    if "which president" in question_lower and not re.search(r"\b\d{1,2}(?:st|nd|rd|th)\b", proposed, flags=re.I):
        return False, "president_without_ordinal"
    if (
        qtype == "number_date"
        and not re.search(r"\d", proposed)
        and not re.search(MONTH_RE, proposed, flags=re.I)
        and not re.search(rf"\b(?:{NUMBER_WORD_RE})\b", proposed, flags=re.I)
    ):
        return False, "number_date_without_number"
    if qtype == "location" and re.search(r"\b(University|Airline|Company|Corporation|Inc\.?|Ltd\.?)\b", proposed):
        return False, "location_org_like"
    return True, "qtype_ok"


def should_accept(
    current: str,
    proposed: str,
    obj: dict[str, Any],
    question: str,
    qtype: str,
    passages: list[str],
    min_confidence: str,
    trust_reader: bool,
) -> tuple[bool, str]:
    proposed = clean_candidate(proposed)
    if normalize(proposed) == normalize(current):
        return False, "same_as_current"
    if looks_bad(proposed):
        return False, "bad_surface"
    if not trust_reader and not bool(obj.get("should_switch", False)):
        return False, "should_switch_false"
    if not trust_reader and not min_confidence_ok(str(obj.get("confidence") or ""), min_confidence):
        return False, "low_confidence"
    ok, reason = qtype_answer_ok(qtype, question, proposed)
    if not ok:
        return False, reason
    if not appears_grounded(proposed, question, passages):
        return False, "not_grounded"
    return True, "accepted"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_extra_candidates(specs: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --extra-candidate value: {spec}")
        label, path = spec.split(":", 1)
        loaded[label] = rows_by_qid(load_jsonl(Path(path)))
    return loaded


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
    current_em = sum(row["current_em"] for row in rows) / len(rows) if rows else 0
    current_f1 = sum(row["current_f1"] for row in rows) / len(rows) if rows else 0
    new_em = sum(row["em"] for row in rows) / len(rows) if rows else 0
    new_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0
    switches = [row for row in rows if row.get("selected_rule") == "type_aware_switch"]
    summary = {
        "mode": "type_aware_candidate_reader",
        "n": len(rows),
        "current_EM": round(current_em, 4),
        "current_F1": round(current_f1, 4),
        "EM": round(new_em, 4),
        "F1": round(new_f1, 4),
        "targets": sum(int(row.get("llm_called", 0)) for row in rows),
        "switches": len(switches),
        "switch_wins": sum(1 for row in switches if row["em"] > row["current_em"]),
        "switch_losses": sum(1 for row in switches if row["em"] < row["current_em"]),
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
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of LLM-targeted rows to run")
    parser.add_argument("--start", type=int, default=0, help="Start offset in the current prediction file")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--max-passage-chars", type=int, default=7500)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--min-confidence", choices=("high", "medium", "low"), default="high")
    parser.add_argument("--trust-reader", action="store_true", help="Accept grounded reader switches even without high confidence")
    parser.add_argument("--target-qtype", action="append", choices=("comparison", "yesno", "number_date", "person", "location", "common", "disjunctive_fact", "other"))
    parser.add_argument("--extra-candidate", action="append", default=[], help="Extra candidate JSONL as label:path")
    parser.add_argument("--diagnosis", type=Path, default=None, help="Optional oracle diagnosis CSV for internal targeted debugging")
    parser.add_argument("--diagnosis-class", action="append", default=["no_candidate_overlap"])
    args = parser.parse_args()

    data_by_id = rows_by_qid(load_data(args.data))
    current_rows = load_jsonl(args.current)
    current_rows = current_rows[args.start:]
    target_qtypes = set(args.target_qtype or DEFAULT_TARGET_QTYPES)
    extra_candidates = load_extra_candidates(args.extra_candidate)
    diagnosis_targets = load_diagnosis_targets(args.diagnosis, set(args.diagnosis_class))
    existing = rows_by_qid(load_jsonl(args.out_jsonl)) if args.out_jsonl.exists() else {}

    targeted_calls = 0
    for index, row in enumerate(current_rows, start=1):
        row_qid = qid(row, index - 1 + args.start)
        if row_qid in existing:
            continue
        item = data_by_id.get(row_qid, {})
        question = str(row.get("question") or item.get("question") or "")
        qtype = question_type(question)
        current = answer(row, "answer", "prediction")
        gold = answer(row, "gold") or answer(item, "answer")
        all_passages = passages_from_context(item.get("context", []))

        extra_values = []
        for label, by_id in extra_candidates.items():
            extra_row = by_id.get(row_qid, {})
            value = answer(extra_row, "answer", "prediction", "judge_answer", "context", "kg")
            if value:
                extra_values.append(f"{label}: {value}")
        candidates = deterministic_candidates(question, qtype, current, all_passages)
        candidates = unique(candidates + [value.split(": ", 1)[-1] for value in extra_values], limit=28)
        passages = compact_passages(all_passages, question, candidates, args.max_passage_chars)

        is_target = qtype in target_qtypes and (diagnosis_targets is None or row_qid in diagnosis_targets)
        should_call = is_target and (args.limit is None or targeted_calls < args.limit)
        selected = current
        selected_rule = "current"
        obj: dict[str, Any] = {}
        raw = ""
        error = ""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        acceptance = "not_target" if not is_target else "limit_not_called"
        started = time.time()

        if should_call:
            targeted_calls += 1
            try:
                prompt = build_prompt(question, qtype, current, candidates, passages)
                obj, usage, raw = llm_json(prompt, args.model, args.max_tokens)
                proposed = clean_candidate(str(obj.get("answer") or ""))
                ok, acceptance = should_accept(
                    current=current,
                    proposed=proposed,
                    obj=obj,
                    question=question,
                    qtype=qtype,
                    passages=passages,
                    min_confidence=args.min_confidence,
                    trust_reader=args.trust_reader,
                )
                if ok:
                    selected = proposed
                    selected_rule = "type_aware_switch"
            except Exception as exc:
                error = str(exc)
                acceptance = "error"

        current_em = em(current, gold)
        current_f1 = f1(current, gold)
        out_row = {
            "_id": row_qid,
            "id": row_qid,
            "mode": "type_aware_candidate_reader",
            "question": question,
            "question_type": qtype,
            "answer": selected,
            "prediction": selected,
            "gold": gold,
            "em": em(selected, gold),
            "f1": round(f1(selected, gold), 4),
            "current_answer": current,
            "current_em": current_em,
            "current_f1": round(current_f1, 4),
            "selected_rule": selected_rule,
            "llm_called": int(should_call),
            "acceptance": acceptance,
            "reader_answer": clean_candidate(str(obj.get("answer") or "")),
            "reader_confidence": obj.get("confidence", ""),
            "reader_should_switch": obj.get("should_switch", ""),
            "reader_generated_candidates": " | ".join(str(x) for x in obj.get("generated_candidates", []) if x),
            "deterministic_candidates": " | ".join(candidates),
            "extra_candidates": " | ".join(extra_values),
            "reader_evidence": obj.get("evidence", ""),
            "raw_response": raw,
            "error": error,
        }
        append_jsonl(args.out_jsonl, out_row)
        existing[row_qid] = out_row

        if args.usage_log and should_call:
            append_jsonl(
                args.usage_log,
                {
                    "_id": row_qid,
                    "mode": "type_aware_candidate_reader",
                    "llm_calls": 0 if error else 1,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "wall_time": round(time.time() - started, 4),
                    "error": error,
                },
            )
        if should_call and targeted_calls % 20 == 0:
            print(f"type-aware reader {targeted_calls} targeted calls", flush=True)

    final_rows = [existing[qid(row, i + args.start)] for i, row in enumerate(current_rows) if qid(row, i + args.start) in existing]
    write_summary(final_rows, args.out_csv)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
