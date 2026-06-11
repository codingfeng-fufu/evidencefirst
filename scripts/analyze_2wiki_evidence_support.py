#!/usr/bin/env python3
"""Offline 2Wiki support/evidence diagnostics for existing runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi


def load_json_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported data shape: {path}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_jsonl(path)


def norm(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = norm(prediction).split()
    gold_tokens = norm(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def context_sentences(item: dict) -> list[dict]:
    rows = []
    for title, sentences in item.get("context", []):
        for idx, sentence in enumerate(sentences):
            rows.append({"title": str(title), "sent_idx": idx, "text": f"{title}: {sentence}"})
    return rows


def support_titles(item: dict) -> set[str]:
    return {norm(title) for title, _idx in item.get("supporting_facts", [])}


def evidence_entities(item: dict) -> set[str]:
    entities = set()
    for triple in item.get("evidences", []):
        if len(triple) >= 3:
            entities.add(norm(triple[0]))
            entities.add(norm(triple[2]))
    answer = item.get("answer")
    if answer:
        entities.add(norm(answer))
    return {entity for entity in entities if entity}


def bm25_top_titles(item: dict, top_k: int) -> set[str]:
    sentences = context_sentences(item)
    if not sentences:
        return set()
    bm25 = BM25Okapi([row["text"].lower().split() for row in sentences])
    scores = bm25.get_scores(item["question"].lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return {norm(sentences[idx]["title"]) for idx in top_idxs}


def all_context_titles(item: dict) -> set[str]:
    return {norm(title) for title, _sentences in item.get("context", [])}


def load_chunk_titles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        chunks = json.load(file)
    return {str(chunk.get("id")): norm(chunk.get("title", "")) for chunk in chunks}


def arag_titles(row: dict | None, chunk_titles: dict[str, str]) -> tuple[set[str], set[str]]:
    if not row:
        return set(), set()
    read_ids = {str(chunk_id) for chunk_id in row.get("chunks_read_ids", []) or []}
    found_ids = set()
    for log in row.get("retrieval_logs", []) or []:
        metadata = log.get("metadata", {}) or {}
        for chunk_id in metadata.get("chunk_ids", []) or []:
            found_ids.add(str(chunk_id))
    read_titles = {chunk_titles[chunk_id] for chunk_id in read_ids if chunk_id in chunk_titles}
    found_titles = {chunk_titles[chunk_id] for chunk_id in found_ids if chunk_id in chunk_titles}
    return read_titles, found_titles


def passage_entity_coverage(item: dict, titles: set[str]) -> float:
    if not titles:
        return 0.0
    text = " ".join(
        sentence
        for title, sentences in item.get("context", [])
        if norm(title) in titles
        for sentence in sentences
    )
    text_norm = norm(text)
    entities = evidence_entities(item)
    if not entities:
        return 0.0
    return sum(1 for entity in entities if entity in text_norm) / len(entities)


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = []
    for value, group in sorted(groups.items(), key=lambda item: item[0]):
        n = len(group)
        out.append({
            key: value,
            "n": n,
            "em": round(sum(float(row["em"]) for row in group) / n, 4),
            "f1": round(sum(float(row["f1"]) for row in group) / n, 4),
            "support_title_recall": round(sum(row["support_title_recall"] for row in group) / n, 4),
            "evidence_entity_coverage": round(sum(row["evidence_entity_coverage"] for row in group) / n, 4),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("comagraag/data/2wiki_sample.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl"),
    )
    parser.add_argument(
        "--arag-predictions",
        type=Path,
        default=Path("results/wise/2wiki_arag_full_post_predictions.jsonl"),
    )
    parser.add_argument(
        "--arag-chunks",
        type=Path,
        default=Path("external_runs/2wiki500/arag/chunks.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/2wiki_support"))
    args = parser.parse_args()

    items = load_json_items(args.data)
    item_by_id = {str(item.get("_id") or item.get("id")): item for item in items}
    predictions = load_jsonl(args.predictions)
    arag_by_id = {str(row.get("qid") or row.get("_id") or row.get("id")): row for row in load_optional_jsonl(args.arag_predictions)}
    chunk_titles = load_chunk_titles(args.arag_chunks)

    rows = []
    for pred in predictions:
        qid = str(pred.get("_id") or pred.get("id"))
        item = item_by_id[qid]
        gold_titles = support_titles(item)
        naive_titles = bm25_top_titles(item, top_k=5)
        full_titles = all_context_titles(item)
        arag_read_titles, arag_found_titles = arag_titles(arag_by_id.get(qid), chunk_titles)
        chain_complete = bool_value(pred.get("evidence_first_chain_complete"))
        gap_type = str(pred.get("evidence_first_gap_type") or "complete")
        repair_steps = str(pred.get("evidence_first_repair_steps") or "")

        rows.append({
            "qid": qid,
            "type": item.get("type", ""),
            "question": item.get("question", ""),
            "gold": item.get("answer", ""),
            "prediction": pred.get("prediction", pred.get("answer", "")),
            "em": float(pred.get("em", 0) or 0),
            "f1": float(pred.get("f1", 0) or 0),
            "chain_complete": chain_complete,
            "gap_type": gap_type,
            "repair_steps": repair_steps,
            "gold_support_titles": " | ".join(sorted(gold_titles)),
            "naive_bm25_top5_titles": " | ".join(sorted(naive_titles)),
            "arag_read_titles": " | ".join(sorted(arag_read_titles)),
            "arag_search_found_titles": " | ".join(sorted(arag_found_titles)),
            "reader_full_titles": " | ".join(sorted(full_titles)),
            "naive_support_title_recall": len(gold_titles & naive_titles) / len(gold_titles) if gold_titles else 0.0,
            "arag_read_support_title_recall": len(gold_titles & arag_read_titles) / len(gold_titles) if gold_titles else 0.0,
            "arag_search_found_support_title_recall": len(gold_titles & arag_found_titles) / len(gold_titles) if gold_titles else 0.0,
            "reader_full_support_title_recall": len(gold_titles & full_titles) / len(gold_titles) if gold_titles else 0.0,
            "support_title_recall": len(gold_titles & full_titles) / len(gold_titles) if gold_titles else 0.0,
            "naive_evidence_entity_coverage": passage_entity_coverage(item, naive_titles),
            "arag_read_evidence_entity_coverage": passage_entity_coverage(item, arag_read_titles),
            "arag_search_found_evidence_entity_coverage": passage_entity_coverage(item, arag_found_titles),
            "reader_full_evidence_entity_coverage": passage_entity_coverage(item, full_titles),
            "evidence_entity_coverage": passage_entity_coverage(item, full_titles),
        })

    summary = {
        "n": len(rows),
        "evidencefirst_reader_full": {
            "em": round(sum(row["em"] for row in rows) / len(rows), 4),
            "f1": round(sum(row["f1"] for row in rows) / len(rows), 4),
            "support_title_recall": round(sum(row["reader_full_support_title_recall"] for row in rows) / len(rows), 4),
            "evidence_entity_coverage": round(sum(row["reader_full_evidence_entity_coverage"] for row in rows) / len(rows), 4),
        },
        "naive_bm25_top5_input": {
            "support_title_recall": round(sum(row["naive_support_title_recall"] for row in rows) / len(rows), 4),
            "evidence_entity_coverage": round(sum(row["naive_evidence_entity_coverage"] for row in rows) / len(rows), 4),
        },
        "arag_actual_read_input": {
            "support_title_recall": round(sum(row["arag_read_support_title_recall"] for row in rows) / len(rows), 4),
            "evidence_entity_coverage": round(sum(row["arag_read_evidence_entity_coverage"] for row in rows) / len(rows), 4),
        },
        "arag_keyword_search_found_upper_bound": {
            "support_title_recall": round(sum(row["arag_search_found_support_title_recall"] for row in rows) / len(rows), 4),
            "evidence_entity_coverage": round(sum(row["arag_search_found_evidence_entity_coverage"] for row in rows) / len(rows), 4),
        },
        "by_chain_complete": summarize(rows, "chain_complete"),
        "by_gap_type": summarize(rows, "gap_type"),
        "by_type": summarize(rows, "type"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "2wiki_support_per_example.csv", rows)
    write_csv(args.out_dir / "2wiki_support_by_chain_complete.csv", summary["by_chain_complete"])
    write_csv(args.out_dir / "2wiki_support_by_gap_type.csv", summary["by_gap_type"])
    write_csv(args.out_dir / "2wiki_support_by_type.csv", summary["by_type"])
    (args.out_dir / "2wiki_support_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
