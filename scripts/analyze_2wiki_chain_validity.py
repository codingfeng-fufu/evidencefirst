#!/usr/bin/env python3
"""Posthoc 2Wiki gold-evidence alignment for EvidenceFirst diagnostics.

This analysis uses gold 2Wiki evidences only after prediction. It is not an
inference-time oracle. The goal is to check whether saved question KGs contain
gold evidence entities, gold evidence endpoint pairs, and connected gold
evidence nodes, then relate those signals to EvidenceFirst diagnostic labels.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


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


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_undirected_graph(graph: nx.Graph) -> nx.Graph:
    out = nx.Graph()
    for node in graph.nodes():
        out.add_node(node)
    for u, v, data in graph.edges(data=True):
        relations = data.get("relations")
        if relations is None:
            relation = data.get("relation", "related_to")
            relations = [relation]
        if out.has_edge(u, v):
            out[u][v].setdefault("relations", []).extend(relations)
        else:
            out.add_edge(u, v, relations=list(relations))
    return out


def find_node(entity: str, nodes: list[str]) -> str | None:
    target = norm(entity)
    if not target:
        return None

    normalized = [(node, norm(node)) for node in nodes]
    for node, node_norm in normalized:
        if node_norm == target:
            return node

    for node, node_norm in normalized:
        if not node_norm:
            continue
        if target in node_norm or node_norm in target:
            longer = max(len(target), len(node_norm))
            if abs(len(target) - len(node_norm)) <= longer * 0.3:
                return node

    target_tokens = set(target.split())
    best_node = None
    best_score = 0.0
    for node, node_norm in normalized:
        node_tokens = set(node_norm.split())
        if not target_tokens or not node_tokens:
            continue
        score = len(target_tokens & node_tokens) / len(target_tokens | node_tokens)
        if score >= 0.85 and score > best_score:
            best_score = score
            best_node = node
    return best_node


def relation_matches(gold_relation: str, graph_relations: list[str]) -> bool:
    gold_tokens = set(norm(gold_relation).split())
    if not gold_tokens:
        return False
    for relation in graph_relations:
        rel_tokens = set(norm(relation).split())
        if not rel_tokens:
            continue
        if gold_tokens <= rel_tokens or rel_tokens <= gold_tokens:
            return True
        overlap = len(gold_tokens & rel_tokens) / len(gold_tokens | rel_tokens)
        if overlap >= 0.5:
            return True
    return False


def gold_entities(item: dict) -> set[str]:
    entities = set()
    for triple in item.get("evidences", []) or []:
        if len(triple) >= 3:
            entities.add(str(triple[0]))
            entities.add(str(triple[2]))
    if item.get("answer"):
        entities.add(str(item["answer"]))
    return {entity for entity in entities if norm(entity)}


def graph_gold_metrics(item: dict, graph: nx.Graph) -> dict:
    graph = to_undirected_graph(graph)
    nodes = list(graph.nodes())
    entity_map = {entity: find_node(entity, nodes) for entity in gold_entities(item)}
    covered_entities = [entity for entity, node in entity_map.items() if node is not None]
    entity_count = len(entity_map)

    evidence_triples = [triple for triple in item.get("evidences", []) or [] if len(triple) >= 3]
    pair_hits = 0
    relation_hits = 0
    matched_nodes = set()
    for head, relation, tail, *_rest in evidence_triples:
        h_node = entity_map.get(str(head)) or find_node(str(head), nodes)
        t_node = entity_map.get(str(tail)) or find_node(str(tail), nodes)
        if h_node:
            matched_nodes.add(h_node)
        if t_node:
            matched_nodes.add(t_node)
        if h_node and t_node and graph.has_edge(h_node, t_node):
            pair_hits += 1
            relations = graph[h_node][t_node].get("relations", [])
            if relation_matches(str(relation), relations):
                relation_hits += 1

    pair_count = len(evidence_triples)
    gold_nodes_connected = False
    if matched_nodes and len(matched_nodes) == entity_count:
        try:
            iterator = iter(matched_nodes)
            first = next(iterator)
            gold_nodes_connected = all(nx.has_path(graph, first, node) for node in iterator)
        except (StopIteration, nx.NetworkXException):
            gold_nodes_connected = False

    entity_coverage = len(covered_entities) / entity_count if entity_count else 0.0
    pair_recall = pair_hits / pair_count if pair_count else 0.0
    relation_recall = relation_hits / pair_count if pair_count else 0.0
    return {
        "gold_entity_count": entity_count,
        "gold_entity_coverage": entity_coverage,
        "gold_pair_count": pair_count,
        "gold_pair_recall": pair_recall,
        "gold_relation_recall": relation_recall,
        "gold_pair_complete": pair_count > 0 and pair_hits == pair_count,
        "gold_relation_complete": pair_count > 0 and relation_hits == pair_count,
        "gold_evidence_nodes_connected": gold_nodes_connected,
        "matched_gold_entities": " | ".join(sorted(covered_entities)),
        "missing_gold_entities": " | ".join(sorted(entity for entity, node in entity_map.items() if node is None)),
    }


def load_cache(cache_paths: list[str]) -> dict[str, dict]:
    merged = {}
    for pattern in cache_paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            for qid, value in data.items():
                merged[str(qid)] = value
    return merged


def repair_metrics(cache_row: dict | None) -> dict:
    history = (cache_row or {}).get("history", []) or []
    repair_steps = [
        row for row in history
        if str(row.get("step", "")) in {"A+_enhancement", "B_repair"}
    ]
    rechecks = [
        row for row in history
        if str(row.get("step", "")).startswith("recheck_after_")
    ]
    attempted = bool(repair_steps)
    triples_added = sum(int((row.get("stats") or {}).get("triples_added", 0) or 0) for row in repair_steps)
    triples_extracted = sum(int((row.get("stats") or {}).get("triples_extracted", 0) or 0) for row in repair_steps)
    llm_calls = sum(int((row.get("stats") or {}).get("llm_calls", 0) or 0) for row in repair_steps)
    repair_to_complete = any(bool_value(row.get("complete")) for row in rechecks)
    if not attempted:
        outcome = "not_attempted"
    elif repair_to_complete:
        outcome = "repair_to_complete"
    else:
        outcome = "repair_residual_gap"
    return {
        "repair_attempted": attempted,
        "repair_outcome": outcome,
        "repair_triples_extracted": triples_extracted,
        "repair_triples_added": triples_added,
        "repair_llm_calls": llm_calls,
    }


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
            "gold_entity_coverage": round(sum(float(row["gold_entity_coverage"]) for row in group) / n, 4),
            "gold_pair_recall": round(sum(float(row["gold_pair_recall"]) for row in group) / n, 4),
            "gold_relation_recall": round(sum(float(row["gold_relation_recall"]) for row in group) / n, 4),
            "gold_pair_complete_rate": round(sum(bool_value(row["gold_pair_complete"]) for row in group) / n, 4),
            "gold_nodes_connected_rate": round(sum(bool_value(row["gold_evidence_nodes_connected"]) for row in group) / n, 4),
            "repair_attempt_rate": round(sum(bool_value(row["repair_attempted"]) for row in group) / n, 4),
            "repair_to_complete_rate": round(
                sum(1 for row in group if row["repair_outcome"] == "repair_to_complete")
                / max(1, sum(bool_value(row["repair_attempted"]) for row in group)),
                4,
            ),
        })
    return out


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gold_proxy_target(row: dict, gap_type: str) -> bool:
    entity_coverage = float_value(row.get("gold_entity_coverage"))
    pair_recall = float_value(row.get("gold_pair_recall"))
    pair_complete = bool_value(row.get("gold_pair_complete"))
    nodes_connected = bool_value(row.get("gold_evidence_nodes_connected"))

    if gap_type == "complete":
        return pair_complete and nodes_connected
    if gap_type == "missing_entities":
        return entity_coverage < 0.999
    if gap_type == "disconnected":
        return entity_coverage >= 0.999 and not nodes_connected
    if gap_type == "short_chain":
        return 0.0 < pair_recall < 0.999 and not pair_complete
    return False


def _gold_proxy_name(gap_type: str) -> str:
    return {
        "complete": "gold_pair_complete_and_nodes_connected",
        "missing_entities": "missing_gold_entity",
        "disconnected": "all_gold_entities_present_but_nodes_disconnected",
        "short_chain": "partial_gold_pair_recall",
    }.get(gap_type, "unmapped_proxy")


def gap_label_gold_audit(rows: list[dict]) -> list[dict]:
    """Gold-derived proxy audit for diagnostic labels.

    These targets are posthoc approximations from gold 2Wiki evidence triples;
    they are not inference-time labels and not human semantic adjudication.
    """
    out = []
    total_n = len(rows)
    for gap_type in ["complete", "missing_entities", "disconnected", "short_chain"]:
        label_rows = [row for row in rows if str(row.get("gap_type") or "complete") == gap_type]
        target_rows = [row for row in rows if _gold_proxy_target(row, gap_type)]
        true_positive = [row for row in label_rows if _gold_proxy_target(row, gap_type)]
        label_n = len(label_rows)
        target_n = len(target_rows)
        tp_n = len(true_positive)
        precision = tp_n / label_n if label_n else 0.0
        recall = tp_n / target_n if target_n else 0.0
        target_prevalence = target_n / total_n if total_n else 0.0
        lift = precision / target_prevalence if target_prevalence else 0.0
        out.append({
            "gap_type": gap_type,
            "target_name": _gold_proxy_name(gap_type),
            "n": total_n,
            "label_n": label_n,
            "target_n": target_n,
            "true_positive_n": tp_n,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "target_prevalence": round(target_prevalence, 4),
            "lift": round(lift, 4),
            "label_em": round(sum(float_value(row.get("em")) for row in label_rows) / label_n, 4) if label_n else 0.0,
            "label_f1": round(sum(float_value(row.get("f1")) for row in label_rows) / label_n, 4) if label_n else 0.0,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("comagraag/data/2wiki_sample.json"))
    parser.add_argument("--kgs", type=Path, default=Path("results/wise/2wiki_evidencefirst_v6_kgs.pkl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl"),
    )
    parser.add_argument(
        "--cache",
        action="append",
        default=[
            "results/cache_wise_2wiki_evidencefirst_v6_readerfull_s100_full.json",
            "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b*_full.json",
        ],
        help="Cache JSON path or glob. Repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/2wiki_chain_validity"))
    args = parser.parse_args()

    items = load_json_items(args.data)
    item_by_id = {str(item.get("_id") or item.get("id")): item for item in items}
    predictions = load_jsonl(args.predictions)
    graphs = pickle.load(args.kgs.open("rb"))
    cache = load_cache(args.cache)

    rows = []
    for pred in predictions:
        qid = str(pred.get("_id") or pred.get("id"))
        item = item_by_id[qid]
        graph = graphs[qid]
        graph_metrics = graph_gold_metrics(item, graph)
        repair = repair_metrics(cache.get(qid))
        chain_complete = bool_value(pred.get("evidence_first_chain_complete"))
        gap_type = str(pred.get("evidence_first_gap_type") or "complete")
        if not chain_complete and gap_type == "complete":
            gap_type = "unlabeled_incomplete"
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
            **graph_metrics,
            **repair,
        })

    overall = summarize(rows, "all") if False else []
    summary = {
        "n": len(rows),
        "overall": {
            "em": round(sum(float(row["em"]) for row in rows) / len(rows), 4),
            "f1": round(sum(float(row["f1"]) for row in rows) / len(rows), 4),
            "gold_entity_coverage": round(sum(row["gold_entity_coverage"] for row in rows) / len(rows), 4),
            "gold_pair_recall": round(sum(row["gold_pair_recall"] for row in rows) / len(rows), 4),
            "gold_relation_recall": round(sum(row["gold_relation_recall"] for row in rows) / len(rows), 4),
            "gold_pair_complete_rate": round(sum(bool_value(row["gold_pair_complete"]) for row in rows) / len(rows), 4),
            "gold_nodes_connected_rate": round(sum(bool_value(row["gold_evidence_nodes_connected"]) for row in rows) / len(rows), 4),
        },
        "by_chain_complete": summarize(rows, "chain_complete"),
        "by_gap_type": summarize(rows, "gap_type"),
        "by_repair_outcome": summarize(rows, "repair_outcome"),
        "by_type": summarize(rows, "type"),
        "gap_label_gold_audit": gap_label_gold_audit(rows),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "2wiki_chain_validity_per_example.csv", rows)
    write_csv(args.out_dir / "2wiki_chain_validity_by_chain_complete.csv", summary["by_chain_complete"])
    write_csv(args.out_dir / "2wiki_chain_validity_by_gap_type.csv", summary["by_gap_type"])
    write_csv(args.out_dir / "2wiki_chain_validity_by_repair_outcome.csv", summary["by_repair_outcome"])
    write_csv(args.out_dir / "2wiki_chain_validity_by_type.csv", summary["by_type"])
    write_csv(args.out_dir / "2wiki_gap_label_gold_audit.csv", summary["gap_label_gold_audit"])
    (args.out_dir / "2wiki_chain_validity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
