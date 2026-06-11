#!/usr/bin/env python3
"""Multi-agent proxy adjudication for EvidenceFirst gap labels.

This script is intentionally no-LLM. It simulates three independent semantic
reviewer perspectives from saved 2Wiki artifacts so reviewers can inspect a
bounded 50-100 example audit without requiring external annotators.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CHAIN = Path("results/analysis/2wiki_chain_validity/2wiki_chain_validity_per_example.csv")
DEFAULT_SUPPORT = Path("results/analysis/2wiki_support/2wiki_support_per_example.csv")
DEFAULT_DATA = Path("comagraag/data/2wiki_sample.json")

AGENTS = (
    "strict_chain_reviewer",
    "support_sufficiency_reviewer",
    "conservative_meta_reviewer",
)
CATEGORIES = ("match", "mismatch", "ambiguous")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_json_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported data shape: {path}")


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label(row: dict[str, Any]) -> str:
    gap_type = str(row.get("gap_type") or "complete")
    if gap_type == "complete" and not bool_value(row.get("chain_complete")):
        return "unlabeled_incomplete"
    return gap_type


def desired_quotas(rows: list[dict[str, Any]], sample_size: int) -> dict[str, int]:
    counts = Counter(_label(row) for row in rows)
    labels = ["complete", "missing_entities", "short_chain", "disconnected"]
    if sample_size == 100:
        base = {
            "complete": 34,
            "missing_entities": 35,
            "short_chain": 20,
            "disconnected": 11,
        }
    else:
        weights = {
            "complete": 0.34,
            "missing_entities": 0.35,
            "short_chain": 0.20,
            "disconnected": 0.11,
        }
        base = {label: int(round(sample_size * weights[label])) for label in labels}

    quotas = {label: min(base.get(label, 0), counts.get(label, 0)) for label in labels}
    while sum(quotas.values()) < min(sample_size, len(rows)):
        candidates = [
            label for label in labels
            if quotas.get(label, 0) < counts.get(label, 0)
        ]
        if not candidates:
            break
        label = max(candidates, key=lambda item: counts[item] - quotas[item])
        quotas[label] += 1
    while sum(quotas.values()) > sample_size:
        label = max(quotas, key=lambda item: quotas[item])
        quotas[label] -= 1
    return quotas


def stratified_sample(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[_label(row)].append(row)
    for group in by_label.values():
        group.sort(key=lambda row: str(row.get("qid", "")))
        rng.shuffle(group)

    quotas = desired_quotas(rows, sample_size)
    selected: list[dict[str, Any]] = []
    for label in ["complete", "missing_entities", "short_chain", "disconnected"]:
        selected.extend(by_label.get(label, [])[: quotas.get(label, 0)])
    selected.sort(key=lambda row: (_label(row), str(row.get("qid", ""))))
    return selected[:sample_size]


def support_snippets(item: dict[str, Any], max_snippets: int = 4) -> list[str]:
    facts = item.get("supporting_facts", []) or []
    context = item.get("context", []) or []
    title_to_sents = {norm(title): sentences for title, sentences in context}
    snippets = []
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) < 2:
            continue
        title, sent_idx = fact[0], int(fact[1])
        sentences = title_to_sents.get(norm(title), [])
        if 0 <= sent_idx < len(sentences):
            snippets.append(f"{title}: {sentences[sent_idx]}")
        if len(snippets) >= max_snippets:
            break
    return snippets


def gold_evidence_text(item: dict[str, Any]) -> str:
    evidences = item.get("evidences", []) or []
    parts = []
    for evidence in evidences:
        if len(evidence) >= 3:
            parts.append(f"{evidence[0]} --{evidence[1]}--> {evidence[2]}")
    return " | ".join(parts)


def strict_chain_reviewer(row: dict[str, Any]) -> dict[str, str]:
    label = _label(row)
    entity_coverage = float_value(row.get("gold_entity_coverage"))
    pair_recall = float_value(row.get("gold_pair_recall"))
    relation_recall = float_value(row.get("gold_relation_recall"))
    pair_complete = bool_value(row.get("gold_pair_complete"))
    nodes_connected = bool_value(row.get("gold_evidence_nodes_connected"))

    if label == "complete":
        if pair_complete and nodes_connected:
            return {
                "judgment": "match",
                "rationale": "Gold evidence endpoint pairs are complete and their matched nodes are connected.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "The system marked complete, but the saved KG does not contain a complete connected gold-evidence chain.",
        }
    if label == "missing_entities":
        if entity_coverage < 0.999:
            return {
                "judgment": "match",
                "rationale": "At least one gold evidence entity is missing from the saved KG.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "Gold evidence entities are covered, so a missing-entity label is not supported by the gold proxy.",
        }
    if label == "disconnected":
        if entity_coverage >= 0.999 and not nodes_connected:
            return {
                "judgment": "match",
                "rationale": "Gold evidence entities are present but their matched nodes are not connected.",
            }
        if relation_recall < 0.999 and pair_recall >= 0.999:
            return {
                "judgment": "ambiguous",
                "rationale": "Endpoint pairs exist, but relation semantics are incomplete; this may be a coarse disconnected label.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "The gold proxy does not show present-but-disconnected gold evidence nodes.",
        }
    if label == "short_chain":
        if 0.0 < pair_recall < 0.999 and not pair_complete:
            return {
                "judgment": "match",
                "rationale": "The KG covers only part of the gold evidence pairs.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "The gold proxy does not show a partial gold-evidence chain.",
        }
    return {
        "judgment": "ambiguous",
        "rationale": "The system label is outside the adjudication schema.",
    }


def support_sufficiency_reviewer(row: dict[str, Any], support: dict[str, Any]) -> dict[str, str]:
    label = _label(row)
    support_recall = float_value(support.get("support_title_recall"))
    reader_entity_coverage = float_value(support.get("evidence_entity_coverage"))
    graph_entity_coverage = float_value(row.get("gold_entity_coverage"))
    pair_recall = float_value(row.get("gold_pair_recall"))
    em = float_value(row.get("em"))

    reader_sufficient = support_recall >= 0.999 and reader_entity_coverage >= 0.9
    if label == "complete":
        if reader_sufficient and (em > 0.0 or pair_recall >= 0.999):
            return {
                "judgment": "match",
                "rationale": "Reader-visible support is sufficient and either the answer or gold-pair proxy supports completeness.",
            }
        return {
            "judgment": "ambiguous",
            "rationale": "Reader support is incomplete or answer quality is low, so completeness is not semantically certain.",
        }
    if label == "missing_entities":
        if reader_sufficient and graph_entity_coverage < 0.999:
            return {
                "judgment": "match",
                "rationale": "The full context contains the support, but the saved KG misses at least one gold entity.",
            }
        if not reader_sufficient:
            return {
                "judgment": "ambiguous",
                "rationale": "Reader-visible support is incomplete, so the missing-entity diagnosis cannot be isolated to the KG.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "The saved KG covers gold entities despite the missing-entity label.",
        }
    if label == "disconnected":
        if graph_entity_coverage >= 0.999 and pair_recall < 0.999:
            return {
                "judgment": "match",
                "rationale": "Gold entities are present, but gold evidence pairs are incomplete.",
            }
        return {
            "judgment": "ambiguous",
            "rationale": "The label points to a graph connection problem, but support-level evidence is not decisive.",
        }
    if label == "short_chain":
        if 0.0 < pair_recall < 0.999:
            return {
                "judgment": "match",
                "rationale": "Some gold evidence pairs are present but the chain is incomplete.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "The support proxy does not show a partial chain.",
        }
    return {
        "judgment": "ambiguous",
        "rationale": "The system label is outside the adjudication schema.",
    }


def conservative_meta_reviewer(row: dict[str, Any], support: dict[str, Any]) -> dict[str, str]:
    label = _label(row)
    em = float_value(row.get("em"))
    f1 = float_value(row.get("f1"))
    entity_coverage = float_value(row.get("gold_entity_coverage"))
    pair_recall = float_value(row.get("gold_pair_recall"))
    relation_recall = float_value(row.get("gold_relation_recall"))
    support_recall = float_value(support.get("support_title_recall"))
    pair_complete = bool_value(row.get("gold_pair_complete"))
    nodes_connected = bool_value(row.get("gold_evidence_nodes_connected"))

    answer_acceptable = em >= 1.0 or f1 >= 0.8
    if label == "complete":
        if answer_acceptable and support_recall >= 0.999 and (pair_complete or nodes_connected):
            return {
                "judgment": "match",
                "rationale": "Answer and support visibility are strong enough to accept the complete label as reviewer-facing.",
            }
        if answer_acceptable and support_recall >= 0.999:
            return {
                "judgment": "ambiguous",
                "rationale": "The answer is good, but the KG gold-chain proxy is incomplete.",
            }
        return {
            "judgment": "mismatch",
            "rationale": "A complete label is too strong when answer/support evidence is weak.",
        }
    if label == "missing_entities":
        if entity_coverage < 0.999 and pair_recall < 0.999:
            return {
                "judgment": "match",
                "rationale": "Missing gold entities coincide with incomplete gold-pair coverage.",
            }
        return {
            "judgment": "ambiguous",
            "rationale": "The label may reflect system-query entities rather than gold evidence entities.",
        }
    if label == "disconnected":
        if entity_coverage >= 0.999 and (not nodes_connected or relation_recall < 0.999):
            return {
                "judgment": "match",
                "rationale": "All gold entities are present but graph connectivity or relation semantics remain incomplete.",
            }
        return {
            "judgment": "ambiguous",
            "rationale": "The disconnected label is plausible but not cleanly confirmed by the gold proxy.",
        }
    if label == "short_chain":
        if pair_recall > 0.0 and pair_recall < 0.999:
            return {
                "judgment": "match",
                "rationale": "Partial gold-pair coverage supports a short-chain diagnosis.",
            }
        return {
            "judgment": "ambiguous",
            "rationale": "The chain-length signal is structural, but the gold proxy does not cleanly isolate a short chain.",
        }
    return {
        "judgment": "ambiguous",
        "rationale": "The system label is outside the adjudication schema.",
    }


def adjudicate(row: dict[str, Any], support: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    annotations = [
        {"agent": "strict_chain_reviewer", **strict_chain_reviewer(row)},
        {"agent": "support_sufficiency_reviewer", **support_sufficiency_reviewer(row, support)},
        {"agent": "conservative_meta_reviewer", **conservative_meta_reviewer(row, support)},
    ]
    counts = Counter(annotation["judgment"] for annotation in annotations)
    if counts["match"] >= 2:
        final = "match"
    elif counts["mismatch"] >= 2:
        final = "mismatch"
    else:
        final = "ambiguous"
    confidence = "high" if counts[final] == 3 else "medium" if counts[final] == 2 else "low"
    disagreement = len({annotation["judgment"] for annotation in annotations}) > 1
    return annotations, {
        "majority_adjudication": final,
        "match_votes": counts["match"],
        "mismatch_votes": counts["mismatch"],
        "ambiguous_votes": counts["ambiguous"],
        "adjudicated_correct": final == "match",
        "adjudication_confidence": confidence,
        "primary_disagreement": disagreement,
    }


def pairwise_agreement(items: list[list[str]]) -> dict[str, float]:
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [
        "strict_vs_support",
        "strict_vs_conservative",
        "support_vs_conservative",
    ]
    out = {}
    for name, (left, right) in zip(names, pairs):
        if not items:
            out[name] = 0.0
            continue
        out[name] = round(sum(row[left] == row[right] for row in items) / len(items), 4)
    return out


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * ((p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def fleiss_kappa(items: list[list[str]], categories: tuple[str, ...] = CATEGORIES) -> float:
    if not items:
        return 0.0
    n_items = len(items)
    n_raters = len(items[0])
    if n_raters <= 1:
        return 0.0
    category_counts = []
    for labels in items:
        counts = Counter(labels)
        category_counts.append([counts[category] for category in categories])
    p_i = [
        (sum(count * count for count in counts) - n_raters) / (n_raters * (n_raters - 1))
        for counts in category_counts
    ]
    p_bar = sum(p_i) / n_items
    p_j = [
        sum(counts[idx] for counts in category_counts) / (n_items * n_raters)
        for idx, _category in enumerate(categories)
    ]
    p_e = sum(value * value for value in p_j)
    if p_e >= 1.0:
        return 1.0
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def build_outputs(
    chain_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    data_items: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    support_by_id = {str(row["qid"]): row for row in support_rows}
    item_by_id = {str(item.get("_id") or item.get("id")): item for item in data_items}
    sampled = stratified_sample(chain_rows, sample_size=sample_size, seed=seed)

    csv_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    label_lists: list[list[str]] = []

    for idx, row in enumerate(sampled, start=1):
        qid = str(row["qid"])
        support = support_by_id[qid]
        item = item_by_id[qid]
        annotations, final = adjudicate(row, support)
        labels = [annotation["judgment"] for annotation in annotations]
        label_lists.append(labels)
        rationales = [
            f"{annotation['agent']}={annotation['judgment']}: {annotation['rationale']}"
            for annotation in annotations
        ]
        base = {
            "dataset": "2wiki",
            "sample_index": idx,
            "qid": qid,
            "type": row.get("type", ""),
            "question": row.get("question", ""),
            "gold": row.get("gold", ""),
            "prediction": row.get("prediction", ""),
            "em": row.get("em", ""),
            "f1": row.get("f1", ""),
            "system_gap_type": _label(row),
            "system_chain_complete": row.get("chain_complete", ""),
            "gold_entity_coverage": row.get("gold_entity_coverage", ""),
            "gold_pair_recall": row.get("gold_pair_recall", ""),
            "gold_relation_recall": row.get("gold_relation_recall", ""),
            "gold_pair_complete": row.get("gold_pair_complete", ""),
            "gold_nodes_connected": row.get("gold_evidence_nodes_connected", ""),
            "reader_support_title_recall": support.get("support_title_recall", ""),
            "reader_evidence_entity_coverage": support.get("evidence_entity_coverage", ""),
            "matched_gold_entities": row.get("matched_gold_entities", ""),
            "missing_gold_entities": row.get("missing_gold_entities", ""),
            "gold_evidences": gold_evidence_text(item),
            "strict_chain_judge": labels[0],
            "support_sufficiency_judge": labels[1],
            "conservative_meta_judge": labels[2],
            **final,
            "rationale": " || ".join(rationales),
        }
        csv_rows.append(base)
        jsonl_rows.append({
            **base,
            "judge_annotations": annotations,
            "support_fact_snippets": support_snippets(item),
            "annotation_protocol": {
                "strict_chain_reviewer": "Checks whether saved KG gold proxies support the system gap label.",
                "support_sufficiency_reviewer": "Checks reader-visible support and whether the gap is isolated to graph state.",
                "conservative_meta_reviewer": "Uses answer quality plus support and graph proxies to mark reviewer-facing plausibility.",
            },
        })

    by_gap: dict[str, dict[str, Any]] = {}
    for label, group in _group_by(csv_rows, "system_gap_type").items():
        outcomes = Counter(str(row["majority_adjudication"]) for row in group)
        by_gap[label] = {
            "n": len(group),
            "match_n": outcomes["match"],
            "mismatch_n": outcomes["mismatch"],
            "ambiguous_n": outcomes["ambiguous"],
            "match_rate": round(outcomes["match"] / len(group), 4) if group else 0.0,
            "match_wilson_ci95": wilson_ci(outcomes["match"], len(group)),
            "disagreement_rate": round(
                sum(bool_value(row["primary_disagreement"]) for row in group) / len(group),
                4,
            ) if group else 0.0,
        }

    outcomes = Counter(str(row["majority_adjudication"]) for row in csv_rows)
    summary = {
        "audit_type": "three_deterministic_proxy_view_gap_label_adjudication",
        "dataset": "2wiki",
        "sample_n": len(csv_rows),
        "seed": seed,
        "requested_sample_size": sample_size,
        "sample_quotas": Counter(str(row["system_gap_type"]) for row in csv_rows),
        "overall": {
            "match_n": outcomes["match"],
            "mismatch_n": outcomes["mismatch"],
            "ambiguous_n": outcomes["ambiguous"],
            "match_rate": round(outcomes["match"] / len(csv_rows), 4) if csv_rows else 0.0,
            "match_wilson_ci95": wilson_ci(outcomes["match"], len(csv_rows)),
            "mismatch_rate": round(outcomes["mismatch"] / len(csv_rows), 4) if csv_rows else 0.0,
            "ambiguous_rate": round(outcomes["ambiguous"] / len(csv_rows), 4) if csv_rows else 0.0,
            "disagreement_rate": round(
                sum(bool_value(row["primary_disagreement"]) for row in csv_rows) / len(csv_rows),
                4,
            ) if csv_rows else 0.0,
        },
        "by_gap_type": by_gap,
        "inter_agent_agreement": {
            **pairwise_agreement(label_lists),
            "fleiss_kappa": fleiss_kappa(label_lists),
        },
        "agents": list(AGENTS),
        "caveat": (
            "This is a no-LLM audit with three deterministic proxy views over saved "
            "2Wiki artifacts, not a professional human annotation study. It is suitable "
            "as reviewer-facing supplemental evidence for diagnostic-label plausibility, "
            "not as a gold standard."
        ),
    }
    return csv_rows, jsonl_rows, summary


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/semantic_gap_adjudication"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    for path in [args.chain, args.support, args.data]:
        if not path.exists():
            raise SystemExit(
                f"Missing required artifact: {path}. Run scripts/analyze_2wiki_evidence_support.py "
                "and scripts/analyze_2wiki_chain_validity.py first."
            )

    csv_rows, jsonl_rows, summary = build_outputs(
        chain_rows=load_csv(args.chain),
        support_rows=load_csv(args.support),
        data_items=load_json_items(args.data),
        sample_size=args.sample_size,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "semantic_gap_adjudication_2wiki100.csv", csv_rows)
    write_jsonl(args.out_dir / "semantic_gap_adjudication_2wiki100.jsonl", jsonl_rows)
    (args.out_dir / "semantic_gap_adjudication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
