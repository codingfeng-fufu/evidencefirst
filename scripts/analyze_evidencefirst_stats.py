#!/usr/bin/env python3
"""Offline statistical and audit-state summaries for EvidenceFirst artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_RUNS = {
    "hotpot": {
        "EvidenceFirst v4 fresh": Path(
            "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v4.jsonl"
        ),
        "EvidenceFirst v6 selector guard": Path(
            "experiments/evidence_first/results/"
            "hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl"
        ),
        "HopRAG strict": Path("results/wise_hotpot1000_hoprag_strict_response_metrics.csv"),
        "A-RAG": Path("results/wise_hotpot1000_arag_full_post_combined_metrics.csv"),
        "IRCoT": Path("results/wise/hotpot1000_ircot_combined_predictions.jsonl"),
        "LightRAG": Path("results/wise/hotpot1000_lightrag_combined_predictions.jsonl"),
        "MS GraphRAG": Path("results/wise/hotpot1000_ms_graphrag_combined_predictions.jsonl"),
        "Naive RAG": Path("results/wise/hotpot1000_naive_rag_combined_predictions.jsonl"),
    },
    "2wiki": {
        "EvidenceFirst v6": Path("results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl"),
        "EvidenceFirst v6 local-context": Path("results/wise/2wiki_evidencefirst_v6_localctx2_predictions.jsonl"),
        "LightRAG": Path("results/wise/2wiki_lightrag_predictions.jsonl"),
        "HopRAG strict": Path("results/wise_2wiki_hoprag_strict_metrics.csv"),
        "A-RAG": Path("results/wise_2wiki_arag_full_post_metrics.csv"),
        "IRCoT": Path("results/wise/2wiki_ircot_predictions.jsonl"),
        "Naive RAG": Path("results/wise/2wiki_naive_rag_predictions.jsonl"),
    },
}

PAIRWISE = [
    ("hotpot", "EvidenceFirst v4 fresh", "HopRAG strict", "all", None),
    ("hotpot", "EvidenceFirst v6 selector guard", "HopRAG strict", "all", None),
    ("hotpot", "EvidenceFirst v6 selector guard", "HopRAG strict", "bridge", "bridge"),
    ("2wiki", "EvidenceFirst v6", "LightRAG", "all", None),
    ("2wiki", "EvidenceFirst v6", "HopRAG strict", "all", None),
    ("2wiki", "EvidenceFirst v6", "A-RAG", "all", None),
    ("2wiki", "EvidenceFirst v6 local-context", "HopRAG strict", "local-context stress", None),
]

AUDIT_RUNS = {
    "hotpot": ("EvidenceFirst v6 selector guard", DEFAULT_RUNS["hotpot"]["EvidenceFirst v6 selector guard"]),
    "2wiki": ("EvidenceFirst v6", DEFAULT_RUNS["2wiki"]["EvidenceFirst v6"]),
}

ABLATION_PAIRWISE = [
    (
        "hotpot",
        "EvidenceFirst v6 selector guard",
        "without_verification",
        Path(
            "experiments/evidence_first/results/"
            "hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl"
        ),
    ),
    (
        "2wiki",
        "EvidenceFirst v6",
        "without_verification",
        Path(
            "experiments/evidence_first/results/"
            "2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl"
        ),
    ),
]


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported file extension: {path}")


def _qid(row: dict, fallback: int) -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def _float(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _positive_count(row: dict, key: str) -> bool:
    try:
        return float(row.get(key) or 0) > 0
    except (TypeError, ValueError):
        return False


def _records(path: Path) -> list[dict]:
    rows = _load_rows(path)
    if not rows:
        return []
    if "qid" not in rows[0] and "_id" not in rows[0] and "id" not in rows[0]:
        raise ValueError(f"{path} is an aggregate file, not a per-example artifact")

    records = []
    for idx, row in enumerate(rows):
        records.append({
            "qid": _qid(row, idx),
            "em": _float(row, "em"),
            "f1": _float(row, "f1"),
            "type": row.get("type", ""),
            "raw": row,
        })
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    samples = []
    for _ in range(iterations):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    lo = samples[int(0.025 * (iterations - 1))]
    hi = samples[int(0.975 * (iterations - 1))]
    return lo, hi


def _paired_bootstrap_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float]:
    return _bootstrap_ci(values, iterations, rng)


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_ci(run_records: dict[str, dict[str, list[dict]]], iterations: int, seed: int) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    for dataset, runs in run_records.items():
        for method, records in runs.items():
            em_values = [record["em"] for record in records]
            f1_values = [record["f1"] for record in records]
            em_lo, em_hi = _bootstrap_ci(em_values, iterations, rng)
            f1_lo, f1_hi = _bootstrap_ci(f1_values, iterations, rng)
            rows.append({
                "dataset": dataset,
                "method": method,
                "n": len(records),
                "em": f"{_mean(em_values):.4f}",
                "em_ci95_low": f"{em_lo:.4f}",
                "em_ci95_high": f"{em_hi:.4f}",
                "f1": f"{_mean(f1_values):.4f}",
                "f1_ci95_low": f"{f1_lo:.4f}",
                "f1_ci95_high": f"{f1_hi:.4f}",
            })
    return rows


def pairwise_stats(
    run_records: dict[str, dict[str, list[dict]]],
    iterations: int,
    seed: int,
) -> list[dict]:
    rows = []
    rng = random.Random(seed + 17)
    for dataset, primary, baseline, subgroup, type_filter in PAIRWISE:
        primary_by_id = {record["qid"]: record for record in run_records[dataset][primary]}
        baseline_by_id = {record["qid"]: record for record in run_records[dataset][baseline]}
        shared = sorted(set(primary_by_id) & set(baseline_by_id))
        if type_filter:
            shared = [
                qid for qid in shared
                if str(primary_by_id[qid].get("type", "")).lower() == type_filter
            ]
        b = c = both = neither = 0
        for qid in shared:
            p_correct = int(primary_by_id[qid]["em"] == 1.0)
            b_correct = int(baseline_by_id[qid]["em"] == 1.0)
            if p_correct and b_correct:
                both += 1
            elif p_correct and not b_correct:
                b += 1
            elif not p_correct and b_correct:
                c += 1
            else:
                neither += 1
        em_diffs = [primary_by_id[qid]["em"] - baseline_by_id[qid]["em"] for qid in shared]
        f1_diffs = [primary_by_id[qid]["f1"] - baseline_by_id[qid]["f1"] for qid in shared]
        p_em = _mean([primary_by_id[qid]["em"] for qid in shared])
        b_em = _mean([baseline_by_id[qid]["em"] for qid in shared])
        p_f1 = _mean([primary_by_id[qid]["f1"] for qid in shared])
        b_f1 = _mean([baseline_by_id[qid]["f1"] for qid in shared])
        em_lo, em_hi = _paired_bootstrap_ci(em_diffs, iterations, rng)
        f1_lo, f1_hi = _paired_bootstrap_ci(f1_diffs, iterations, rng)
        rows.append({
            "dataset": dataset,
            "subgroup": subgroup,
            "primary": primary,
            "baseline": baseline,
            "n": len(shared),
            "primary_em": f"{p_em:.4f}",
            "baseline_em": f"{b_em:.4f}",
            "em_delta": f"{p_em - b_em:.4f}",
            "em_delta_ci95_low": f"{em_lo:.4f}",
            "em_delta_ci95_high": f"{em_hi:.4f}",
            "primary_f1": f"{p_f1:.4f}",
            "baseline_f1": f"{b_f1:.4f}",
            "f1_delta": f"{p_f1 - b_f1:.4f}",
            "f1_delta_ci95_low": f"{f1_lo:.4f}",
            "f1_delta_ci95_high": f"{f1_hi:.4f}",
            "both_correct": both,
            "primary_only": b,
            "baseline_only": c,
            "neither": neither,
            "mcnemar_exact_p": f"{_mcnemar_exact(b, c):.6f}",
        })
    return rows


def ablation_pairwise_stats(
    run_records: dict[str, dict[str, list[dict]]],
    iterations: int,
    seed: int,
) -> list[dict]:
    rows = []
    rng = random.Random(seed + 31)
    for dataset, primary, ablation, path in ABLATION_PAIRWISE:
        primary_by_id = {record["qid"]: record for record in run_records[dataset][primary]}
        ablation_records = _records(path)
        ablation_by_id = {record["qid"]: record for record in ablation_records}
        shared = sorted(set(primary_by_id) & set(ablation_by_id))
        b = c = both = neither = 0
        for qid in shared:
            p_correct = int(primary_by_id[qid]["em"] == 1.0)
            a_correct = int(ablation_by_id[qid]["em"] == 1.0)
            if p_correct and a_correct:
                both += 1
            elif p_correct and not a_correct:
                b += 1
            elif not p_correct and a_correct:
                c += 1
            else:
                neither += 1
        em_diffs = [primary_by_id[qid]["em"] - ablation_by_id[qid]["em"] for qid in shared]
        f1_diffs = [primary_by_id[qid]["f1"] - ablation_by_id[qid]["f1"] for qid in shared]
        p_em = _mean([primary_by_id[qid]["em"] for qid in shared])
        a_em = _mean([ablation_by_id[qid]["em"] for qid in shared])
        p_f1 = _mean([primary_by_id[qid]["f1"] for qid in shared])
        a_f1 = _mean([ablation_by_id[qid]["f1"] for qid in shared])
        em_lo, em_hi = _paired_bootstrap_ci(em_diffs, iterations, rng)
        f1_lo, f1_hi = _paired_bootstrap_ci(f1_diffs, iterations, rng)
        rows.append({
            "dataset": dataset,
            "primary": primary,
            "ablation": ablation,
            "n": len(shared),
            "primary_em": f"{p_em:.4f}",
            "ablation_em": f"{a_em:.4f}",
            "em_delta": f"{p_em - a_em:.4f}",
            "em_delta_ci95_low": f"{em_lo:.4f}",
            "em_delta_ci95_high": f"{em_hi:.4f}",
            "primary_f1": f"{p_f1:.4f}",
            "ablation_f1": f"{a_f1:.4f}",
            "f1_delta": f"{p_f1 - a_f1:.4f}",
            "f1_delta_ci95_low": f"{f1_lo:.4f}",
            "f1_delta_ci95_high": f"{f1_hi:.4f}",
            "both_correct": both,
            "primary_only": b,
            "ablation_only": c,
            "neither": neither,
            "mcnemar_exact_p": f"{_mcnemar_exact(b, c):.6f}",
        })
    return rows


def _audit_state(row: dict) -> str:
    complete = _bool(row.get("evidence_first_chain_complete"))
    repaired = bool(str(row.get("evidence_first_repair_steps") or "").strip())
    fallback = bool(str(row.get("evidence_first_fallback_steps") or "").strip())
    if complete and repaired:
        return "repaired_complete"
    if complete:
        return "verified_complete"
    if fallback:
        return "fallback_incomplete"
    if repaired:
        return "repaired_incomplete"
    return "unrepaired_incomplete"


def _gap_type(row: dict) -> str:
    return str(row.get("evidence_first_gap_type") or "complete")


def _gap_consistency_rule(gap_type: str) -> str:
    if gap_type == "complete":
        return "chain_complete=true"
    if gap_type == "missing_entities":
        return "missing_entity_count>0 or missing_entities non-empty"
    if gap_type == "disconnected":
        return "disconnected_pair_count>0"
    if gap_type == "short_chain":
        return "chain_complete=false and chain_length>0"
    if gap_type == "empty_evidence":
        return "fallback_steps non-empty"
    return "unrecognized label recorded"


def _gap_label_consistent(row: dict, gap_type: str) -> bool:
    if gap_type == "complete":
        return _bool(row.get("evidence_first_chain_complete"))
    if gap_type == "missing_entities":
        return (
            _positive_count(row, "evidence_first_missing_entity_count")
            or bool(str(row.get("evidence_first_missing_entities") or "").strip())
        )
    if gap_type == "disconnected":
        return _positive_count(row, "evidence_first_disconnected_pair_count")
    if gap_type == "short_chain":
        return (
            not _bool(row.get("evidence_first_chain_complete"))
            and _positive_count(row, "evidence_first_chain_length")
        )
    if gap_type == "empty_evidence":
        return bool(str(row.get("evidence_first_fallback_steps") or "").strip())
    return bool(gap_type)


def gap_label_consistency(run_records: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for dataset, (method, _path) in AUDIT_RUNS.items():
        records = run_records[dataset][method]
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            groups[_gap_type(record["raw"])].append(record)

        for gap_type, group in sorted(groups.items()):
            consistent = [
                _gap_label_consistent(record["raw"], gap_type)
                for record in group
            ]
            consistent_n = sum(consistent)
            n = len(group)
            rows.append({
                "dataset": dataset,
                "method": method,
                "gap_type": gap_type,
                "consistency_rule": _gap_consistency_rule(gap_type),
                "n": n,
                "consistent_n": consistent_n,
                "inconsistent_n": n - consistent_n,
                "consistent_rate": f"{consistent_n / n:.4f}" if n else "0.0000",
                "em": f"{_mean([record['em'] for record in group]):.4f}",
                "f1": f"{_mean([record['f1'] for record in group]):.4f}",
            })
    return rows


def _group_summary(dataset: str, group_key: str, records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        raw = record["raw"]
        if group_key == "audit_state":
            value = _audit_state(raw)
        elif group_key == "gap_type":
            value = _gap_type(raw)
        else:
            value = str(raw.get(group_key, ""))
        groups[value].append(record)

    rows = []
    n_total = len(records)
    for value, group in sorted(groups.items()):
        rows.append({
            "dataset": dataset,
            group_key: value,
            "n": len(group),
            "rate": f"{len(group) / n_total:.4f}",
            "em": f"{_mean([record['em'] for record in group]):.4f}",
            "f1": f"{_mean([record['f1'] for record in group]):.4f}",
        })
    return rows


def audit_stats(run_records: dict[str, dict[str, list[dict]]]) -> tuple[list[dict], list[dict], list[dict]]:
    overview_rows = []
    state_rows = []
    gap_rows = []

    for dataset, (method, _path) in AUDIT_RUNS.items():
        records = run_records[dataset][method]
        n = len(records)
        raw_rows = [record["raw"] for record in records]
        complete = [_bool(row.get("evidence_first_chain_complete")) for row in raw_rows]
        repaired = [bool(str(row.get("evidence_first_repair_steps") or "").strip()) for row in raw_rows]
        fallback = [bool(str(row.get("evidence_first_fallback_steps") or "").strip()) for row in raw_rows]
        selected = [
            _bool(row.get("evidence_first_postprocess_selected_v6"))
            if "evidence_first_postprocess_selected_v6" in row and str(row.get("evidence_first_postprocess_selected_v6")) != ""
            else _bool(row.get("evidence_first_postprocess_selected"))
            for row in raw_rows
        ]
        changed = [
            _bool(row.get("posthoc_clean_changed"))
            or _bool(row.get("posthoc_selector_guard_changed"))
            or _bool(row.get("canonicalized"))
            for row in raw_rows
        ]
        repair_attempts = sum(repaired)
        repair_to_complete = sum(1 for is_repaired, is_complete in zip(repaired, complete) if is_repaired and is_complete)

        overview_rows.append({
            "dataset": dataset,
            "method": method,
            "n": n,
            "chain_complete_rate": f"{sum(complete) / n:.4f}",
            "repair_attempt_rate": f"{repair_attempts / n:.4f}",
            "repair_to_complete_rate": f"{repair_to_complete / repair_attempts:.4f}" if repair_attempts else "0.0000",
            "fallback_rate": f"{sum(fallback) / n:.4f}",
            "selector_selected_rate": f"{sum(selected) / n:.4f}",
            "posthoc_changed_rate": f"{sum(changed) / n:.4f}",
        })
        state_rows.extend(_group_summary(dataset, "audit_state", records))
        gap_rows.extend(_group_summary(dataset, "gap_type", records))

    return overview_rows, state_rows, gap_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/evidencefirst_stats"))
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    run_records = {
        dataset: {method: _records(path) for method, path in runs.items()}
        for dataset, runs in DEFAULT_RUNS.items()
    }

    metric_rows = metric_ci(run_records, args.bootstrap_iters, args.seed)
    pairwise_rows = pairwise_stats(run_records, args.bootstrap_iters, args.seed)
    ablation_pairwise_rows = ablation_pairwise_stats(run_records, args.bootstrap_iters, args.seed)
    overview_rows, state_rows, gap_rows = audit_stats(run_records)
    label_rows = gap_label_consistency(run_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "main_metric_ci.csv", metric_rows)
    _write_csv(args.out_dir / "pairwise_mcnemar.csv", pairwise_rows)
    _write_csv(args.out_dir / "ablation_mcnemar.csv", ablation_pairwise_rows)
    _write_csv(args.out_dir / "audit_overview.csv", overview_rows)
    _write_csv(args.out_dir / "audit_by_state.csv", state_rows)
    _write_csv(args.out_dir / "audit_by_gap_type.csv", gap_rows)
    _write_csv(args.out_dir / "audit_label_consistency.csv", label_rows)

    summary = {
        "metric_ci": metric_rows,
        "pairwise_mcnemar": pairwise_rows,
        "ablation_mcnemar": ablation_pairwise_rows,
        "audit_overview": overview_rows,
        "audit_by_state": state_rows,
        "audit_by_gap_type": gap_rows,
        "audit_label_consistency": label_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
