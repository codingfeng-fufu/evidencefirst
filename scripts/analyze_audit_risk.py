#!/usr/bin/env python3
"""Audit-risk stratification from saved EvidenceFirst artifacts.

The risk score is a deterministic, no-LLM summary of already saved audit fields.
It is not trained on gold answers. It is intended to show whether the structural
record separates low-quality and high-quality answer strata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_RUNS = {
    "hotpot": Path(
        "experiments/evidence_first/results/"
        "hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl"
    ),
    "2wiki": Path("results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl"),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def selected(row: dict) -> bool:
    return (
        bool_value(row.get("evidence_first_postprocess_selected"))
        or bool_value(row.get("evidence_first_postprocess_selected_v6"))
    )


def audit_state(row: dict) -> str:
    complete = bool_value(row.get("evidence_first_chain_complete"))
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


def gap_type(row: dict) -> str:
    return str(row.get("evidence_first_gap_type") or "complete")


def risk_components(row: dict) -> dict[str, bool]:
    complete = bool_value(row.get("evidence_first_chain_complete"))
    gap = gap_type(row)
    state = audit_state(row)
    return {
        "answer_not_selected": not selected(row),
        "chain_incomplete": not complete,
        "residual_graph_gap": gap in {"short_chain", "disconnected", "empty_evidence"},
        "unrepaired_incomplete": state == "unrepaired_incomplete",
    }


def risk_score(row: dict) -> int:
    components = risk_components(row)
    return (
        2 * int(components["answer_not_selected"])
        + int(components["chain_incomplete"])
        + int(components["residual_graph_gap"])
        + int(components["unrepaired_incomplete"])
    )


def graph_audit_score(row: dict) -> int:
    components = risk_components(row)
    return (
        int(components["chain_incomplete"])
        + int(components["residual_graph_gap"])
        + int(components["unrepaired_incomplete"])
    )


def risk_bin(score: int) -> str:
    return "4+" if score >= 4 else str(score)


def summarize_group(dataset: str, label: str, rows: list[dict]) -> dict:
    n = len(rows)
    em_values = [float(row.get("em") or 0) for row in rows]
    f1_values = [float(row.get("f1") or 0) for row in rows]
    return {
        "dataset": dataset,
        "risk_bin": label,
        "n": n,
        "rate": f"{n / sum(1 for _ in rows):.4f}" if False else "",
        "em": f"{mean(em_values):.4f}",
        "f1": f"{mean(f1_values):.4f}",
        "error_rate": f"{mean([1.0 - value for value in em_values]):.4f}",
        "selected_rate": f"{mean([float(selected(row)) for row in rows]):.4f}",
        "chain_complete_rate": f"{mean([float(bool_value(row.get('evidence_first_chain_complete'))) for row in rows]):.4f}",
        "unrepaired_incomplete_rate": f"{mean([float(audit_state(row) == 'unrepaired_incomplete') for row in rows]):.4f}",
    }


def build_rows(dataset: str, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    detailed = []
    for row in rows:
        components = risk_components(row)
        score = risk_score(row)
        detailed.append({
            "dataset": dataset,
            "qid": row.get("_id") or row.get("id") or row.get("qid"),
            "type": row.get("type", ""),
            "em": float(row.get("em") or 0),
            "f1": float(row.get("f1") or 0),
            "risk_score": score,
            "graph_audit_score": graph_audit_score(row),
            "risk_bin": risk_bin(score),
            "audit_state": audit_state(row),
            "gap_type": gap_type(row),
            "answer_selected": selected(row),
            **components,
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detailed:
        grouped[row["risk_bin"]].append(row)

    summary = []
    total = len(detailed)
    for label in ["0", "1", "2", "3", "4+"]:
        group = grouped.get(label, [])
        if not group:
            continue
        em_values = [row["em"] for row in group]
        f1_values = [row["f1"] for row in group]
        summary.append({
            "dataset": dataset,
            "risk_bin": label,
            "n": len(group),
            "rate": f"{len(group) / total:.4f}",
            "em": f"{mean(em_values):.4f}",
            "f1": f"{mean(f1_values):.4f}",
            "error_rate": f"{mean([1.0 - value for value in em_values]):.4f}",
            "selected_rate": f"{mean([float(row['answer_selected']) for row in group]):.4f}",
            "chain_complete_rate": f"{mean([float(not row['chain_incomplete']) for row in group]):.4f}",
            "unrepaired_incomplete_rate": f"{mean([float(row['unrepaired_incomplete']) for row in group]):.4f}",
        })
    return detailed, summary


TRIAGE_SIGNALS = [
    "audit_risk_score",
    "graph_audit_score",
    "answer_not_selected",
    "chain_incomplete",
    "residual_graph_gap",
    "unrepaired_incomplete",
]


def triage_signal_value(row: dict, signal: str) -> float:
    if signal == "audit_risk_score":
        return float(row["risk_score"])
    if signal == "graph_audit_score":
        return float(row["graph_audit_score"])
    if signal == "answer_not_selected":
        return float(row["answer_not_selected"])
    if signal == "chain_incomplete":
        return float(row["chain_incomplete"])
    if signal == "residual_graph_gap":
        return float(row["residual_graph_gap"])
    if signal == "unrepaired_incomplete":
        return float(row["unrepaired_incomplete"])
    raise ValueError(f"Unknown triage signal: {signal}")


def error_auc(rows: list[dict], signal: str) -> float:
    positives = [row for row in rows if float(row["em"]) < 1.0]
    negatives = [row for row in rows if float(row["em"]) >= 1.0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        positive_score = triage_signal_value(positive, signal)
        for negative in negatives:
            negative_score = triage_signal_value(negative, signal)
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / total


def triage_metrics(dataset: str, rows: list[dict], top_fraction: float = 0.2) -> list[dict]:
    total = len(rows)
    if total == 0:
        return []
    top_n = max(1, math.ceil(total * top_fraction))
    overall_error_rate = mean([1.0 - float(row["em"]) for row in rows])
    output = []
    for signal in TRIAGE_SIGNALS:
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                triage_signal_value(row, signal),
                str(row.get("qid") or ""),
            ),
            reverse=True,
        )
        top_rows = sorted_rows[:top_n]
        top_error_rate = mean([1.0 - float(row["em"]) for row in top_rows])
        output.append({
            "dataset": dataset,
            "signal": signal,
            "n": total,
            "overall_error_rate": f"{overall_error_rate:.4f}",
            "error_auc": f"{error_auc(rows, signal):.4f}",
            "topk_fraction": f"{top_fraction:.2f}",
            "topk_n": top_n,
            "topk_error_rate": f"{top_error_rate:.4f}",
            "topk_lift": f"{(top_error_rate / overall_error_rate):.4f}" if overall_error_rate else "0.0000",
            "topk_error_n": int(sum(1 for row in top_rows if float(row["em"]) < 1.0)),
        })
    return output


def sample_size_stability(
    dataset: str,
    rows: list[dict],
    sample_sizes: tuple[int, ...] = (250, 500, 1000),
    top_fraction: float = 0.2,
) -> list[dict]:
    output = []
    for sample_size in sample_sizes:
        if len(rows) < sample_size:
            continue
        sample = rows[:sample_size]
        top_n = max(1, math.ceil(sample_size * top_fraction))
        overall_error_rate = mean([1.0 - float(row["em"]) for row in sample])
        sorted_rows = sorted(
            sample,
            key=lambda row: (
                triage_signal_value(row, "audit_risk_score"),
                str(row.get("qid") or ""),
            ),
            reverse=True,
        )
        top_rows = sorted_rows[:top_n]
        top_error_rate = mean([1.0 - float(row["em"]) for row in top_rows])
        output.append({
            "dataset": dataset,
            "n": sample_size,
            "signal": "audit_risk_score",
            "error_auc": f"{error_auc(sample, 'audit_risk_score'):.4f}",
            "topk_fraction": f"{top_fraction:.2f}",
            "topk_n": top_n,
            "topk_error_rate": f"{top_error_rate:.4f}",
            "topk_lift": f"{(top_error_rate / overall_error_rate):.4f}" if overall_error_rate else "0.0000",
        })
    return output


def conditional_selection_graph_state(dataset: str, rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[bool, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(bool(row["answer_selected"]), int(row["graph_audit_score"]))].append(row)

    output = []
    for answer_selected in [False, True]:
        for graph_score in [0, 1, 2, 3]:
            group = grouped.get((answer_selected, graph_score), [])
            if not group:
                continue
            em = mean([float(row["em"]) for row in group])
            f1 = mean([float(row["f1"]) for row in group])
            output.append({
                "dataset": dataset,
                "answer_selected": str(answer_selected).lower(),
                "graph_score": graph_score,
                "n": len(group),
                "em": f"{em:.4f}",
                "error_rate": f"{1.0 - em:.4f}",
                "f1": f"{f1:.4f}",
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(path: Path, summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = ["hotpot", "2wiki"]
    labels = ["0", "1", "2", "3", "4+"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, dataset in zip(axes, datasets):
        by_label = {
            row["risk_bin"]: row
            for row in summary
            if row["dataset"] == dataset
        }
        values = [float(by_label[label]["em"]) if label in by_label else 0.0 for label in labels]
        counts = [int(by_label[label]["n"]) if label in by_label else 0 for label in labels]
        bars = ax.bar(labels, values, color="#4C78A8")
        ax.set_title("HotpotQA" if dataset == "hotpot" else "2Wiki")
        ax.set_xlabel("Audit-risk bin")
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                str(count),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[0].set_ylabel("Exact match")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", type=Path, default=DEFAULT_RUNS["hotpot"])
    parser.add_argument("--2wiki", dest="twowiki", type=Path, default=DEFAULT_RUNS["2wiki"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/audit_risk"))
    parser.add_argument("--figure", type=Path, default=Path("paper/figures/audit_risk_strata.pdf"))
    args = parser.parse_args()

    all_detailed = []
    all_summary = []
    all_triage = []
    all_conditional = []
    all_stability = []
    for dataset, path in [("hotpot", args.hotpot), ("2wiki", args.twowiki)]:
        detailed, summary = build_rows(dataset, load_jsonl(path))
        all_detailed.extend(detailed)
        all_summary.extend(summary)
        all_triage.extend(triage_metrics(dataset, detailed))
        all_conditional.extend(conditional_selection_graph_state(dataset, detailed))
        all_stability.extend(sample_size_stability(dataset, detailed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "audit_risk_per_example.csv", all_detailed)
    write_csv(args.out_dir / "audit_risk_summary.csv", all_summary)
    write_csv(args.out_dir / "audit_triage_utility.csv", all_triage)
    write_csv(args.out_dir / "conditional_selection_graph_state.csv", all_conditional)
    write_csv(args.out_dir / "sample_size_stability.csv", all_stability)
    (args.out_dir / "audit_risk_summary.json").write_text(
        json.dumps({
            "summary": all_summary,
            "triage_utility": all_triage,
            "conditional_selection_graph_state": all_conditional,
            "sample_size_stability": all_stability,
        }, indent=2),
        encoding="utf-8",
    )
    write_figure(args.figure, all_summary)
    print(json.dumps({
        "summary": all_summary,
        "triage_utility": all_triage,
        "sample_size_stability": all_stability,
        "figure": str(args.figure),
    }, indent=2))


if __name__ == "__main__":
    main()
