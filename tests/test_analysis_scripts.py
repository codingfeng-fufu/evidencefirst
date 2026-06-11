import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_usage_reports_source_and_end_to_end_cost(tmp_path):
    summarize_usage = _load_script("summarize_usage")
    path = tmp_path / "usage.jsonl"
    rows = [
        {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "wall_time": 0,
            "source_llm_calls": 5,
            "source_input_tokens": 100,
            "source_output_tokens": 20,
            "source_total_tokens": 120,
            "source_wall_time": 3,
        },
        {
            "llm_calls": 1,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "wall_time": 1,
            "source_llm_calls": 4,
            "source_input_tokens": 80,
            "source_output_tokens": 10,
            "source_total_tokens": 90,
            "source_wall_time": 2,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_usage.summarize(path)

    assert summary["avg_llm_calls"] == 0.5
    assert summary["avg_source_llm_calls"] == 4.5
    assert summary["avg_source_total_tokens"] == 105.0
    assert summary["avg_end_to_end_llm_calls"] == 5.0
    assert summary["avg_end_to_end_total_tokens"] == 111.0


def test_verify_paper_artifacts_expected_metrics_are_reviewer_facing():
    verify = _load_script("verify_paper_artifacts")

    assert "docs/experiments/paper_claims_manifest.json" in verify.REQUIRED_FILES
    assert ("hotpot", "EvidenceFirst v4 fresh") in verify.EXPECTED_MAIN
    assert ("2wiki", "EvidenceFirst v6") in verify.EXPECTED_MAIN
    assert "results/wise/2wiki_evidencefirst_v6_kgs.pkl" in verify.REQUIRED_FILES
    assert "results/wise/2wiki_arag_full_post_predictions.jsonl" in verify.REQUIRED_FILES
    assert "external_runs/2wiki500/arag/chunks.json" in verify.REQUIRED_FILES
    assert verify.EXPECTED_ADJUDICATION["sample_n"] == 100
    assert verify.REQUIRED_PATTERNS
    assert "2Wiki EvidenceFirst reader-full" in verify.EXPECTED_PROTOCOL_BOUNDARY
    assert "2Wiki EvidenceFirst local-context stress" in verify.EXPECTED_PROTOCOL_BOUNDARY
    assert verify.FORBIDDEN_2WIKI_INFERENCE_FIELDS == {
        "context",
        "supporting_facts",
        "evidences",
        "evidences_id",
        "answer_id",
    }


def test_verify_paper_artifacts_checks_machine_readable_claim_manifest(tmp_path):
    verify = _load_script("verify_paper_artifacts")
    manifest_dir = tmp_path / "docs" / "experiments"
    manifest_dir.mkdir(parents=True)
    source_manifest = ROOT / "docs" / "experiments" / "paper_claims_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    (manifest_dir / "paper_claims_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    failures = []
    verify.verify_claim_manifest(tmp_path, failures)

    assert failures == []

    manifest["main_metrics"] = manifest["main_metrics"][:-1]
    (manifest_dir / "paper_claims_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    failures = []
    verify.verify_claim_manifest(tmp_path, failures)

    assert any("claim manifest missing main metric" in failure for failure in failures)


def test_verify_paper_artifacts_checks_protocol_boundary_manifest(tmp_path):
    verify = _load_script("verify_paper_artifacts")
    manifest_dir = tmp_path / "docs" / "experiments"
    manifest_dir.mkdir(parents=True)
    source_manifest = ROOT / "docs" / "experiments" / "paper_claims_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    (manifest_dir / "paper_claims_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    failures = []
    verify.verify_claim_manifest(tmp_path, failures)

    assert failures == []

    manifest["protocol_boundary"] = manifest["protocol_boundary"][:-1]
    (manifest_dir / "paper_claims_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    failures = []
    verify.verify_claim_manifest(tmp_path, failures)

    assert any("claim manifest missing protocol boundary" in failure for failure in failures)


def test_verify_paper_artifacts_flags_2wiki_source_only_fields_in_inference_artifacts(tmp_path):
    verify = _load_script("verify_paper_artifacts")
    pred = tmp_path / "predictions.jsonl"
    cache = tmp_path / "cache.json"
    pred.write_text(
        json.dumps({
            "_id": "q1",
            "question": "Who?",
            "prediction": "A",
            "gold": "A",
            "supporting_facts": [["T", 0]],
        })
        + "\n",
        encoding="utf-8",
    )
    cache.write_text(
        json.dumps({"q1": {"answer": "A", "history": [], "answer_id": "Q1"}}),
        encoding="utf-8",
    )

    failures = []
    verify.verify_no_2wiki_source_only_fields(
        tmp_path,
        failures,
        artifact_paths=["predictions.jsonl", "cache.json"],
    )

    assert any("source-only inference field" in failure for failure in failures)


def test_verify_paper_artifacts_checks_manifest_fingerprint_and_schema(tmp_path):
    verify = _load_script("verify_paper_artifacts")
    artifact = tmp_path / "toy.csv"
    artifact.write_text("qid,prediction,gold,em,f1\nq1,a,a,1,1.0\n", encoding="utf-8")
    checks = {
        "toy.csv": {
            "rows": 1,
            "size": artifact.stat().st_size,
            "sha256": "incorrect",
            "required_fields": ["qid", "prediction", "missing_field"],
        }
    }

    result = verify.verify_artifact_integrity(tmp_path, checks)

    assert result["status"] == "fail"
    assert any("sha256" in failure for failure in result["failures"])
    assert any("missing required fields" in failure for failure in result["failures"])


def test_verify_paper_artifacts_independently_rescores_saved_metrics(tmp_path):
    verify = _load_script("verify_paper_artifacts")
    artifact = tmp_path / "toy.jsonl"
    rows = [
        {"_id": "q1", "prediction": "The EGOT.", "gold": "EGOT", "em": 1, "f1": 1.0},
        {"_id": "q2", "answer": "wrong", "gold": "right answer", "em": 1, "f1": 1.0},
    ]
    artifact.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = verify.verify_artifact_integrity(
        tmp_path,
        {
            "toy.jsonl": {
                "rows": 2,
                "required_fields": ["_id", "gold", "em", "f1"],
                "rescore": True,
            }
        },
    )

    assert result["status"] == "fail"
    assert result["checked"]["toy.jsonl"]["rescored_rows"] == 2
    assert any("independent rescore mismatch" in failure for failure in result["failures"])


def test_verify_paper_artifacts_rescores_ablation_jsonl_rows(tmp_path, monkeypatch):
    verify = _load_script("verify_paper_artifacts")
    summary = tmp_path / "summary.csv"
    rows = [
        {
            "_id": "q1",
            "answer": "The EGOT.",
            "gold": "EGOT",
            "em": 1,
            "f1": 1.0,
            "ablation": "without_repair",
            "evidence_first_ablation": "without_repair",
        },
        {
            "_id": "q2",
            "answer": "wrong",
            "gold": "right answer",
            "em": 0,
            "f1": 0.0,
            "ablation": "without_repair",
            "evidence_first_ablation": "none",
        },
    ]
    score = tmp_path / "score.jsonl"
    summary.write_text("ablation,n,EM,F1\nwithout_repair,2,0.5,0.5\n", encoding="utf-8")
    score.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    key = ("toy", "without_repair")
    monkeypatch.setattr(
        verify,
        "EXPECTED_ABLATIONS",
        {key: {"n": 2, "em": 0.5, "f1": 0.5}},
    )
    monkeypatch.setattr(verify, "ABLATION_SUMMARY_FILES", {key: "summary.csv"})
    monkeypatch.setattr(verify, "ABLATION_SCORE_FILES", {key: "score.jsonl"})

    failures = []
    verify.verify_ablation_claims(tmp_path, failures)

    assert failures == []

    rows[1]["answer"] = "right answer"
    score.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    failures = []
    verify.verify_ablation_claims(tmp_path, failures)

    assert any("ablation JSONL" in failure and "EM" in failure for failure in failures)


def test_verify_paper_artifacts_checks_extended_paper_claims():
    verify = _load_script("verify_paper_artifacts")
    stats = {
        "pairwise_mcnemar": [
            {
                "dataset": "hotpot",
                "subgroup": "all",
                "primary": "EvidenceFirst v4 fresh",
                "baseline": "HopRAG strict",
                "n": 1000,
                "em_delta": "0.0000",
                "f1_delta": "0.0047",
                "mcnemar_exact_p": "0.593902",
            }
        ],
        "audit_overview": [],
        "audit_label_consistency": [],
        "audit_by_gap_type": [],
    }

    failures = []
    verify.verify_extended_paper_claims(stats, {}, {}, failures)

    assert any("pairwise" in failure for failure in failures)
    assert any("audit overview" in failure for failure in failures)


def test_pairwise_stats_reports_paired_ci_and_type_subgroup(monkeypatch):
    stats = _load_script("analyze_evidencefirst_stats")
    monkeypatch.setattr(
        stats,
        "PAIRWISE",
        [("toy", "primary", "baseline", "bridge", "bridge")],
    )
    run_records = {
        "toy": {
            "primary": [
                {"qid": "1", "em": 1.0, "f1": 1.0, "type": "bridge", "raw": {}},
                {"qid": "2", "em": 1.0, "f1": 0.8, "type": "bridge", "raw": {}},
                {"qid": "3", "em": 0.0, "f1": 0.0, "type": "comparison", "raw": {}},
            ],
            "baseline": [
                {"qid": "1", "em": 0.0, "f1": 0.5, "type": "", "raw": {}},
                {"qid": "2", "em": 1.0, "f1": 0.7, "type": "", "raw": {}},
                {"qid": "3", "em": 1.0, "f1": 1.0, "type": "", "raw": {}},
            ],
        }
    }

    rows = stats.pairwise_stats(run_records, iterations=100, seed=7)

    assert len(rows) == 1
    assert rows[0]["subgroup"] == "bridge"
    assert rows[0]["n"] == 2
    assert rows[0]["primary_only"] == 1
    assert rows[0]["baseline_only"] == 0
    assert rows[0]["em_delta"] == "0.5000"
    assert "em_delta_ci95_low" in rows[0]


def test_gap_label_consistency_reports_structural_invariants(monkeypatch):
    stats = _load_script("analyze_evidencefirst_stats")
    monkeypatch.setattr(
        stats,
        "AUDIT_RUNS",
        {"toy": ("method", Path("unused.jsonl"))},
    )
    run_records = {
        "toy": {
            "method": [
                {
                    "qid": "1",
                    "em": 1.0,
                    "f1": 1.0,
                    "raw": {
                        "evidence_first_gap_type": "",
                        "evidence_first_chain_complete": True,
                    },
                },
                {
                    "qid": "2",
                    "em": 0.0,
                    "f1": 0.0,
                    "raw": {
                        "evidence_first_gap_type": "",
                        "evidence_first_chain_complete": False,
                    },
                },
                {
                    "qid": "3",
                    "em": 1.0,
                    "f1": 0.5,
                    "raw": {
                        "evidence_first_gap_type": "missing_entities",
                        "evidence_first_missing_entity_count": 2,
                    },
                },
                {
                    "qid": "4",
                    "em": 0.0,
                    "f1": 0.0,
                    "raw": {
                        "evidence_first_gap_type": "missing_entities",
                        "evidence_first_missing_entity_count": 0,
                        "evidence_first_missing_entities": "",
                    },
                },
                {
                    "qid": "5",
                    "em": 1.0,
                    "f1": 1.0,
                    "raw": {
                        "evidence_first_gap_type": "disconnected",
                        "evidence_first_disconnected_pair_count": 1,
                    },
                },
                {
                    "qid": "6",
                    "em": 1.0,
                    "f1": 1.0,
                    "raw": {
                        "evidence_first_gap_type": "short_chain",
                        "evidence_first_chain_complete": False,
                        "evidence_first_chain_length": 1,
                    },
                },
            ]
        }
    }

    rows = stats.gap_label_consistency(run_records)

    by_gap = {row["gap_type"]: row for row in rows}
    assert by_gap["complete"]["n"] == 2
    assert by_gap["complete"]["consistent_n"] == 1
    assert by_gap["complete"]["consistent_rate"] == "0.5000"
    assert by_gap["missing_entities"]["n"] == 2
    assert by_gap["missing_entities"]["consistent_n"] == 1
    assert by_gap["disconnected"]["consistent_rate"] == "1.0000"
    assert by_gap["short_chain"]["consistent_rate"] == "1.0000"


def test_2wiki_gap_label_gold_audit_reports_proxy_precision_and_lift():
    chain = _load_script("analyze_2wiki_chain_validity")
    rows = [
        {
            "gap_type": "missing_entities",
            "gold_entity_coverage": 0.5,
            "gold_pair_recall": 0.0,
            "gold_pair_complete": False,
            "gold_evidence_nodes_connected": False,
            "em": 0,
            "f1": 0,
        },
        {
            "gap_type": "missing_entities",
            "gold_entity_coverage": 1.0,
            "gold_pair_recall": 1.0,
            "gold_pair_complete": True,
            "gold_evidence_nodes_connected": True,
            "em": 1,
            "f1": 1,
        },
        {
            "gap_type": "disconnected",
            "gold_entity_coverage": 1.0,
            "gold_pair_recall": 0.0,
            "gold_pair_complete": False,
            "gold_evidence_nodes_connected": False,
            "em": 0,
            "f1": 0,
        },
        {
            "gap_type": "short_chain",
            "gold_entity_coverage": 1.0,
            "gold_pair_recall": 0.5,
            "gold_pair_complete": False,
            "gold_evidence_nodes_connected": True,
            "em": 1,
            "f1": 0.5,
        },
        {
            "gap_type": "complete",
            "gold_entity_coverage": 1.0,
            "gold_pair_recall": 1.0,
            "gold_pair_complete": True,
            "gold_evidence_nodes_connected": True,
            "em": 1,
            "f1": 1,
        },
    ]

    audit = chain.gap_label_gold_audit(rows)

    by_label = {row["gap_type"]: row for row in audit}
    assert by_label["missing_entities"]["target_name"] == "missing_gold_entity"
    assert by_label["missing_entities"]["label_n"] == 2
    assert by_label["missing_entities"]["true_positive_n"] == 1
    assert by_label["missing_entities"]["precision"] == 0.5
    assert by_label["missing_entities"]["recall"] == 1.0
    assert by_label["missing_entities"]["target_prevalence"] == 0.2
    assert by_label["missing_entities"]["lift"] == 2.5
    assert by_label["disconnected"]["precision"] == 1.0
    assert by_label["short_chain"]["precision"] == 1.0


def test_audit_risk_triage_beats_single_signal_baselines_on_toy_rows():
    audit = _load_script("analyze_audit_risk")
    raw_rows = [
        {
            "_id": "a",
            "em": 1,
            "f1": 1,
            "evidence_first_postprocess_selected": True,
            "evidence_first_chain_complete": True,
            "evidence_first_gap_type": "",
        },
        {
            "_id": "b",
            "em": 1,
            "f1": 1,
            "evidence_first_postprocess_selected": True,
            "evidence_first_chain_complete": False,
            "evidence_first_gap_type": "missing_entities",
            "evidence_first_repair_steps": "B_repair",
        },
        {
            "_id": "c",
            "em": 0,
            "f1": 0,
            "evidence_first_postprocess_selected": True,
            "evidence_first_chain_complete": False,
            "evidence_first_gap_type": "short_chain",
            "evidence_first_repair_steps": "B_repair",
        },
        {
            "_id": "d",
            "em": 0,
            "f1": 0,
            "evidence_first_postprocess_selected": False,
            "evidence_first_chain_complete": True,
            "evidence_first_gap_type": "",
        },
        {
            "_id": "e",
            "em": 1,
            "f1": 1,
            "evidence_first_postprocess_selected": False,
            "evidence_first_chain_complete": True,
            "evidence_first_gap_type": "",
        },
        {
            "_id": "f",
            "em": 0,
            "f1": 0,
            "evidence_first_postprocess_selected": False,
            "evidence_first_chain_complete": False,
            "evidence_first_gap_type": "short_chain",
        },
    ]
    detailed, _summary = audit.build_rows("toy", raw_rows)
    metrics = {
        row["signal"]: row
        for row in audit.triage_metrics("toy", detailed, top_fraction=0.34)
    }

    assert "graph_audit_score" in metrics
    assert float(metrics["graph_audit_score"]["error_auc"]) > float(
        metrics["chain_incomplete"]["error_auc"]
    )
    assert float(metrics["audit_risk_score"]["error_auc"]) > float(
        metrics["answer_not_selected"]["error_auc"]
    )
    assert float(metrics["audit_risk_score"]["error_auc"]) > float(
        metrics["chain_incomplete"]["error_auc"]
    )
    assert metrics["audit_risk_score"]["topk_error_n"] == 2


def test_semantic_gap_adjudication_majority_and_agreement():
    adjudication = _load_script("run_semantic_gap_adjudication")
    chain_row = {
        "qid": "q1",
        "gap_type": "missing_entities",
        "chain_complete": "False",
        "gold_entity_coverage": "0.5",
        "gold_pair_recall": "0.0",
        "gold_relation_recall": "0.0",
        "gold_pair_complete": "False",
        "gold_evidence_nodes_connected": "False",
        "em": "0",
        "f1": "0",
    }
    support_row = {
        "support_title_recall": "1.0",
        "evidence_entity_coverage": "1.0",
    }

    annotations, final = adjudication.adjudicate(chain_row, support_row)

    assert [row["agent"] for row in annotations] == list(adjudication.AGENTS)
    assert final["majority_adjudication"] == "match"
    assert final["match_votes"] == 3
    assert final["adjudicated_correct"]
    assert not final["primary_disagreement"]
    assert adjudication.fleiss_kappa([["match", "match", "match"]]) == 1.0


def test_semantic_gap_adjudication_stratified_sample_oversamples_rare_labels():
    adjudication = _load_script("run_semantic_gap_adjudication")
    rows = (
        [{"qid": f"c{i}", "gap_type": "complete", "chain_complete": "True"} for i in range(40)]
        + [{"qid": f"m{i}", "gap_type": "missing_entities", "chain_complete": "False"} for i in range(40)]
        + [{"qid": f"s{i}", "gap_type": "short_chain", "chain_complete": "False"} for i in range(25)]
        + [{"qid": f"d{i}", "gap_type": "disconnected", "chain_complete": "False"} for i in range(5)]
    )

    sample = adjudication.stratified_sample(rows, sample_size=50, seed=13)
    counts = {}
    for row in sample:
        counts[row["gap_type"]] = counts.get(row["gap_type"], 0) + 1

    assert len(sample) == 50
    assert counts["disconnected"] == 5
    assert counts["complete"] > 0
    assert counts["missing_entities"] > 0
    assert counts["short_chain"] > 0


def test_semantic_gap_adjudication_current_summary_is_reproducible():
    adjudication = _load_script("run_semantic_gap_adjudication")
    chain_rows = adjudication.load_csv(
        ROOT / "results/analysis/2wiki_chain_validity/2wiki_chain_validity_per_example.csv"
    )
    support_rows = adjudication.load_csv(
        ROOT / "results/analysis/2wiki_support/2wiki_support_per_example.csv"
    )
    data_items = adjudication.load_json_items(ROOT / "comagraag/data/2wiki_sample.json")

    _csv_rows, _jsonl_rows, summary = adjudication.build_outputs(
        chain_rows=chain_rows,
        support_rows=support_rows,
        data_items=data_items,
        sample_size=100,
        seed=20260608,
    )

    assert summary["sample_n"] == 100
    assert dict(summary["sample_quotas"]) == {
        "complete": 34,
        "disconnected": 11,
        "missing_entities": 35,
        "short_chain": 20,
    }
    assert summary["overall"]["match_n"] == 40
    assert summary["overall"]["mismatch_n"] == 31
    assert summary["overall"]["ambiguous_n"] == 29
    assert summary["overall"]["match_wilson_ci95"] == (0.3094, 0.498)
    assert summary["by_gap_type"]["missing_entities"]["match_wilson_ci95"] == (0.5793, 0.8584)
    assert summary["inter_agent_agreement"]["fleiss_kappa"] == 0.1836
