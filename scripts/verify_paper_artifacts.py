#!/usr/bin/env python3
"""Reviewer-facing no-LLM verification for EvidenceFirst paper artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
import string
from collections import Counter
from pathlib import Path


ABLATION_SCORE_FILES = {
    ("hotpot", "without_verification"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl",
    ("hotpot", "without_repair"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_repair.jsonl",
    ("hotpot", "without_reader_context"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_reader_context.jsonl",
    ("hotpot", "without_answer_refinement"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.jsonl",
    ("2wiki", "without_verification"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl",
    ("2wiki", "without_repair"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_repair.jsonl",
    ("2wiki", "without_reader_context"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_reader_context.jsonl",
    ("2wiki", "without_answer_refinement"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.jsonl",
}

REQUIRED_FILES = [
    "docs/experiments/paper_claims_manifest.json",
    "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v4.jsonl",
    "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl",
    "results/wise_hotpot1000_hoprag_strict_response_metrics.csv",
    "results/wise_hotpot1000_arag_full_post_combined_metrics.csv",
    "results/wise/hotpot1000_ircot_combined_predictions.jsonl",
    "results/wise/hotpot1000_lightrag_combined_predictions.jsonl",
    "results/wise/hotpot1000_ms_graphrag_combined_predictions.jsonl",
    "results/wise/hotpot1000_naive_rag_combined_predictions.jsonl",
    "results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl",
    "results/wise/2wiki_evidencefirst_v6_localctx2_predictions.jsonl",
    "results/wise/2wiki_lightrag_predictions.jsonl",
    "results/wise_2wiki_hoprag_strict_metrics.csv",
    "results/wise/2wiki_hoprag_strict_predictions.jsonl",
    "results/wise_2wiki_arag_full_post_metrics.csv",
    "results/wise/2wiki_arag_full_post_predictions.jsonl",
    "external_runs/2wiki500/arag/chunks.json",
    "results/wise/2wiki_ircot_predictions.jsonl",
    "results/wise/2wiki_naive_rag_predictions.jsonl",
    "comagraag/data/2wiki_sample.json",
    "results/wise/2wiki_evidencefirst_v6_kgs.pkl",
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.csv",
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_repair.csv",
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv",
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv",
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.csv",
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_repair.csv",
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv",
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv",
    "results/analysis/audit_risk/audit_risk_per_example.csv",
    "results/analysis/2wiki_support/2wiki_support_summary.json",
    "results/analysis/2wiki_chain_validity/2wiki_chain_validity_summary.json",
    *ABLATION_SCORE_FILES.values(),
]

REQUIRED_PATTERNS = [
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_s100_full.json",
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b*_full.json",
]

ARTIFACT_CHECKS = {
    "docs/experiments/paper_claims_manifest.json": {
        "required_fields": [
            "version",
            "paper",
            "main_metrics",
            "paired_statistics",
            "semantic_gap_adjudication",
            "blind_gap_adjudication",
            "ablations",
        ],
    },
    "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v4.jsonl": {
        "size": 1485810,
        "sha256": "7efd2a874415d32b1b7c273696e5977f27732cf3ebd46ce569e840c9461cbb38",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl": {
        "size": 1791586,
        "sha256": "7cf263ceadecf5078ab5fbd6a02ecebec73262c4bf363b0f169301ee1d2d8cef",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise_hotpot1000_hoprag_strict_response_metrics.csv": {
        "size": 153118,
        "sha256": "18aba29cd059b3aac79c68fe628a64ff14901a9dff442a7b69729a0203bfdf9f",
        "rows": 1000,
        "required_fields": ["qid", "prediction", "gold", "em", "f1"],
    },
    "results/wise_hotpot1000_arag_full_post_combined_metrics.csv": {
        "size": 153498,
        "sha256": "4a5236239f384ced96573dd88de7393da1ce7737783c1f8f0169b3d41c0a4416",
        "rows": 1000,
        "required_fields": ["qid", "prediction", "gold", "em", "f1"],
    },
    "results/wise/hotpot1000_ircot_combined_predictions.jsonl": {
        "size": 525933,
        "sha256": "138fcc6d9d6ac6aef2ce72854d4f26a8dcce9246f82a90a2f5ac07a82ecc2681",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/hotpot1000_lightrag_combined_predictions.jsonl": {
        "size": 502955,
        "sha256": "597c9d2e2385eca55b493f1b057127f69f5934531104ecf15672224c1d406b99",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/hotpot1000_ms_graphrag_combined_predictions.jsonl": {
        "size": 585069,
        "sha256": "b86d33f97d4bd0028e818d0f89e4770ee57c50e6565be19c1ac44a90218d4181",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/hotpot1000_naive_rag_combined_predictions.jsonl": {
        "size": 459331,
        "sha256": "e41fbea6dcbef7571161352c145a832024b9250de114ee5d087b5b9882b35e52",
        "rows": 1000,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl": {
        "size": 809804,
        "sha256": "9c947c7c91c2ce1cae1e751570701de72c817678cddf7259c30f0bea1f7734a8",
        "rows": 500,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/2wiki_evidencefirst_v6_localctx2_predictions.jsonl": {
        "size": 751956,
        "sha256": "40e78b36baf8570af2bfc107a09c8852ca6a768bf0c77c25a36d93a5637a1415",
        "rows": 500,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/2wiki_lightrag_predictions.jsonl": {
        "size": 254550,
        "sha256": "a0ebab92bd77cb1ae4f977c90e7137cb3bf541369f1497713fe32b70f2c69cb4",
        "rows": 500,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise_2wiki_hoprag_strict_metrics.csv": {
        "size": 73061,
        "sha256": "f95368b0366813bce1abc3ec91a1b43d219563489e357bea455225cdcfe09c59",
        "rows": 500,
        "required_fields": ["qid", "prediction", "gold", "em", "f1"],
    },
    "results/wise/2wiki_hoprag_strict_predictions.jsonl": {
        "size": 114885,
        "sha256": "6b9b01c4846f2b9029db93cef7013938d31375a9562fc892386e2f004aff8a18",
        "rows": 500,
        "required_fields": ["_id", "answer", "question", "response"],
    },
    "results/wise_2wiki_arag_full_post_metrics.csv": {
        "size": 73902,
        "sha256": "a430467318c1b94837b5279e8055d0bbc846030ad67cf2c1621a5d865042866f",
        "rows": 500,
        "required_fields": ["qid", "prediction", "gold", "em", "f1"],
    },
    "results/wise/2wiki_arag_full_post_predictions.jsonl": {
        "size": 4197765,
        "sha256": "ef5b6508805a1e283cd4356fd63c41c822fb67c5eb55098854c359689e4e4b37",
        "rows": 500,
        "required_fields": ["qid", "question", "pred_answer", "gold_answer"],
    },
    "external_runs/2wiki500/arag/chunks.json": {
        "size": 2547192,
        "sha256": "458e74faa3d5edc0cacf5a4f592f5e2bb90ee18cbb973df6f0bc1d263a466308",
        "rows": 5000,
    },
    "results/wise/2wiki_ircot_predictions.jsonl": {
        "size": 260142,
        "sha256": "eb299f646f833f702916a70a759489c01662411ae8c2a18974852e3793946b76",
        "rows": 500,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "results/wise/2wiki_naive_rag_predictions.jsonl": {
        "size": 255955,
        "sha256": "0664bf9cb865705319af40c0eaf28ca7bdd2926bda000a7e4c2d52085098b37d",
        "rows": 500,
        "required_fields": ["_id", "answer", "em", "f1", "gold"],
    },
    "comagraag/data/2wiki_sample.json": {
        "size": 2795730,
        "sha256": "81548e452da402932769b17c0b9741ecede6e3ede1fbac494a7ab3527191d141",
        "rows": 500,
    },
    "results/wise/2wiki_evidencefirst_v6_kgs.pkl": {
        "size": 4260099,
        "sha256": "096ae66cafac2d08014f73b426e7fc217098aa07a002d78f7e01776d94951296",
    },
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_s100_full.json": {
        "size": 223396,
        "sha256": "09d57853126408dd136c7ee3983be3b97db50cfbc5bdd8292320b5ba43c12b95",
        "rows": 100,
    },
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b1_full.json": {
        "size": 222859,
        "sha256": "d00c7fef622c401e57365b2feddacdee9e3c6680c611d7d430ababe1954c0a90",
        "rows": 100,
    },
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b2_full.json": {
        "size": 218459,
        "sha256": "5f58904d52f7ed628ada3b00c881cea156e6b3c29b8acda438924faa69b543f2",
        "rows": 100,
    },
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b3_full.json": {
        "size": 223411,
        "sha256": "0008f3441455fe344b87ab2208218032a83c0b460b90a03717d0299c3ce6e0b1",
        "rows": 100,
    },
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b4_full.json": {
        "size": 228931,
        "sha256": "c3b11e7193f586fa3c72a8118195e248cfed8bc4bf5a08a1c724defbe3b47018",
        "rows": 100,
    },
    "results/analysis/2wiki_support/2wiki_support_summary.json": {
        "size": 2335,
        "sha256": "6281c6d02142ce1a2e556237ec427a6fff048bd6217fbfedf6688068f146e987",
        "rows": 8,
        "required_fields": [
            "n",
            "evidencefirst_reader_full",
            "naive_bm25_top5_input",
            "arag_actual_read_input",
            "arag_keyword_search_found_upper_bound",
        ],
    },
    "results/analysis/2wiki_chain_validity/2wiki_chain_validity_summary.json": {
        "size": 6593,
        "sha256": "4c5cd9d9897b56438842ae574318127b7710dda4498613ee17114defe2aa2953",
        "rows": 7,
        "required_fields": ["n", "overall", "gap_label_gold_audit"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.csv": {
        "size": 219,
        "sha256": "29ea743a2121a2d89924d4d7dc0cf48d436507729a8784724d044cd2975f8d54",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_repair.csv": {
        "size": 212,
        "sha256": "93f0bf8426c733cb4d83b1e44fe2ab526e66a07981c8bcfdd1a51cbbe5a6c742",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv": {
        "size": 221,
        "sha256": "ec213ad9f17cc424c6a1fe52f70aba4b2c58e49cc5af1828f987b92a4a79ec1f",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv": {
        "size": 223,
        "sha256": "325bfe4f0bb9384d42cf698f67da49b26943ef6e7c14bc6046c5404c3e71fd9b",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.csv": {
        "size": 216,
        "sha256": "d76e6de201de5e6e0ba592d988023730338a2c48d05ccf8ad0f848c083cb3e7f",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_repair.csv": {
        "size": 209,
        "sha256": "24870b5cb9b56dcf1a6d4ab3687f3a30b956e60a67ccb4660587c67d658d4a07",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv": {
        "size": 218,
        "sha256": "e107ef4086a7af9354369f2a6f700ecda05f910b6aa5587526dd9d51e59bab85",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv": {
        "size": 221,
        "sha256": "6f1f821be1f31c7669ef2298f4296b0b4069c0108f6be4a0aaa33985f27a6359",
        "rows": 1,
        "required_fields": ["ablation", "n", "EM", "F1"],
    },
}

ARTIFACT_CHECKS.update({
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl": {
        "size": 1732521,
        "sha256": "7361bd89dc67805090da20688b09749e4578aa47a569407b420036779c6d17ca",
        "rows": 1000,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_repair.jsonl": {
        "size": 1749946,
        "sha256": "b0b768e8f35e3c1109fdb6ebc0780116d257e27f19ba602ee1751d746d439f66",
        "rows": 1000,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_reader_context.jsonl": {
        "size": 1754790,
        "sha256": "25156254cddd16cf57e855dc3e7372a4efc076e0b9eca420ce0cb85edb909294",
        "rows": 1000,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.jsonl": {
        "size": 1758144,
        "sha256": "09997b61c4c0c4633095a3be5b34c1925bafeb6ec6c38a7dad99e26002b1ba72",
        "rows": 1000,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.jsonl": {
        "size": 871957,
        "sha256": "672f675c4d3e994098ea3c13dc9e7f840ab1dc6adce7897a9b8da186c7e6b441",
        "rows": 500,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_repair.jsonl": {
        "size": 879882,
        "sha256": "ee6de80470fbb805794a875382eb2ce1cc06dd09a8f4cfdb2f18873b39884ef6",
        "rows": 500,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_reader_context.jsonl": {
        "size": 879908,
        "sha256": "01044dfe207e7550af8030020b238fddf9b70eb4d0a4d60ff9ab9ae21d25452a",
        "rows": 500,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
    "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.jsonl": {
        "size": 883954,
        "sha256": "3769604789a0f38a231a888992be0f73afc5800ac977f9107059416406273940",
        "rows": 500,
        "required_fields": ["_id", "answer", "gold", "em", "f1", "evidence_first_ablation"],
    },
})

STALE_CLAIM_PATTERNS = [
    "HopRAG strict for 2Wiki is still running",
    "HopRAG strict on 2Wiki is still running",
    "2Wiki comparison lacks completed HopRAG strict",
    "The 2Wiki comparison lacks completed HopRAG strict",
    "HopRAG strict & Running",
]

EXPECTED_PROTOCOL_BOUNDARY = {
    "HotpotQA EvidenceFirst end-to-end",
    "HotpotQA EvidenceFirst selector variant",
    "2Wiki EvidenceFirst reader-full",
    "2Wiki EvidenceFirst local-context stress",
    "2Wiki visibility audit",
    "Naive RAG / IRCoT / A-RAG",
    "LightRAG / MS GraphRAG",
    "HopRAG strict",
}

FORBIDDEN_2WIKI_INFERENCE_FIELDS = {
    "context",
    "supporting_facts",
    "evidences",
    "evidences_id",
    "answer_id",
}

DEFAULT_2WIKI_INFERENCE_ARTIFACTS = [
    "results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl",
    "results/wise/2wiki_evidencefirst_v6_localctx2_predictions.jsonl",
    "results/wise/2wiki_lightrag_predictions.jsonl",
    "results/wise/2wiki_hoprag_strict_predictions.jsonl",
    "results/wise/2wiki_arag_full_post_predictions.jsonl",
    "results/wise/2wiki_ircot_predictions.jsonl",
    "results/wise/2wiki_naive_rag_predictions.jsonl",
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_s100_full.json",
    "results/cache_wise_2wiki_evidencefirst_v6_readerfull_b*_full.json",
]

EXPECTED_MAIN = {
    ("hotpot", "EvidenceFirst v4 fresh"): ("0.5500", "0.6484"),
    ("hotpot", "EvidenceFirst v6 selector guard"): ("0.5620", "0.6564"),
    ("hotpot", "HopRAG strict"): ("0.5410", "0.6438"),
    ("2wiki", "EvidenceFirst v6"): ("0.6900", "0.7581"),
    ("2wiki", "EvidenceFirst v6 local-context"): ("0.5040", "0.5945"),
    ("2wiki", "LightRAG"): ("0.6740", "0.7716"),
    ("2wiki", "HopRAG strict"): ("0.5640", "0.6157"),
}

EXPECTED_SUPPORT = {
    "em": 0.69,
    "f1": 0.7581,
    "support_title_recall": 1.0,
    "evidence_entity_coverage": 0.9468,
}

EXPECTED_CHAIN = {
    "em": 0.69,
    "f1": 0.7581,
    "gold_entity_coverage": 0.7657,
    "gold_pair_recall": 0.4778,
    "gold_relation_recall": 0.042,
}

EXPECTED_CONTEXT_VISIBILITY = {
    "EvidenceFirst reader-full input": {
        "summary_key": "evidencefirst_reader_full",
        "n": 500,
        "support_title_recall": 1.0,
        "evidence_entity_coverage": 0.9468,
        "em": 0.69,
        "f1": 0.7581,
    },
    "Naive BM25 top-5 input": {
        "summary_key": "naive_bm25_top5_input",
        "n": 500,
        "support_title_recall": 0.602,
        "evidence_entity_coverage": 0.6685,
    },
    "A-RAG actual read chunks": {
        "summary_key": "arag_actual_read_input",
        "n": 500,
        "support_title_recall": 0.2955,
        "evidence_entity_coverage": 0.3326,
    },
    "A-RAG keyword-search found upper bound": {
        "summary_key": "arag_keyword_search_found_upper_bound",
        "n": 500,
        "support_title_recall": 0.5885,
        "evidence_entity_coverage": 0.6085,
    },
    "EvidenceFirst saved KG": {
        "summary_key": "overall",
        "n": 500,
        "gold_entity_coverage": 0.7657,
        "gold_pair_recall": 0.4778,
        "gold_relation_recall": 0.042,
        "em": 0.69,
        "f1": 0.7581,
    },
}

EXPECTED_ADJUDICATION = {
    "sample_n": 100,
    "match_rate": 0.4,
    "match_ci_low": 0.3094,
    "match_ci_high": 0.498,
    "mismatch_rate": 0.31,
    "ambiguous_rate": 0.29,
    "fleiss_kappa": 0.1836,
}

EXPECTED_BLIND_ADJUDICATION = {
    "sample_n": 100,
    "match_rate": 0.53,
    "mismatch_rate": 0.36,
    "ambiguous_rate": 0.11,
}

EXPECTED_PAIRWISE = {
    ("hotpot", "all", "EvidenceFirst v4 fresh", "HopRAG strict"): {
        "n": 1000, "em_delta": 0.009, "f1_delta": 0.0047, "mcnemar_exact_p": 0.593902,
    },
    ("hotpot", "all", "EvidenceFirst v6 selector guard", "HopRAG strict"): {
        "n": 1000, "em_delta": 0.021, "f1_delta": 0.0126, "mcnemar_exact_p": 0.174424,
    },
    ("hotpot", "bridge", "EvidenceFirst v6 selector guard", "HopRAG strict"): {
        "n": 500, "em_delta": 0.058, "f1_delta": 0.0468, "mcnemar_exact_p": 0.010611,
    },
    ("2wiki", "all", "EvidenceFirst v6", "LightRAG"): {
        "n": 500, "em_delta": 0.016, "f1_delta": -0.0135, "mcnemar_exact_p": 0.402963,
    },
    ("2wiki", "all", "EvidenceFirst v6", "HopRAG strict"): {
        "n": 500, "em_delta": 0.126, "f1_delta": 0.1424, "mcnemar_exact_p": 0.0,
    },
    ("2wiki", "all", "EvidenceFirst v6", "A-RAG"): {
        "n": 500, "em_delta": 0.024, "f1_delta": 0.0259, "mcnemar_exact_p": 0.256442,
    },
    ("2wiki", "local-context stress", "EvidenceFirst v6 local-context", "HopRAG strict"): {
        "n": 500, "em_delta": -0.06, "f1_delta": -0.0212, "mcnemar_exact_p": 0.021574,
    },
}

EXPECTED_AUDIT_OVERVIEW = {
    ("hotpot", "EvidenceFirst v6 selector guard"): {
        "n": 1000,
        "chain_complete_rate": 0.568,
        "repair_attempt_rate": 0.637,
        "repair_to_complete_rate": 0.4349,
        "selector_selected_rate": 0.821,
    },
    ("2wiki", "EvidenceFirst v6"): {
        "n": 500,
        "chain_complete_rate": 0.478,
        "repair_attempt_rate": 0.6,
        "repair_to_complete_rate": 0.22,
        "selector_selected_rate": 0.704,
    },
}

EXPECTED_GAP_LABELS = {
    ("2wiki", "complete"): {
        "n": 239, "em": 0.7029, "f1": 0.7681, "consistent_rate": 1.0,
    },
    ("2wiki", "missing_entities"): {
        "n": 217, "em": 0.6959, "f1": 0.7667, "consistent_rate": 1.0,
    },
    ("2wiki", "short_chain"): {
        "n": 33, "em": 0.6364, "f1": 0.7035, "consistent_rate": 1.0,
    },
    ("2wiki", "disconnected"): {
        "n": 11, "em": 0.4545, "f1": 0.5351, "consistent_rate": 1.0,
    },
    ("hotpot", "complete"): {"n": 577, "consistent_rate": 0.9844},
    ("hotpot", "missing_entities"): {"n": 302, "consistent_rate": 1.0},
    ("hotpot", "short_chain"): {"n": 118, "consistent_rate": 1.0},
    ("hotpot", "disconnected"): {"n": 3, "consistent_rate": 1.0},
}

EXPECTED_DIAGNOSTIC_UTILITY = {
    "complete": {"gold_proxy_precision": 0.1506, "semantic_match": 0.1176, "blind_match": 0.6765},
    "missing_entities": {"gold_proxy_precision": 0.6866, "semantic_match": 0.7429, "blind_match": 0.5714},
    "short_chain": {"gold_proxy_precision": 0.4545, "semantic_match": 0.4, "blind_match": 0.2},
    "disconnected": {"gold_proxy_precision": 0.0909, "semantic_match": 0.1818, "blind_match": 0.5455},
}

EXPECTED_ANSWER_SELECTION = {
    "hotpot": {
        "selected_em": 0.6273,
        "not_selected_em": 0.2626,
    },
    "2wiki": {
        "selected_em": 0.7358,
        "not_selected_em": 0.5811,
    },
}

EXPECTED_RISK_BINS = {
    ("hotpot", "0"): {"n": 468, "em": 0.6581},
    ("hotpot", "1"): {"n": 255, "em": 0.6078},
    ("hotpot", "2"): {"n": 148, "em": 0.3986},
    ("hotpot", "3"): {"n": 103, "em": 0.3592},
    ("hotpot", "4+"): {"n": 26, "em": 0.1154},
    ("2wiki", "0"): {"n": 170, "em": 0.7412},
    ("2wiki", "1"): {"n": 149, "em": 0.7383},
    ("2wiki", "2"): {"n": 81, "em": 0.6049},
    ("2wiki", "3"): {"n": 89, "em": 0.6404},
    ("2wiki", "4+"): {"n": 11, "em": 0.2727},
}

EXPECTED_TRIAGE_UTILITY = {
    ("hotpot", "audit_risk_score"): {
        "error_auc": 0.6321,
        "topk_error_rate": 0.67,
        "topk_lift": 1.5297,
    },
    ("hotpot", "graph_audit_score"): {
        "error_auc": 0.5434,
        "topk_error_rate": 0.525,
        "topk_lift": 1.1986,
    },
    ("hotpot", "answer_not_selected"): {
        "error_auc": 0.6077,
        "topk_error_rate": 0.7,
        "topk_lift": 1.5982,
    },
    ("hotpot", "chain_incomplete"): {
        "error_auc": 0.5321,
        "topk_error_rate": 0.505,
        "topk_lift": 1.153,
    },
    ("2wiki", "audit_risk_score"): {
        "error_auc": 0.5787,
        "topk_error_rate": 0.4,
        "topk_lift": 1.2903,
    },
    ("2wiki", "graph_audit_score"): {
        "error_auc": 0.523,
        "topk_error_rate": 0.35,
        "topk_lift": 1.129,
    },
    ("2wiki", "answer_not_selected"): {
        "error_auc": 0.5754,
        "topk_error_rate": 0.4,
        "topk_lift": 1.2903,
    },
    ("2wiki", "chain_incomplete"): {
        "error_auc": 0.5144,
        "topk_error_rate": 0.29,
        "topk_lift": 0.9355,
    },
}

EXPECTED_ROUTING_QUEUES = {
    ("hotpot", "selected_checked"): {
        "n": 468,
        "error_rate": 0.3419,
        "error_ci_low": 0.3003,
        "error_ci_high": 0.3860,
    },
    ("hotpot", "selected_unrepaired_incomplete"): {
        "n": 51,
        "error_rate": 0.6078,
        "error_ci_low": 0.4708,
        "error_ci_high": 0.7297,
    },
    ("hotpot", "not_selected_no_residual"): {
        "n": 156,
        "error_rate": 0.7179,
        "error_ci_low": 0.6428,
        "error_ci_high": 0.7827,
    },
    ("hotpot", "not_selected_residual_gap"): {
        "n": 22,
        "error_rate": 0.8636,
        "error_ci_low": 0.6666,
        "error_ci_high": 0.9525,
    },
    ("2wiki", "not_selected_no_residual"): {
        "n": 137,
        "error_rate": 0.3942,
        "error_ci_low": 0.3163,
        "error_ci_high": 0.4778,
    },
    ("2wiki", "not_selected_residual_gap"): {
        "n": 11,
        "error_rate": 0.7273,
        "error_ci_low": 0.4343,
        "error_ci_high": 0.9025,
    },
}

EXPECTED_ROUTING_DELTAS = {
    ("hotpot", "selected_unrepaired_minus_checked"): {
        "delta_error": 0.2660,
        "delta_ci_low": 0.1220,
        "delta_ci_high": 0.3947,
    },
    ("2wiki", "not_selected_residual_minus_no_residual"): {
        "delta_error": 0.3331,
        "delta_ci_low": 0.0285,
        "delta_ci_high": 0.5249,
    },
}

EXPECTED_ABLATIONS = {
    ("hotpot", "full"): {"n": 1000, "em": 0.562, "f1": 0.6564},
    ("hotpot", "without_verification"): {"n": 1000, "em": 0.568, "f1": 0.6647},
    ("hotpot", "without_repair"): {"n": 1000, "em": 0.554, "f1": 0.658},
    ("hotpot", "without_reader_context"): {"n": 1000, "em": 0.236, "f1": 0.2899},
    ("hotpot", "without_answer_refinement"): {"n": 1000, "em": 0.551, "f1": 0.6485},
    ("2wiki", "full"): {"n": 500, "em": 0.69, "f1": 0.7581},
    ("2wiki", "without_verification"): {"n": 500, "em": 0.678, "f1": 0.7579},
    ("2wiki", "without_repair"): {"n": 500, "em": 0.65, "f1": 0.7424},
    ("2wiki", "without_reader_context"): {"n": 500, "em": 0.184, "f1": 0.2289},
    ("2wiki", "without_answer_refinement"): {"n": 500, "em": 0.69, "f1": 0.7782},
}

ABLATION_SUMMARY_FILES = {
    ("hotpot", "without_verification"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_verification.csv",
    ("hotpot", "without_repair"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_repair.csv",
    ("hotpot", "without_reader_context"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv",
    ("hotpot", "without_answer_refinement"): "experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv",
    ("2wiki", "without_verification"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_verification.csv",
    ("2wiki", "without_repair"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_repair.csv",
    ("2wiki", "without_reader_context"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_reader_context.csv",
    ("2wiki", "without_answer_refinement"): "experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_answer_refinement.csv",
}

ABLATION_FILES = ABLATION_SUMMARY_FILES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _missing_paths(root: Path) -> list[str]:
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    for pattern in REQUIRED_PATTERNS:
        if not glob.glob(str(root / pattern)):
            missing.append(pattern)
    return missing


def _pattern_match_counts(root: Path) -> dict[str, int]:
    return {
        pattern: len(glob.glob(str(root / pattern)))
        for pattern in REQUIRED_PATTERNS
    }


def _rescore_summary(integrity: dict) -> dict[str, int]:
    rescored_rows = [
        int(item.get("rescored_rows", 0))
        for item in integrity.get("checked", {}).values()
    ]
    return {
        "independently_rescored_files": sum(1 for count in rescored_rows if count > 0),
        "independently_rescored_rows": sum(rescored_rows),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count_and_fields(path: Path) -> tuple[int | None, set[str]]:
    if path.suffix == ".jsonl":
        count = 0
        fields: set[str] = set()
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                count += 1
                if not fields:
                    fields = set(json.loads(line).keys())
        return count, fields
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or [])
            return sum(1 for _ in reader), fields
    if path.suffix == ".json":
        data = _load_json(path)
        if isinstance(data, list):
            fields = set(data[0].keys()) if data and isinstance(data[0], dict) else set()
            return len(data), fields
        if isinstance(data, dict):
            return len(data), set(data.keys())
    return None, set()


def _normalize_answer(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in string.punctuation)
    return " ".join(value.split())


def _em(prediction: object, gold: object) -> int:
    return int(_normalize_answer(prediction) == _normalize_answer(gold))


def _f1(prediction: object, gold: object) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_score_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    return []


def _prediction_for_rescore(row: dict) -> object | None:
    for field in ["prediction", "pred_answer", "predicted_answer"]:
        if field in row:
            return row.get(field)
    if ("gold" in row or "gold_answer" in row) and "answer" in row:
        return row.get("answer")
    return None


def _gold_for_rescore(row: dict) -> object | None:
    for field in ["gold", "gold_answer"]:
        if field in row:
            return row.get(field)
    return None


def _should_rescore(expected: dict, fields: set[str]) -> bool:
    if "rescore" in expected:
        return bool(expected["rescore"])
    return (
        {"em", "f1"}.issubset(fields)
        and bool(fields & {"prediction", "pred_answer", "predicted_answer", "answer"})
        and bool(fields & {"gold", "gold_answer"})
    )


def _independent_rescore(path: Path) -> dict:
    failures: list[str] = []
    rows = _load_score_rows(path)
    if not rows:
        return {
            "rescored_rows": 0,
            "failures": [f"{path}: cannot independently rescore unsupported or empty artifact"],
        }
    checked = 0
    for index, row in enumerate(rows, start=1):
        if "em" not in row or "f1" not in row:
            continue
        prediction = _prediction_for_rescore(row)
        gold = _gold_for_rescore(row)
        if prediction is None or gold is None:
            continue
        checked += 1
        actual_em = _em(prediction, gold)
        actual_f1 = _f1(prediction, gold)
        saved_em = float(row.get("em") or 0)
        saved_f1 = float(row.get("f1") or 0)
        if round(saved_em, 4) != round(float(actual_em), 4):
            qid = row.get("_id") or row.get("id") or row.get("qid") or index
            failures.append(
                f"{path}: independent rescore mismatch at row {index} qid={qid}: "
                f"saved EM {saved_em:.4f}, recomputed EM {actual_em:.4f}"
            )
        if round(saved_f1, 4) != round(float(actual_f1), 4):
            qid = row.get("_id") or row.get("id") or row.get("qid") or index
            failures.append(
                f"{path}: independent rescore mismatch at row {index} qid={qid}: "
                f"saved F1 {saved_f1:.4f}, recomputed F1 {actual_f1:.4f}"
            )
    if checked == 0:
        failures.append(f"{path}: no rows could be independently rescored")
    return {
        "rescored_rows": checked,
        "failures": failures,
    }


def verify_artifact_integrity(root: Path, checks: dict[str, dict] | None = None) -> dict:
    checks = checks or ARTIFACT_CHECKS
    failures: list[str] = []
    checked = {}
    for relative_path, expected in checks.items():
        path = root / relative_path
        if not path.exists():
            failures.append(f"{relative_path}: missing artifact")
            continue
        actual_size = path.stat().st_size
        actual_sha = _sha256(path)
        row_count, fields = _row_count_and_fields(path)
        checked[relative_path] = {
            "size": actual_size,
            "sha256": actual_sha,
            "rows": row_count,
        }
        if "size" in expected and actual_size != expected["size"]:
            failures.append(f"{relative_path}: size expected {expected['size']}, got {actual_size}")
        if "sha256" in expected and actual_sha != expected["sha256"]:
            failures.append(f"{relative_path}: sha256 expected {expected['sha256']}, got {actual_sha}")
        if "rows" in expected and row_count != expected["rows"]:
            failures.append(f"{relative_path}: rows expected {expected['rows']}, got {row_count}")
        required_fields = set(expected.get("required_fields", []))
        missing_fields = sorted(required_fields - fields)
        if missing_fields:
            failures.append(f"{relative_path}: missing required fields {missing_fields}")
        if _should_rescore(expected, fields):
            rescore = _independent_rescore(path)
            checked[relative_path]["rescored_rows"] = rescore["rescored_rows"]
            for failure in rescore["failures"]:
                failures.append(str(Path(relative_path)) + failure.removeprefix(str(path)))
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "checked": checked,
    }


def _stale_claims(root: Path) -> list[str]:
    findings = []
    paper_paths = [
        root / "paper" / "evidencefirst_submission.tex",
        root / "paper" / "evidencefirst_current_results.tex",
    ]
    for path in paper_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_CLAIM_PATTERNS:
            if pattern in text:
                findings.append(f"{path.relative_to(root)} contains stale claim: {pattern}")
    return findings


def _stale_analysis_outputs(root: Path) -> list[str]:
    findings = []
    pairwise = root / "results" / "analysis" / "evidencefirst_stats" / "pairwise_mcnemar.csv"
    if pairwise.exists():
        text = pairwise.read_text(encoding="utf-8")
        required_rows = [
            "EvidenceFirst v6,HopRAG strict",
            "EvidenceFirst v6 local-context,HopRAG strict",
        ]
        for row_marker in required_rows:
            if row_marker not in text:
                findings.append(
                    f"{pairwise.relative_to(root)} appears stale; missing row marker: {row_marker}"
                )
    return findings


def _run(cmd: list[str], root: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _expand_artifact_paths(root: Path, artifact_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for relative_path in artifact_paths:
        if any(char in relative_path for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(str(root / relative_path))))
        else:
            paths.append(root / relative_path)
    return paths


def _forbidden_field_paths(value, forbidden: set[str], prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                findings.append(child_path)
            findings.extend(_forbidden_field_paths(child, forbidden, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            findings.extend(_forbidden_field_paths(child, forbidden, child_path))
    return findings


def _json_records_for_scan(path: Path) -> list:
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".json":
        data = _load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    return []


def verify_no_2wiki_source_only_fields(
    root: Path,
    failures: list[str],
    artifact_paths: list[str] | None = None,
) -> int:
    artifact_paths = artifact_paths or DEFAULT_2WIKI_INFERENCE_ARTIFACTS
    checked = 0
    for path in _expand_artifact_paths(root, artifact_paths):
        if not path.exists():
            failures.append(f"{path.relative_to(root)}: missing 2Wiki inference artifact")
            continue
        checked += 1
        for index, row in enumerate(_json_records_for_scan(path), start=1):
            field_paths = _forbidden_field_paths(row, FORBIDDEN_2WIKI_INFERENCE_FIELDS)
            for field_path in field_paths:
                if isinstance(row, dict):
                    qid = row.get("_id") or row.get("id") or row.get("qid") or index
                else:
                    qid = index
                failures.append(
                    f"{path.relative_to(root)}: source-only inference field {field_path} at row {index} qid={qid}"
                )
    return checked


def _assert_equal(actual, expected, label: str, failures: list[str]) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected}, got {actual}")


def _assert_close(actual: float, expected: float, label: str, failures: list[str]) -> None:
    if round(float(actual), 4) != round(float(expected), 4):
        failures.append(f"{label}: expected {expected:.4f}, got {float(actual):.4f}")


def _rows_by_key(rows: list[dict], fields: list[str]) -> dict[tuple, dict]:
    return {
        tuple(row.get(field) for field in fields): row
        for row in rows
    }


def verify_claim_manifest(root: Path, failures: list[str]) -> None:
    """Check the machine-readable paper-claim manifest against verifier targets."""
    manifest_path = root / "docs" / "experiments" / "paper_claims_manifest.json"
    if not manifest_path.exists():
        failures.append(f"missing claim manifest: {manifest_path.relative_to(root)}")
        return
    manifest = _load_json(manifest_path)
    _assert_equal(
        manifest.get("paper"),
        "paper/evidencefirst_submission.tex",
        "claim manifest paper",
        failures,
    )

    main_rows = _rows_by_key(
        manifest.get("main_metrics", []),
        ["dataset", "method"],
    )
    for key, (expected_em, expected_f1) in EXPECTED_MAIN.items():
        row = main_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing main metric: {key}")
            continue
        _assert_equal(str(row.get("em")), expected_em, f"claim manifest {key} EM", failures)
        _assert_equal(str(row.get("f1")), expected_f1, f"claim manifest {key} F1", failures)

    pairwise_rows = _rows_by_key(
        manifest.get("paired_statistics", []),
        ["dataset", "subgroup", "primary", "baseline"],
    )
    for key, expected in EXPECTED_PAIRWISE.items():
        row = pairwise_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing paired statistic: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest pairwise {key} n", failures)
        for field in ["em_delta", "f1_delta", "mcnemar_exact_p"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest pairwise {key} {field}", failures)

    protocol_rows = manifest.get("protocol_boundary", [])
    protocol_names = {row.get("row_group") for row in protocol_rows if isinstance(row, dict)}
    for expected in EXPECTED_PROTOCOL_BOUNDARY:
        if expected not in protocol_names:
            failures.append(f"claim manifest missing protocol boundary: {expected}")

    for section, expected_values in [
        ("support_summary", EXPECTED_SUPPORT),
        ("chain_validity_summary", EXPECTED_CHAIN),
        ("semantic_gap_adjudication", EXPECTED_ADJUDICATION),
        ("blind_gap_adjudication", EXPECTED_BLIND_ADJUDICATION),
    ]:
        values = manifest.get(section, {})
        for field, expected in expected_values.items():
            _assert_close(float(values[field]), expected, f"claim manifest {section} {field}", failures)

    context_rows = _rows_by_key(
        manifest.get("context_visibility", []),
        ["condition"],
    )
    for condition, expected in EXPECTED_CONTEXT_VISIBILITY.items():
        row = context_rows.get((condition,))
        if not row:
            failures.append(f"claim manifest missing context visibility: {condition}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest context {condition} n", failures)
        for field in [
            "support_title_recall",
            "evidence_entity_coverage",
            "gold_entity_coverage",
            "gold_pair_recall",
            "gold_relation_recall",
            "em",
            "f1",
        ]:
            if field in expected:
                _assert_close(
                    float(row[field]),
                    expected[field],
                    f"claim manifest context {condition} {field}",
                    failures,
                )

    overview_rows = _rows_by_key(
        manifest.get("audit_overview", []),
        ["dataset", "method"],
    )
    for key, expected in EXPECTED_AUDIT_OVERVIEW.items():
        row = overview_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing audit overview: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest audit overview {key} n", failures)
        for field in [
            "chain_complete_rate",
            "repair_attempt_rate",
            "repair_to_complete_rate",
            "selector_selected_rate",
        ]:
            _assert_close(float(row[field]), expected[field], f"claim manifest audit overview {key} {field}", failures)

    gap_rows = _rows_by_key(
        manifest.get("gap_labels", []),
        ["dataset", "gap_type"],
    )
    for key, expected in EXPECTED_GAP_LABELS.items():
        row = gap_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing gap label: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest gap {key} n", failures)
        for field in ["em", "f1", "consistent_rate"]:
            if field in expected:
                _assert_close(float(row[field]), expected[field], f"claim manifest gap {key} {field}", failures)

    utility_rows = _rows_by_key(
        manifest.get("diagnostic_utility", []),
        ["gap_type"],
    )
    for gap_type, expected in EXPECTED_DIAGNOSTIC_UTILITY.items():
        row = utility_rows.get((gap_type,))
        if not row:
            failures.append(f"claim manifest missing diagnostic utility: {gap_type}")
            continue
        for field in ["gold_proxy_precision", "semantic_match", "blind_match"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest diagnostic utility {gap_type} {field}", failures)

    blind_summary_path = root / "results/analysis/blind_multiagent_gap_adjudication/blind_gap_adjudication_summary.json"
    if not blind_summary_path.exists():
        failures.append(f"missing blind adjudication summary: {blind_summary_path.relative_to(root)}")
    else:
        blind = _load_json(blind_summary_path)
        _assert_equal(
            int(blind.get("sample_n", -1)),
            EXPECTED_BLIND_ADJUDICATION["sample_n"],
            "blind adjudication summary sample_n",
            failures,
        )
        blind_rates = blind.get("overall_rates", {})
        for actual_key, expected_key in [
            ("match", "match_rate"),
            ("mismatch", "mismatch_rate"),
            ("ambiguous", "ambiguous_rate"),
        ]:
            _assert_close(
                float(blind_rates.get(actual_key, -1)),
                EXPECTED_BLIND_ADJUDICATION[expected_key],
                f"blind adjudication summary {expected_key}",
                failures,
            )

    selection_rows = _rows_by_key(
        manifest.get("answer_selection", []),
        ["dataset"],
    )
    for dataset, expected in EXPECTED_ANSWER_SELECTION.items():
        row = selection_rows.get((dataset,))
        if not row:
            failures.append(f"claim manifest missing answer selection: {dataset}")
            continue
        for field in ["selected_em", "not_selected_em"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest answer selection {dataset} {field}", failures)

    risk_rows = _rows_by_key(
        manifest.get("risk_bins", []),
        ["dataset", "risk_bin"],
    )
    for key, expected in EXPECTED_RISK_BINS.items():
        row = risk_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing risk bin: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest risk bin {key} n", failures)
        _assert_close(float(row["em"]), expected["em"], f"claim manifest risk bin {key} EM", failures)

    triage_rows = _rows_by_key(
        manifest.get("triage_utility", []),
        ["dataset", "signal"],
    )
    for key, expected in EXPECTED_TRIAGE_UTILITY.items():
        row = triage_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing triage row: {key}")
            continue
        for field in ["error_auc", "topk_error_rate", "topk_lift"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest triage {key} {field}", failures)

    routing_rows = _rows_by_key(
        manifest.get("routing_queues", []),
        ["dataset", "queue"],
    )
    for key, expected in EXPECTED_ROUTING_QUEUES.items():
        row = routing_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing routing queue: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest routing queue {key} n", failures)
        for field in ["error_rate", "error_ci_low", "error_ci_high"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest routing queue {key} {field}", failures)

    routing_delta_rows = _rows_by_key(
        manifest.get("routing_deltas", []),
        ["dataset", "comparison"],
    )
    for key, expected in EXPECTED_ROUTING_DELTAS.items():
        row = routing_delta_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing routing delta: {key}")
            continue
        for field in ["delta_error", "delta_ci_low", "delta_ci_high"]:
            _assert_close(float(row[field]), expected[field], f"claim manifest routing delta {key} {field}", failures)

    ablation_rows = _rows_by_key(
        manifest.get("ablations", []),
        ["dataset", "variant"],
    )
    for key, expected in EXPECTED_ABLATIONS.items():
        row = ablation_rows.get(key)
        if not row:
            failures.append(f"claim manifest missing ablation: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"claim manifest ablation {key} n", failures)
        _assert_close(float(row["em"]), expected["em"], f"claim manifest ablation {key} EM", failures)
        _assert_close(float(row["f1"]), expected["f1"], f"claim manifest ablation {key} F1", failures)


def _read_csv_first(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} has no data rows") from exc


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _selected(row: dict) -> bool:
    if "evidence_first_postprocess_selected_v6" in row and str(
        row.get("evidence_first_postprocess_selected_v6")
    ) != "":
        return _bool_value(row.get("evidence_first_postprocess_selected_v6"))
    return _bool_value(row.get("evidence_first_postprocess_selected"))


def _ablation_variant(row: dict) -> str:
    value = row.get("evidence_first_ablation")
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        value = row.get("ablation")
    return str(value or "")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return center - half, center + half


def _newcombe_delta_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    low1, high1 = _wilson_ci(k1, n1)
    low2, high2 = _wilson_ci(k2, n2)
    low = (p2 - p1) - math.sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2)
    high = (p2 - p1) + math.sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2)
    return low, high


def _routing_queue_predicate(queue: str, row: dict) -> bool:
    answer_selected = _bool_value(row.get("answer_selected"))
    chain_incomplete = _bool_value(row.get("chain_incomplete"))
    residual_graph_gap = _bool_value(row.get("residual_graph_gap"))
    unrepaired_incomplete = _bool_value(row.get("unrepaired_incomplete"))
    if queue == "selected_checked":
        return answer_selected and not chain_incomplete
    if queue == "selected_unrepaired_incomplete":
        return answer_selected and unrepaired_incomplete
    if queue == "not_selected_no_residual":
        return (not answer_selected) and not residual_graph_gap
    if queue == "not_selected_residual_gap":
        return (not answer_selected) and residual_graph_gap
    return False


def verify_routing_queue_claims(root: Path, failures: list[str]) -> None:
    path = root / "results/analysis/audit_risk/audit_risk_per_example.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    observed: dict[tuple[str, str], tuple[int, int]] = {}
    for key, expected in EXPECTED_ROUTING_QUEUES.items():
        dataset, queue = key
        queue_rows = [
            row for row in rows
            if row["dataset"] == dataset and _routing_queue_predicate(queue, row)
        ]
        errors = sum(1 - int(float(row["em"])) for row in queue_rows)
        n = len(queue_rows)
        observed[key] = (errors, n)
        _assert_equal(n, expected["n"], f"routing queue {key} n", failures)
        error_rate = errors / n if n else 0.0
        low, high = _wilson_ci(errors, n)
        _assert_close(error_rate, expected["error_rate"], f"routing queue {key} error_rate", failures)
        _assert_close(low, expected["error_ci_low"], f"routing queue {key} error_ci_low", failures)
        _assert_close(high, expected["error_ci_high"], f"routing queue {key} error_ci_high", failures)

    delta_specs = {
        ("hotpot", "selected_unrepaired_minus_checked"): (
            ("hotpot", "selected_checked"),
            ("hotpot", "selected_unrepaired_incomplete"),
        ),
        ("2wiki", "not_selected_residual_minus_no_residual"): (
            ("2wiki", "not_selected_no_residual"),
            ("2wiki", "not_selected_residual_gap"),
        ),
    }
    for key, (base_key, target_key) in delta_specs.items():
        expected = EXPECTED_ROUTING_DELTAS[key]
        base_errors, base_n = observed[base_key]
        target_errors, target_n = observed[target_key]
        delta = target_errors / target_n - base_errors / base_n
        low, high = _newcombe_delta_ci(base_errors, base_n, target_errors, target_n)
        _assert_close(delta, expected["delta_error"], f"routing delta {key} delta_error", failures)
        _assert_close(low, expected["delta_ci_low"], f"routing delta {key} delta_ci_low", failures)
        _assert_close(high, expected["delta_ci_high"], f"routing delta {key} delta_ci_high", failures)


def verify_ablation_claims(root: Path, failures: list[str]) -> None:
    for key, relative_path in ABLATION_SUMMARY_FILES.items():
        expected = EXPECTED_ABLATIONS[key]
        row = _read_csv_first(root / relative_path)
        _assert_equal(int(row["n"]), expected["n"], f"ablation {key} n", failures)
        _assert_close(float(row["EM"]), expected["em"], f"ablation {key} EM", failures)
        _assert_close(float(row["F1"]), expected["f1"], f"ablation {key} F1", failures)

        score_rows = _load_jsonl(root / ABLATION_SCORE_FILES[key])
        scored_em = []
        scored_f1 = []
        expected_variant = key[1]
        for index, score_row in enumerate(score_rows, start=1):
            actual_variant = _ablation_variant(score_row)
            if actual_variant != expected_variant:
                failures.append(
                    f"ablation {key} row {index}: expected variant {expected_variant}, got {actual_variant}"
                )
            prediction = _prediction_for_rescore(score_row)
            gold = _gold_for_rescore(score_row)
            if prediction is None or gold is None:
                failures.append(f"ablation {key} row {index}: missing prediction/gold for rescore")
                continue
            scored_em.append(_em(prediction, gold))
            scored_f1.append(_f1(prediction, gold))

        _assert_equal(len(score_rows), expected["n"], f"ablation JSONL {key} n", failures)
        _assert_equal(len(scored_em), expected["n"], f"ablation JSONL {key} scored rows", failures)
        _assert_close(_mean(scored_em), expected["em"], f"ablation JSONL {key} EM", failures)
        _assert_close(_mean(scored_f1), expected["f1"], f"ablation JSONL {key} F1", failures)


def verify_context_visibility_claims(
    root: Path,
    failures: list[str],
    support_path: Path | None = None,
    chain_path: Path | None = None,
) -> None:
    support_path = support_path or root / "results/analysis/2wiki_support/2wiki_support_summary.json"
    chain_path = chain_path or root / "results/analysis/2wiki_chain_validity/2wiki_chain_validity_summary.json"
    support = _load_json(support_path)
    chain = _load_json(chain_path)
    summaries = {
        **{key: support[key] for key in support if isinstance(support.get(key), dict)},
        "overall": chain["overall"],
    }
    support_n = int(support["n"])
    chain_n = int(chain["n"])

    for condition, expected in EXPECTED_CONTEXT_VISIBILITY.items():
        summary_key = expected["summary_key"]
        row = summaries.get(summary_key)
        if row is None:
            failures.append(f"context visibility missing summary key {summary_key}: {condition}")
            continue
        expected_n = expected["n"]
        observed_n = chain_n if summary_key == "overall" else support_n
        _assert_equal(observed_n, expected_n, f"context visibility {condition} n", failures)
        for field in [
            "support_title_recall",
            "evidence_entity_coverage",
            "gold_entity_coverage",
            "gold_pair_recall",
            "gold_relation_recall",
            "em",
            "f1",
        ]:
            if field in expected:
                _assert_close(
                    float(row[field]),
                    expected[field],
                    f"context visibility {condition} {field}",
                    failures,
                )


def verify_answer_selection_claims(root: Path, failures: list[str]) -> None:
    paths = {
        "hotpot": root
        / "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl",
        "2wiki": root / "results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl",
    }
    for dataset, path in paths.items():
        rows = _load_jsonl(path)
        selected_em = [float(row.get("em") or 0) for row in rows if _selected(row)]
        not_selected_em = [float(row.get("em") or 0) for row in rows if not _selected(row)]
        expected = EXPECTED_ANSWER_SELECTION[dataset]
        _assert_close(
            _mean(selected_em),
            expected["selected_em"],
            f"{dataset} selected-answer EM",
            failures,
        )
        _assert_close(
            _mean(not_selected_em),
            expected["not_selected_em"],
            f"{dataset} non-selected-answer EM",
            failures,
        )


def verify_extended_paper_claims(
    stats: dict,
    chain: dict,
    adjudication: dict,
    failures: list[str],
    risk: dict | None = None,
    root: Path | None = None,
) -> None:
    pairwise = {
        (row["dataset"], row["subgroup"], row["primary"], row["baseline"]): row
        for row in stats.get("pairwise_mcnemar", [])
    }
    for key, expected in EXPECTED_PAIRWISE.items():
        row = pairwise.get(key)
        if not row:
            failures.append(f"missing pairwise row: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"pairwise {key} n", failures)
        for field in ["em_delta", "f1_delta", "mcnemar_exact_p"]:
            _assert_close(float(row[field]), expected[field], f"pairwise {key} {field}", failures)

    overview = {
        (row["dataset"], row["method"]): row
        for row in stats.get("audit_overview", [])
    }
    for key, expected in EXPECTED_AUDIT_OVERVIEW.items():
        row = overview.get(key)
        if not row:
            failures.append(f"missing audit overview row: {key}")
            continue
        _assert_equal(int(row["n"]), expected["n"], f"audit overview {key} n", failures)
        for field in [
            "chain_complete_rate",
            "repair_attempt_rate",
            "repair_to_complete_rate",
            "selector_selected_rate",
        ]:
            _assert_close(float(row[field]), expected[field], f"audit overview {key} {field}", failures)

    by_gap = {
        (row["dataset"], row["gap_type"]): row
        for row in stats.get("audit_by_gap_type", [])
    }
    consistency = {
        (row["dataset"], row["gap_type"]): row
        for row in stats.get("audit_label_consistency", [])
    }
    for key, expected in EXPECTED_GAP_LABELS.items():
        row = by_gap.get(key)
        label_row = consistency.get(key)
        if not row:
            failures.append(f"missing audit gap row: {key}")
        else:
            _assert_equal(int(row["n"]), expected["n"], f"audit gap {key} n", failures)
            if "em" in expected:
                _assert_close(float(row["em"]), expected["em"], f"audit gap {key} EM", failures)
            if "f1" in expected:
                _assert_close(float(row["f1"]), expected["f1"], f"audit gap {key} F1", failures)
        if not label_row:
            failures.append(f"missing audit label-consistency row: {key}")
        else:
            _assert_close(
                float(label_row["consistent_rate"]),
                expected["consistent_rate"],
                f"audit label {key} consistency",
                failures,
            )

    chain_label_rows = {
        row["gap_type"]: row
        for row in chain.get("gap_label_gold_audit", [])
    }
    adjudication_rows = adjudication.get("by_gap_type", {})
    blind_rows = {}
    if root is not None:
        blind_summary_path = root / "results/analysis/blind_multiagent_gap_adjudication/blind_gap_adjudication_summary.json"
        if blind_summary_path.exists():
            blind_rows = _load_json(blind_summary_path).get("by_gap_type", {})
        else:
            failures.append(f"missing blind adjudication summary: {blind_summary_path.relative_to(root)}")
    for gap_type, expected in EXPECTED_DIAGNOSTIC_UTILITY.items():
        chain_row = chain_label_rows.get(gap_type)
        adjudication_row = adjudication_rows.get(gap_type)
        blind_row = blind_rows.get(gap_type)
        if not chain_row:
            failures.append(f"missing diagnostic utility gold-proxy row: {gap_type}")
        else:
            _assert_close(
                float(chain_row["precision"]),
                expected["gold_proxy_precision"],
                f"diagnostic utility {gap_type} gold-proxy precision",
                failures,
            )
        if not adjudication_row:
            failures.append(f"missing diagnostic utility semantic row: {gap_type}")
        else:
            _assert_close(
                float(adjudication_row["match_rate"]),
                expected["semantic_match"],
                f"diagnostic utility {gap_type} semantic match",
                failures,
            )
        if not blind_row:
            failures.append(f"missing diagnostic utility blind row: {gap_type}")
        else:
            _assert_close(
                float(blind_row["match_rate"]),
                expected["blind_match"],
                f"diagnostic utility {gap_type} blind match",
                failures,
            )

    if risk is not None:
        risk_rows = {
            (row["dataset"], row["risk_bin"]): row
            for row in risk.get("summary", [])
        }
        for key, expected in EXPECTED_RISK_BINS.items():
            row = risk_rows.get(key)
            if not row:
                failures.append(f"missing audit-risk row: {key}")
                continue
            _assert_equal(int(row["n"]), expected["n"], f"audit-risk {key} n", failures)
            _assert_close(float(row["em"]), expected["em"], f"audit-risk {key} EM", failures)
        triage_rows = {
            (row["dataset"], row["signal"]): row
            for row in risk.get("triage_utility", [])
        }
        for key, expected in EXPECTED_TRIAGE_UTILITY.items():
            row = triage_rows.get(key)
            if not row:
                failures.append(f"missing audit triage row: {key}")
                continue
            for field in ["error_auc", "topk_error_rate", "topk_lift"]:
                _assert_close(
                    float(row[field]),
                    expected[field],
                    f"audit triage {key} {field}",
                    failures,
                )

    if root is not None:
        verify_context_visibility_claims(root, failures)
        verify_ablation_claims(root, failures)
        verify_answer_selection_claims(root, failures)
        verify_routing_queue_claims(root, failures)


def verify_outputs(out_dir: Path) -> dict:
    failures: list[str] = []
    verify_claim_manifest(_repo_root(), failures)

    stats = _load_json(out_dir / "evidencefirst_stats" / "summary.json")
    metrics = {
        (row["dataset"], row["method"]): row
        for row in stats["metric_ci"]
    }
    for key, (expected_em, expected_f1) in EXPECTED_MAIN.items():
        row = metrics.get(key)
        if not row:
            failures.append(f"missing main metric row: {key}")
            continue
        _assert_equal(row["em"], expected_em, f"{key} EM", failures)
        _assert_equal(row["f1"], expected_f1, f"{key} F1", failures)

    support = _load_json(out_dir / "2wiki_support" / "2wiki_support_summary.json")
    support_main = support["evidencefirst_reader_full"]
    for key, expected in EXPECTED_SUPPORT.items():
        _assert_close(support_main[key], expected, f"2Wiki support {key}", failures)

    chain = _load_json(out_dir / "2wiki_chain_validity" / "2wiki_chain_validity_summary.json")
    chain_main = chain["overall"]
    for key, expected in EXPECTED_CHAIN.items():
        _assert_close(chain_main[key], expected, f"2Wiki chain {key}", failures)

    verify_context_visibility_claims(
        _repo_root(),
        failures,
        support_path=out_dir / "2wiki_support" / "2wiki_support_summary.json",
        chain_path=out_dir / "2wiki_chain_validity" / "2wiki_chain_validity_summary.json",
    )

    adjudication = _load_json(
        out_dir / "semantic_gap_adjudication" / "semantic_gap_adjudication_summary.json"
    )
    _assert_equal(
        adjudication["sample_n"],
        EXPECTED_ADJUDICATION["sample_n"],
        "semantic adjudication sample_n",
        failures,
    )
    adjudication_main = adjudication["overall"]
    for key in ["match_rate", "mismatch_rate", "ambiguous_rate"]:
        _assert_close(
            adjudication_main[key],
            EXPECTED_ADJUDICATION[key],
            f"semantic adjudication {key}",
            failures,
        )
    _assert_close(
        adjudication_main["match_wilson_ci95"][0],
        EXPECTED_ADJUDICATION["match_ci_low"],
        "semantic adjudication match_ci_low",
        failures,
    )
    _assert_close(
        adjudication_main["match_wilson_ci95"][1],
        EXPECTED_ADJUDICATION["match_ci_high"],
        "semantic adjudication match_ci_high",
        failures,
    )
    _assert_close(
        adjudication["inter_agent_agreement"]["fleiss_kappa"],
        EXPECTED_ADJUDICATION["fleiss_kappa"],
        "semantic adjudication fleiss_kappa",
        failures,
    )

    risk_path = out_dir / "audit_risk" / "audit_risk_summary.json"
    risk = _load_json(risk_path) if risk_path.exists() else None
    verify_extended_paper_claims(stats, chain, adjudication, failures, risk=risk, root=_repo_root())

    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "checked": {
            "main_metric_rows": len(stats["metric_ci"]),
            "pairwise_rows": len(stats["pairwise_mcnemar"]),
            "2wiki_support_n": support["n"],
            "2wiki_chain_n": chain["n"],
            "semantic_adjudication_n": adjudication["sample_n"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/reviewer_verify"))
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--check-only", action="store_true", help="Only check required artifact presence")
    args = parser.parse_args()

    root = _repo_root()
    missing = _missing_paths(root)
    if missing:
        print(json.dumps({"status": "missing_artifacts", "missing": missing}, indent=2))
        raise SystemExit(2)
    stale_claims = _stale_claims(root)
    if stale_claims:
        print(json.dumps({"status": "stale_claims", "findings": stale_claims}, indent=2))
        raise SystemExit(2)
    stale_analysis = _stale_analysis_outputs(root)
    if stale_analysis:
        print(json.dumps({"status": "stale_analysis_outputs", "findings": stale_analysis}, indent=2))
        raise SystemExit(2)
    claim_manifest_failures: list[str] = []
    verify_claim_manifest(root, claim_manifest_failures)
    if claim_manifest_failures:
        print(json.dumps({
            "status": "claim_manifest_failed",
            "failures": claim_manifest_failures,
        }, indent=2))
        raise SystemExit(2)
    routing_failures: list[str] = []
    verify_routing_queue_claims(root, routing_failures)
    if routing_failures:
        print(json.dumps({
            "status": "routing_queue_claims_failed",
            "failures": routing_failures,
        }, indent=2))
        raise SystemExit(2)
    context_failures: list[str] = []
    verify_context_visibility_claims(root, context_failures)
    if context_failures:
        print(json.dumps({
            "status": "context_visibility_claims_failed",
            "failures": context_failures,
        }, indent=2))
        raise SystemExit(2)
    source_only_failures: list[str] = []
    source_only_checked = verify_no_2wiki_source_only_fields(root, source_only_failures)
    if source_only_failures:
        print(json.dumps({
            "status": "2wiki_source_only_fields_failed",
            "checked_files": source_only_checked,
            "failures": source_only_failures,
        }, indent=2))
        raise SystemExit(2)
    integrity = verify_artifact_integrity(root)
    if integrity["status"] != "pass":
        print(json.dumps({"status": "artifact_integrity_failed", **integrity}, indent=2))
        raise SystemExit(2)
    if args.check_only:
        print(json.dumps({
            "status": "pass",
            "checked_artifact_files": len(REQUIRED_FILES),
            "required_pattern_matches": _pattern_match_counts(root),
            "integrity_checked_files": len(integrity["checked"]),
            "2wiki_source_only_checked_files": source_only_checked,
            "claim_manifest": "docs/experiments/paper_claims_manifest.json",
            **_rescore_summary(integrity),
        }, indent=2))
        return

    out_dir = args.out_dir
    _run([
        sys.executable,
        "scripts/analyze_evidencefirst_stats.py",
        "--out-dir",
        str(out_dir / "evidencefirst_stats"),
        "--bootstrap-iters",
        str(args.bootstrap_iters),
    ], root)
    _run([
        sys.executable,
        "scripts/analyze_2wiki_evidence_support.py",
        "--out-dir",
        str(out_dir / "2wiki_support"),
    ], root)
    _run([
        sys.executable,
        "scripts/analyze_2wiki_chain_validity.py",
        "--out-dir",
        str(out_dir / "2wiki_chain_validity"),
    ], root)
    _run([
        sys.executable,
        "scripts/run_semantic_gap_adjudication.py",
        "--chain",
        str(out_dir / "2wiki_chain_validity" / "2wiki_chain_validity_per_example.csv"),
        "--support",
        str(out_dir / "2wiki_support" / "2wiki_support_per_example.csv"),
        "--out-dir",
        str(out_dir / "semantic_gap_adjudication"),
        "--sample-size",
        "100",
        "--seed",
        "20260608",
    ], root)
    _run([
        sys.executable,
        "scripts/analyze_audit_risk.py",
        "--out-dir",
        str(out_dir / "audit_risk"),
        "--figure",
        str(out_dir / "audit_risk" / "audit_risk_strata.pdf"),
    ], root)

    result = verify_outputs(root / out_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
