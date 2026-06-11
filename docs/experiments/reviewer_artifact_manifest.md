# Reviewer Artifact Manifest

This manifest defines the WISE-facing, no-LLM reproduction boundary for the
active EvidenceFirst submission. It is intended for an artifact archive, not a
minimal source-only checkout.

Reviewer-facing field definitions are in
`docs/experiments/artifact_data_dictionary.md`. The 100-example offline proxy
gap-state plausibility audit protocol is in
`docs/experiments/gap_label_adjudication.md`. The separate blind multi-agent
reviewer-style packet is summarized in
`docs/experiments/blind_multiagent_gap_adjudication.md`.
Machine-readable paper claim values are in
`docs/experiments/paper_claims_manifest.json`; the verifier checks regenerated
summaries against this manifest.

## Official No-LLM Check

Run this from the repository root:

```bash
python scripts/verify_paper_artifacts.py \
    --out-dir results/analysis/reviewer_verify
```

The script checks that required saved artifacts exist, verifies pinned
SHA256/size/row-count/schema fingerprints for the saved bundle, independently
rescoring saved prediction/gold rows where per-example EM/F1 is reported,
including the full-sample ablation JSONL files. It then recomputes the paper's
main metric summaries, recomputes the 2Wiki support/evidence audit, recomputes
the 2Wiki gold-chain proxy audit, runs the 100-example offline proxy gap-state
plausibility audit, recomputes audit-risk strata, and verifies the reported
main, paired, ablation, diagnostic, and audit values.

Use a faster smoke check with fewer bootstrap samples:

```bash
python scripts/verify_paper_artifacts.py \
    --out-dir results/analysis/reviewer_verify_smoke \
    --bootstrap-iters 200
```

Use a presence-only check before running analysis:

```bash
python scripts/verify_paper_artifacts.py --check-only
```

The presence-only mode still performs artifact fingerprint, schema, row-count,
2Wiki source-field hygiene checks, and independent per-example EM/F1 rescoring
checks; it only skips recomputing derived aggregate summaries. In the current
bundle, this quick check reports 40 required artifact files, 44 integrity
fingerprints, 12 checked 2Wiki inference/cache files, and 17,500 independently
rescored rows.

## Reviewer Quick Check

For a sub-minute reviewer sanity check, run:

```bash
python scripts/verify_paper_artifacts.py --check-only
```

This checks `docs/experiments/paper_claims_manifest.json`, required artifact
presence, required glob matches, pinned SHA256/size/row-count/schema
fingerprints, and independent per-example EM/F1 rescoring for scored artifacts.
It does not rerun aggregate analyses, rebuild KGs, call LLM APIs, or rerun
external baselines.

## Required Bundle Contents

The artifact archive must include generated files that are intentionally ignored
by the source repository's `.gitignore`:

- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v4.jsonl`
- `docs/experiments/paper_claims_manifest.json`
- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl`
- `results/wise_hotpot1000_hoprag_strict_response_metrics.csv`
- `results/wise_hotpot1000_arag_full_post_combined_metrics.csv`
- `results/wise/hotpot1000_*_combined_predictions.jsonl`
- `results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl`
- `results/wise/2wiki_evidencefirst_v6_localctx2_predictions.jsonl`
- `results/wise/2wiki_lightrag_predictions.jsonl`
- `results/wise/2wiki_hoprag_strict_predictions.jsonl`
- `results/wise_2wiki_hoprag_strict_metrics.csv`
- `results/wise/2wiki_arag_full_post_predictions.jsonl`
- `external_runs/2wiki500/arag/chunks.json`
- `results/wise/2wiki_ircot_predictions.jsonl`
- `results/wise/2wiki_naive_rag_predictions.jsonl`
- `results/wise_2wiki_arag_full_post_metrics.csv`
- `results/wise/2wiki_evidencefirst_v6_kgs.pkl`
- `results/cache_wise_2wiki_evidencefirst_v6_readerfull_s100_full.json`
- `results/cache_wise_2wiki_evidencefirst_v6_readerfull_b*_full.json`
- `comagraag/data/2wiki_sample.json`
- `experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_*.csv`
- `experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_*.csv`
- `experiments/evidence_first/results/hotpotqa_evidencefirst_v6_ablation_20260608_004220_without_*.jsonl`
- `experiments/evidence_first/results/2wiki_evidencefirst_v6_ablation_20260608_004220_without_*.jsonl`
- `results/analysis/blind_multiagent_gap_adjudication/protocol.md`
- `results/analysis/blind_multiagent_gap_adjudication/packet_*.jsonl`
- `results/analysis/blind_multiagent_gap_adjudication/blind_gap_adjudication_summary.json`

The verification script is the authoritative list if this document and code
ever diverge.

Files whose names end in `_metrics.csv` in this bundle are scored per-example
outputs with `qid`, `prediction`, `gold`, `em`, and `f1` columns unless a
specific document says otherwise. Aggregate reviewer summaries are regenerated
under the selected verification output directory, for example
`evidencefirst_stats/main_metric_ci.csv` and
`evidencefirst_stats/pairwise_mcnemar.csv`.

## Recomputed Reviewer Outputs

The no-LLM verification entrypoint writes the following reviewer-facing
summaries under the selected `--out-dir`:

- `evidencefirst_stats/main_metric_ci.csv`
- `evidencefirst_stats/pairwise_mcnemar.csv`
- `evidencefirst_stats/audit_label_consistency.csv`
- `2wiki_support/2wiki_support_per_example.csv`
- `2wiki_chain_validity/2wiki_chain_validity_per_example.csv`
- `2wiki_chain_validity/2wiki_gap_label_gold_audit.csv`
- `semantic_gap_adjudication/semantic_gap_adjudication_2wiki100.csv`
- `semantic_gap_adjudication/semantic_gap_adjudication_2wiki100.jsonl`
- `semantic_gap_adjudication/semantic_gap_adjudication_summary.json`
- `audit_risk/audit_risk_per_example.csv`
- `audit_risk/audit_risk_summary.csv`
- `audit_risk/audit_risk_summary.json`
- `audit_risk/audit_triage_utility.csv`
- `audit_risk/conditional_selection_graph_state.csv`

These outputs are generated from the required bundle. They do not require fresh
LLM calls.

## Claim-To-Source Map

| Paper claim / table | Source artifacts | Verifier output |
| --- | --- | --- |
| HotpotQA and 2Wiki main metric rows | `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_*.jsonl`; `results/wise/*predictions.jsonl`; `results/wise_*metrics.csv` | `evidencefirst_stats/main_metric_ci.csv` |
| Paired deltas and McNemar tests | Shared question IDs from the main metric artifacts | `evidencefirst_stats/pairwise_mcnemar.csv` |
| Controlled ablations | `experiments/evidence_first/results/*ablation_20260608_004220_without_*.csv`; `experiments/evidence_first/results/*ablation_20260608_004220_without_*.jsonl` | Aggregate rows checked and per-example JSONL rows independently rescored by `verify_paper_artifacts.py` |
| 2Wiki evidence visibility audit | `results/wise/2wiki_evidencefirst_v6_readerfull_canon_predictions.jsonl`; `comagraag/data/2wiki_sample.json`; `external_runs/2wiki500/arag/chunks.json`; `results/analysis/2wiki_support/2wiki_support_summary.json` | `2wiki_support/2wiki_support_summary.json` |
| 2Wiki saved-KG gold-chain audit | `results/wise/2wiki_evidencefirst_v6_kgs.pkl`; `comagraag/data/2wiki_sample.json` | `2wiki_chain_validity/2wiki_chain_validity_summary.json` |
| 100-example proxy plausibility audit | Generated support and chain validity per-example outputs | `semantic_gap_adjudication/semantic_gap_adjudication_summary.json` |
| Audit-risk strata and triage utility | Saved \sys prediction JSONL files and recomputed audit fields | `audit_risk/audit_risk_summary.json` |
| Conditional routing utility | Saved \sys prediction JSONL files with answer-selection and graph-state fields | `audit_risk/conditional_selection_graph_state.csv` |
| Machine-readable paper claim boundary | `docs/experiments/paper_claims_manifest.json` | Checked before both `--check-only` and full verification |

## Protocol Boundary Map

| Row group | Context source | Global context | Posthoc transform | Purpose |
| --- | --- | --- | --- | --- |
| HotpotQA EvidenceFirst end-to-end | HotpotQA local passages plus pipeline BM25 augmentation from the evaluation context pool | Allowed | None beyond saved answer normalization | Main completed EvidenceFirst run |
| HotpotQA EvidenceFirst selector variant | Same saved source artifacts as HotpotQA EvidenceFirst end-to-end | Allowed | Deterministic selector; no fresh LLM calls | Sensitivity row for answer selection and cleanup |
| 2Wiki EvidenceFirst reader-full | Dataset-provided 2Wiki passages with full reader visibility for the saved sample | No external global pool | Deterministic canonicalizer over saved predictions | Main 2Wiki saved artifact |
| 2Wiki EvidenceFirst local-context stress | Local 2Wiki context only, without reader-full visibility | No | Same scoring/canonical boundary as saved artifact | Negative stress row for context-budget sensitivity |
| 2Wiki visibility audit | Posthoc 2Wiki support titles, evidence entities, A-RAG read chunks, and saved KG state | No inference access | No-generation support/entity coverage analysis | Context-budget and evidence-visibility audit |
| Naive RAG / IRCoT / A-RAG | Completed baseline artifacts generated from the same benchmark samples | Method-specific | Baseline-specific saved postprocessing only | Retrieval and iterative/agentic baselines |
| LightRAG / MS GraphRAG | Completed graph-baseline artifacts on available full-sample rows | Method-specific | Baseline-specific saved postprocessing only | GraphRAG comparison rows; MS GraphRAG is incomplete on 2Wiki |
| HopRAG strict | Completed strict response-rescored artifacts on HotpotQA and 2Wiki | Method-specific | Strict answer extraction and independent QA rescoring | Closest graph traversal/pruning baseline |

## HopRAG 2Wiki Integration Note

The 2Wiki HopRAG strict row is verified as a completed saved baseline artifact,
not as a context-budget-matched rerun. The official scored files are:

- `results/wise/2wiki_hoprag_strict_predictions.jsonl`
- `results/wise_2wiki_hoprag_strict_metrics.csv`

The prediction file contains 500 rows with 500 unique 2Wiki question ids, and
the metrics CSV contains the same 500 examples plus a header row. The external
run directory `external_runs/hoprag_2wiki500_strict/` preserves the 3346 2Wiki
document bodies, HopRAG cache outputs, and logs. Its two offline workers each
report `docs=1673`, matching the 3346 document-body total. The upstream HopRAG
configuration string in the logs still prints `dataset_name: hotpot`; this is a
configuration-name artifact, while the scored qids and document bodies are from
the 2Wiki sample. The offline logs also contain 14 document-build/parser error
lines; the reported paper row therefore uses completed HopRAG predictions that
were independently strict-rescored, not a claim that every HopRAG document-build
attempt succeeded.

## What This Verifies

The no-LLM check verifies saved-artifact claims only:

- HotpotQA end-to-end saved run: `0.5500 EM / 0.6484 F1`
- HotpotQA deterministic selector variant: `0.5620 EM / 0.6564 F1`
- HopRAG strict HotpotQA baseline: `0.5410 EM / 0.6438 F1`
- 2Wiki EvidenceFirst reader-full: `0.6900 EM / 0.7581 F1`
- 2Wiki EvidenceFirst local-context stress: `0.5040 EM / 0.5945 F1`
- 2Wiki LightRAG baseline: `0.6740 EM / 0.7716 F1`
- 2Wiki HopRAG strict baseline: `0.5640 EM / 0.6157 F1`
- 2Wiki reader-full support-title recall: `1.0000`
- Naive BM25 top-5 support-title recall / evidence-entity coverage:
  `0.6020 / 0.6685`
- A-RAG actual-read support-title recall / evidence-entity coverage:
  `0.2955 / 0.3326`
- A-RAG search-found upper-bound support-title recall / evidence-entity
  coverage: `0.5885 / 0.6085`
- 2Wiki saved-KG gold relation recall: `0.0420`
- 100-example offline proxy gap-state plausibility audit: `0.4000` match rate,
  Wilson 95% CI `[0.3094, 0.4980]`, `0.3100` mismatch rate, `0.2900`
  ambiguous rate, and Fleiss kappa `0.1836`
- separate 100-case blind reviewer-style packet: `0.5300` match rate,
  `0.3600` mismatch rate, and `0.1100` ambiguous rate; by-label blind match
  rates are `complete=0.6765`, `missing_entities=0.5714`,
  `short_chain=0.2000`, and `disconnected=0.5455`
- controlled ablation rows in the ablation table, including `w/o reader context +
  refinement`: HotpotQA `0.2360 EM / 0.2899 F1` and 2Wiki
  `0.1840 EM / 0.2289 F1`, with 6000 ablation JSONL rows independently
  rescored during the quick artifact check
- paired deterministic statistics in Table 6, including the HotpotQA bridge
  delta against HopRAG strict and the 2Wiki local-context stress delta
- diagnostic utility rows in Table 8, including 2Wiki `missing_entities`
  structural consistency `1.0000`, gold-proxy precision `0.6866`,
  proxy-audit match `0.7429`, and blind match `0.5714`
- audit-risk strata used as a no-LLM error-risk sanity check, for example
  HotpotQA risk-bin `0` EM `0.6581` versus risk-bin `4+` EM `0.1154`
- audit triage utility against simple no-LLM signals: composite audit-risk AUC
  `0.6321` on HotpotQA and `0.5787` on 2Wiki, compared with chain-incomplete
  AUC `0.5321` and `0.5144`; top-20% high-risk error rates are `0.6700` on
  HotpotQA and `0.4000` on 2Wiki

It does not rebuild KGs, rerun LLM generation, or rerun external baselines.

The proxy plausibility audit is a three-proxy-view audit over saved artifacts,
and the blind packet is a reviewer-style multi-agent simulation. Neither is a
professional human annotation study. Their main value is exposing that
structural audit states are only partially semantically supported, especially
outside `missing_entities`.

## Fresh-Rerun Boundary

Fresh runs require API credentials, model availability, generated KG caches, and
external baseline dependencies. They may not reproduce exact paper numbers
because KG extraction and model APIs can drift. Use fresh runs to demonstrate
pipeline functionality; use the no-LLM check above to verify paper tables.

For future context-budget-controlled reruns, the evaluation entrypoint supports:

```bash
python evaluate.py --variant evidence_first --no-global-context ...
```

The current reported HotpotQA artifacts use the original pipeline setting,
which can append global BM25 passages for pipeline modes. The paper treats the
main tables as completed-artifact comparisons rather than a context-budget
isolated retrieval benchmark.
