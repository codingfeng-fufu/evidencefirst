# EvidenceFirst Ablation Results and Writing Plan

Date: 2026-06-08

## Goal

Report a compact, defensible ablation study for EvidenceFirst. The completed
study uses four component ablations on two datasets:

- HotpotQA-1000
- 2WikiMultiHopQA-500

All 8 ablation runs completed under the run suffix `20260608_004220`.

Main artifact prefix:

- `experiments/evidence_first/results/*evidencefirst_v6_ablation_20260608_004220_*`

## Reporting Boundary

HotpotQA has one main model: EvidenceFirst v6. Earlier Hotpot development
attempts are process artifacts and should not be mixed into the main result
table. They can only be used when they cleanly correspond to a named ablation
or refinement analysis.

The ablation section should make a different claim from the baseline section:

- Baseline tables compare EvidenceFirst v6 against external systems.
- Ablation tables explain which EvidenceFirst components matter for answer
  quality and which ones mainly support traceability.
- Refinement analysis can separately explain v4 -> v5 -> v6, but it is not a
  substitute for component ablation.

## Ablated Core Components

### 1. w/o Evidence-Chain Verification

Disable the pre-generation chain completeness check. The system retrieves graph
evidence and then generates without verifying whether the evidence forms a
complete multi-hop chain.

Purpose:

- Tests the paper's central claim that structural evidence state should be
  checked before generation.
- Current answer-level result is mixed: Hotpot improves slightly without this
  switch, while 2Wiki is slightly lower. Therefore this component should be
  framed as diagnostic traceability, not as a guaranteed EM/F1 boost.

Main comparison:

- Full v6 vs `w/o chain check + repair`

Ablation setting:

- The completed switch skips `EvidenceChainChecker.check_evidence_chain...`
  and gap-based diagnostics before generation.
- Keep the same retrieved triples, local context, generator, and answer
  refinement policy as the full model where possible.

### 2. w/o Graph Repair

Keep structural chain checking and diagnostics, but disable localized repair
for missing entities, disconnected pairs, and bridge gaps.

Purpose:

- Tests whether graph-level diagnostics are useful operationally, not just for
  logging.
- Current answer-level result shows a clearer EM drop on both datasets, but
  Hotpot F1 is roughly tied/slightly higher. Frame as helpful but interacting
  with reader context and answer selection.

Main comparison:

- Full v6 vs `w/o graph repair`

Ablation setting:

- Run chain completeness checks normally.
- Do not call A+ missing-entity enhancement or B disconnected-entity repair.
- Generate from the original retrieved evidence and local context.

### 3. w/o Full/Local Context Reader

Disable the reader's access to full local passages and rely on graph triples or
the structurally checked evidence chain only.

Purpose:

- Tests whether structurally checked graph evidence alone is enough, or whether local text
  grounding is necessary for answer surface forms.
- Observed effect is large on both datasets, especially on 2Wiki, confirming
  that graph triples alone are not enough for robust answer surface forms.

Main comparison:

- Full v6 vs `w/o full/local context reader`

Related 2Wiki context variants:

- `results/wise_2wiki_evidencefirst_v6_metrics.csv`
- `results/wise_2wiki_evidencefirst_v6_localctx2_metrics.csv`
- `results/wise_2wiki_evidencefirst_v6_readerfull_metrics.csv`
- `results/wise_2wiki_evidencefirst_v6_readerfull_canon_metrics.csv`

Ablation setting:

- The completed ablation disables passage/full-context reader support in the
  v6 pipeline.
- The related 2Wiki variants above are useful for interpretation but should not
  replace the explicit `w/o reader context` ablation row.

### 4. w/o Answer Refinement

Disable deterministic final-answer cleanup and selector guard.

Purpose:

- Tests how much of the final score comes from output-level normalization and
  answer-type guarding rather than evidence retrieval/structural checking.
- Current result is dataset-dependent: it helps Hotpot, while on 2Wiki it
  preserves EM and increases token F1 when disabled. Treat as answer-surface
  behavior rather than universal quality gain.

Main comparison:

- Full v6 vs `w/o answer refinement`

## Completed Results

### Main Ablation Table

Report one combined table with dataset columns:

| Variant | HotpotQA-1000 EM | HotpotQA-1000 F1 | 2Wiki-500 EM | 2Wiki-500 F1 |
| --- | ---: | ---: | ---: | ---: |
| Full EvidenceFirst v6 | 0.5620 | 0.6564 | 0.6900 | 0.7581 |
| w/o chain check + repair | 0.5680 | 0.6647 | 0.6780 | 0.7579 |
| w/o graph repair | 0.5540 | 0.6580 | 0.6500 | 0.7424 |
| w/o full/local context reader | 0.2360 | 0.2899 | 0.1840 | 0.2289 |
| w/o answer refinement | 0.5510 | 0.6485 | 0.6900 | 0.7782 |

Notes:

- `w/o full/local context reader` is the strongest supported accuracy ablation.
- `w/o answer refinement` is helpful on Hotpot but not on 2Wiki F1.
- `w/o chain check + repair` and `w/o graph repair` should be discussed
  carefully: they support the diagnostic/traceability story more than a simple
  monotonic answer-quality story.
- These rows use explicit ablation switches, not legacy `no_verif/no_decomp`
  files.

### Optional Refinement Analysis Table

This table is optional and should not replace component ablation:

| Hotpot Variant | N | EM | F1 | Meaning |
| --- | ---: | ---: | ---: | --- |
| v4 fresh end-to-end | 1000 | 0.5500 | 0.6484 | Fresh EvidenceFirst run |
| v5 offline clean | 1000 | 0.5570 | 0.6507 | Deterministic answer cleanup |
| v6 selector guard | 1000 | 0.5620 | 0.6564 | Answer-type mismatch guard |

This table can go in the appendix or a short analysis paragraph if space allows.

## Interpretation

The results do not support a blanket claim that every EvidenceFirst component
independently improves answer EM/F1. The safest paper claim is:

1. EvidenceFirst's core innovation is pre-generation evidence-chain
   verification: the evidence graph is checked before answer generation.
2. Graph-structural diagnostics are methodologically important because they
   expose missing entities, disconnected pairs, bridge gaps, repair attempts,
   and fallback cases.
3. Answer-quality gains are strongest when the verified graph path is paired
   with full/local reader context.
4. Repair and verification are useful for traceability and selective
   intervention, but current answer-level gains are mixed and should not be
   oversold.

## Implementation Notes

The completed implementation uses explicit ablation controls in the
EvidenceFirst pipeline rather than unrelated historical modes:

- `disable_evidence_verification`
- `disable_graph_repair`
- `disable_reader_context`
- `disable_answer_refinement`

The existing legacy `full/no_verif/no_decomp` Hotpot 500-row files should not
be used as final v6 ablations. They can be cited only as old diagnostic runs if
needed:

- `results/hotpotqa_full.csv`
- `results/hotpotqa_no_verif.csv`
- `results/hotpotqa_no_decomp.csv`

These legacy files are not aligned with final EvidenceFirst v6 and should not
enter the main ablation table.

## Reporting Checks

For every reported ablation run:

1. Confirm sample count matches the planned dataset.
2. Confirm EM/F1 were computed with the same `scripts/compute_metrics.py`
   normalization or the same metrics path as the full model.
3. Confirm the intended component is actually disabled by inspecting per-row
   metadata or aggregate counters.
4. Save both metrics CSV and per-example JSONL.
5. Add a short artifact mapping paragraph to the paper before reporting the
   numbers.

## Paper Integration

Main text:

- Include the combined ablation table.
- Keep one paragraph on verification and repair.
- Keep one paragraph on reader context and answer refinement.

Appendix:

- Optional v4 -> v5 -> v6 refinement table.
- Optional legacy 500-row diagnostic table only if it helps explain early
  architecture decisions.
