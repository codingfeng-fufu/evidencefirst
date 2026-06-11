# WISE-Style Simulated Review Round 3

Date: 2026-06-08

## Setup

This round used three parallel reviewer agents to approximate a WISE-style
review process:

- Reviewer A: empirical validity and baseline completeness.
- Reviewer B: semantic gap-label adjudication.
- Reviewer C: writing, contribution framing, and reviewer-facing clarity.

The reviewers inspected the paper drafts, artifact verifier, 2Wiki support
diagnostics, semantic gap-label adjudication outputs, and the completed HopRAG
2Wiki run.

## Main Findings

1. HopRAG strict on 2Wiki-500 has completed and must no longer be described as
   running or pending. The completed artifact reports 0.5640 EM and 0.6157 F1
   over 500 examples.
2. The paper should not frame the method as semantic evidence-chain
   verification. The safer claim is pre-generation structural evidence-state
   diagnostics/auditing.
3. The current-results draft needed related-work positioning against GraphRAG,
   HopRAG, S2G-RAG, FAIR-RAG, and RAGChecker.
4. HotpotQA accuracy claims require statistical and context-budget caveats
   because the full-set paired interval crosses zero and the saved protocols are
   not context-budget-isolated.
5. The 100-example semantic gap-label adjudication should weaken rather than
   strengthen semantic claims. Overall match is 0.4000 and Fleiss kappa is
   0.1836, with 72% proxy disagreement.
6. The HotpotQA v6 selector-guard provenance was underspecified. It is a
   deterministic post-processing chain over the v5 offline-clean artifact, not
   the unchanged v4 prediction file.

## Fixes Applied

- Added stable 2Wiki HopRAG strict artifacts:
  - `results/wise_2wiki_hoprag_strict_metrics.csv`
  - `results/wise/2wiki_hoprag_strict_predictions.jsonl`
- Added HopRAG strict to 2Wiki tables in the paper.
- Added HopRAG strict to `scripts/analyze_evidencefirst_stats.py`,
  `scripts/verify_paper_artifacts.py`, README reproduction notes, and the
  reviewer artifact manifest.
- Added verifier stale-claim checks for obsolete text saying 2Wiki HopRAG is
  still running or missing.
- Reframed `paper/evidencefirst_current_results.tex` around structural
  evidence-state diagnostics rather than semantic verification.
- Added related-work positioning to `paper/evidencefirst_current_results.tex`.
- Added context-budget/statistical caveats to the current-results draft.
- Added the 100-example proxy adjudication and the 72% disagreement signal to
  the paper.
- Corrected `paper/evidencefirst_submission.tex` to describe the HotpotQA v6
  selector guard as a deterministic v4 -> v5 -> v6 post-processing chain.

## Remaining Reviewer Risks

- MS GraphRAG still has no completed 2Wiki-500 run.
- HotpotQA remains a saved-protocol artifact comparison rather than a
  context-budget-isolated retrieval benchmark.
- The semantic gap-label study is a reproducible proxy audit, not professional
  human annotation.
- Structural labels should be used as audit states, not as semantic correctness
  certificates.
