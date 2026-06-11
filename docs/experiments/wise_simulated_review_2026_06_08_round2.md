# WISE Simulated Review Round 2

Date: 2026-06-08

Status note: this round has been superseded by
`docs/experiments/wise_simulated_review_2026_06_08_round3.md`. In particular,
HopRAG strict on 2Wiki-500 later completed with 0.5640 EM and 0.6157 F1, so the
round-2 note that 2Wiki lacked a completed HopRAG baseline is no longer current.

This file records the second simulated WISE review pass requested by the user.
Four independent subagents reviewed the active EvidenceFirst submission from
different reviewer roles: contribution/novelty, experiments/statistics,
artifact reproducibility, and senior/meta review.

## Simulated Scores

| Reviewer role | Simulated score | Confidence |
|---|---:|---:|
| Contribution and novelty | Weak Reject | 4/5 |
| Experiments and statistics | Weak Reject | 4/5 |
| Artifact reproducibility | 2.5/5, borderline not ready | 4/5 |
| Senior/meta reviewer | Borderline Reject / Weak Reject | 4/5 |

## Meta-Review Summary

The paper is plausible for WISE as a Web-style GraphRAG reliability/system
paper, but the acceptance risk remains high unless the claim is sharply framed
as auditable structural evidence-state instrumentation with competitive
accuracy. The empirical evidence does not prove semantic chain verification or
monotonic answer-quality improvement from the verifier/repair component.

## Consensus Blockers

1. The term "verification" can overstate the evidence because the chain labels
   are structural, not semantic proof of gold support.
2. S2G-RAG, FAIR-RAG, and RAGChecker are close enough that related work must
   explicitly distinguish text-level sufficiency/gap judging from graph-state
   diagnostics.
3. Full-set accuracy gains are descriptive; paired confidence intervals cross
   zero for the main HotpotQA and 2Wiki comparisons.
4. The verifier+repair ablation is mixed or negative on HotpotQA answer metrics,
   so traceability must be evaluated and presented as the core contribution.
5. Artifact reproduction depends on ignored generated files; a source-only
   checkout is insufficient for paper-table verification.
6. HotpotQA completed-run comparisons are not context-budget isolated because
   the reported EvidenceFirst pipeline can append global BM25 context.
7. The 2Wiki reader-full support audit is a context-visibility audit, not a
   retrieval-quality leaderboard.

## Fixes Applied In This Pass

- Renamed the active paper framing toward "structural evidence-state
  diagnostics" and "fixed-context Web-style GraphRAG."
- Added S2G-RAG and RAGChecker bibliography entries and related-work discussion.
- Rewrote the positioning table to include sufficiency/gap-judging and
  diagnostic-evaluation systems.
- Added a context-budget boundary paragraph and a new `--no-global-context`
  evaluation switch for future strict reruns.
- Added `scripts/verify_paper_artifacts.py` as the official no-LLM artifact
  verification entrypoint.
- Added `docs/experiments/reviewer_artifact_manifest.md` to define the required
  reviewer artifact bundle.
- Removed an oracle-sensitive "gold answer" phrase from the answer
  canonicalizer prompt.
- Added missing direct dependencies for tests/runtime helpers: `loguru` and
  `pytest`.

## Remaining Risks

- A manual semantic gap-label adjudication on 50-100 examples is still the
  strongest missing evidence.
- Superseded: 2Wiki later gained a completed HopRAG strict baseline; MS GraphRAG
  remains missing as a full-sample 2Wiki baseline.
- The paper still relies on completed-artifact comparisons rather than fresh,
  context-budget-controlled reruns.
- The active claim should stay bounded to operational auditability plus
  competitive accuracy, not semantic verification or leaderboard dominance.
