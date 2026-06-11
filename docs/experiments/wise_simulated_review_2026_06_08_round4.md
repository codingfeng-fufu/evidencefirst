# WISE-Style Simulated Review Round 4

Date: 2026-06-08

## Setup

This round reran the WISE-style simulated review after addressing the round-3
and first round-4 rejection risks. Four reviewer agents independently reviewed
the active submission:

- Reviewer R1: method novelty and soundness.
- Reviewer R2: experiments, statistics, and artifacts.
- Reviewer R3: writing, positioning, and WISE fit.
- Reviewer R4: systems and reproducibility.

The reviewed source is `paper/evidencefirst_submission.tex`. The active paper
now explicitly treats EvidenceFirst as a pre-reader structural audit layer for
fixed-context GraphRAG, not as a retrieval-budget leaderboard system or semantic
proof checker.

## Final Simulated Scores

| Reviewer | Focus | Score | Confidence |
| --- | --- | --- | --- |
| R1 | Method and soundness | Weak Accept | 0.70 |
| R2 | Experiments and artifacts | Weak Accept | 4/5 |
| R3 | Writing and positioning | Weak Accept | 4/5 |
| R4 | Systems and reproducibility | Weak Accept | 0.75 |

Meta-outcome: **Weak Accept**.

No reviewer naturally gave Accept. R3 noted that an Accept is possible from a
reviewer who strongly values auditable systems and artifact-backed
reproducibility, but R1/R2/R4 all kept the score at Weak Accept because the
strict empirical story remains bounded.

## Fixes Applied Before This Round

- Added a 2Wiki local-context stress result to the paper:
  - `EvidenceFirst v6 reader-full canonical`: 0.6900 EM / 0.7581 F1.
  - `EvidenceFirst v6 local-context stress`: 0.5040 EM / 0.5945 F1.
  - Paired local-context stress vs. HopRAG strict:
    EM delta -0.0600, 95% CI [-0.1100, -0.0120], McNemar p=0.021574.
- Added a diagnostic utility table reporting per-label structural consistency,
  gold-proxy precision, and 100-example proxy-adjudication match.
- Reframed the 100-example semantic gap-label adjudication as negative
  calibration rather than semantic validation.
- Removed remaining leaderboard-style phrasing from the abstract, introduction,
  results, and discussion.
- Added artifact hygiene checks in `scripts/verify_paper_artifacts.py` for stale
  analysis outputs.
- Updated the reviewer artifact manifest and README to include the local-context
  stress artifact and clarify that `*_metrics.csv` files are scored per-example
  outputs, not aggregate-only summaries.

## Verification

Commands run successfully:

```bash
python scripts/verify_paper_artifacts.py --check-only
python scripts/verify_paper_artifacts.py \
    --out-dir results/analysis/reviewer_verify_after_stress \
    --bootstrap-iters 1000
python -m pytest tests/test_analysis_scripts.py tests/test_evaluate_pipeline_inputs.py -q
cd paper && pdflatex -interaction=nonstopmode -halt-on-error evidencefirst_submission.tex
```

Verification status:

- `verify_paper_artifacts.py --check-only`: pass, 20 required artifact files.
- Full no-LLM verifier: pass, 15 metric rows, 7 paired rows, 500 2Wiki support
  rows, 500 2Wiki chain rows, and 100 semantic-adjudication rows.
- Tests: 13 passed.
- PDF compile: pass. Remaining warnings are a standard `amsmath` warning and a
  0.9pt overfull hbox in an existing positioning table.

## Remaining Risks

1. The 2Wiki main result is a reader-full completed artifact and depends on
   strong context visibility; the local-context stress row is lower than
   completed graph baselines.
2. Chain diagnostics and repair are supported as audit/control mechanisms, not
   as the main EM/F1 driver.
3. Diagnostic labels are structural states. Their semantic calibration is weak
   outside `missing_entities`, especially for `complete`, `disconnected`, and
   `short_chain`.
4. Accept-level evidence would require either a full strict context-budget
   comparison with stable gains or professional/manual semantic gap-label
   adjudication showing reliable evidence-sufficiency calibration.

## Decision

The current draft satisfies the user goal of reaching at least **Weak Accept**
in simulated WISE review. It does not satisfy the optional stronger target of
obtaining a natural **Accept** review.
