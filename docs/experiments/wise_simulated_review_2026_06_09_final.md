# WISE-Style Simulated Review Final Round

Date: 2026-06-09

## Setup

This round used three fresh reviewer agents with no prior conversation context.
Each reviewer was given the current submission files and reviewer-facing
artifacts, including `paper/evidencefirst_submission.tex`,
`paper/evidencefirst_submission.pdf`,
`docs/experiments/reviewer_artifact_manifest.md`,
`docs/experiments/paper_claims_manifest.json`, and
`scripts/verify_paper_artifacts.py`.

The review target is the WISE 2026 Knowledge and RAG on Web track. The paper is
the submission-oriented version in `paper/evidencefirst_submission.tex`.

## Scores

| Reviewer | Focus | Score | Boundary |
| --- | --- | --- | --- |
| R1 | Track fit and novelty | 6/10 | Borderline to weak accept |
| R2 | Experiments, fairness, statistics | 7/10 | Weak accept |
| R3 | Writing, reproducibility, formatting | 7/10 | Weak accept / borderline accept |

Meta-outcome: **borderline-positive / weak accept**.

## Main Positive Signals

- The paper fits the track as a Web Knowledge-RAG / GraphRAG systems paper about
  evidence-state control, auditability, and replayable artifacts.
- The strongest contribution is the pre-reader evidence-state contract: checked,
  repaired, residual, and fallback states control repair, fallback, routing,
  answer selection, and replay.
- The artifact package is stronger than typical submissions: a claim manifest,
  reviewer artifact manifest, pinned fingerprints, independent rescoring, 2Wiki
  source-field checks, and a no-LLM verifier.
- The new 2Wiki evidence-visibility audit materially improves fairness
  disclosure by quantifying reader-full support visibility, local top-k support
  visibility, A-RAG read-chunk visibility, and saved-KG coverage.

## Remaining Risks

- The main acceptance risk is protocol comparability. The 2Wiki reader-full row
  is a context-rich audit condition, not a uniform-budget leaderboard row.
- Novelty is strongest as a systems contract and audit/control layer, not as a
  new retrieval or reasoning algorithm.
- Diagnostic labels are structural metadata. They are useful for routing and
  replay, but the paper must not imply semantic proof of evidence correctness.
- Ablation evidence supports observability and routing more than broad answer
  accuracy improvement.

## Post-Review Edits Applied

- The abstract and introduction now pair the 2Wiki reader-full audit result with
  the stricter 2Wiki local-context stress boundary.
- The unnecessary `WeKGR` shorthand was removed from the introduction.
- Table 3 now uses `saved run` instead of `fresh saved run`.
- The deployment-validity paragraph now uses a less defensive phrase,
  `artifact boundary`, instead of `bounded by four artifact facts`.

## Verification Gate

The final verification pass for this revision used:

```bash
python scripts/verify_paper_artifacts.py --check-only
cd paper && pdflatex -interaction=nonstopmode -halt-on-error evidencefirst_submission.tex && \
    bibtex evidencefirst_submission && \
    pdflatex -interaction=nonstopmode -halt-on-error evidencefirst_submission.tex && \
    pdflatex -interaction=nonstopmode -halt-on-error evidencefirst_submission.tex
pdfinfo paper/evidencefirst_submission.pdf
```

Observed status:

- Quick verifier: pass, with 40 required artifact files, 44 integrity
  fingerprints, 12 2Wiki source-field files, and 17,500 independently rescored
  rows.
- PDF: 15-page A4 LNCS output.
- LaTeX log: no undefined references/citations and no overfull boxes. The known
  remaining messages are the existing `amsmath` `\vec` warning and underfull
  table/spacing warnings.
