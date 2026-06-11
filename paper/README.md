# Paper Draft Assets

This folder contains LaTeX fragments, tables, bibliography material, and WISE
paper draft assets that were previously stored at the repository root.

They are kept separate from runtime code and reproducibility scripts.

## Current Drafts

- `evidencefirst_submission.tex`: active submission-oriented EvidenceFirst
  draft. It uses the conservative WISE-facing story: v4 is the fresh HotpotQA
  run, v6 is a deterministic selector-guard pass over the same predictions,
  2Wiki is reported over completed artifacts, and paired/audit diagnostics come
  from `results/analysis/evidencefirst_stats/`.
- `evidencefirst_current_results.tex`: earlier current-results draft generated
  from completed artifacts available on 2026-06-07. Keep it as a historical
  snapshot rather than the active submission source.
- `evidencefirst_draft.tex`: earlier EvidenceFirst narrative draft.
- `main_wise.tex`: earlier CoMaGRAG/Wise-oriented draft. It follows the legacy
  multi-agent CoMaGRAG story and should not be used as the active EvidenceFirst
  submission text.

The active submission maps each main table row to saved artifacts in the
Experimental Setup section and can be checked with the scripts documented in
the repository root README.
