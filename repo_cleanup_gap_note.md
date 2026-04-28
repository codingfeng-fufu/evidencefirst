# Repo Cleanup Gap Note

This repository was cleaned toward the target structure in
[repo_cleanup_for_claudecode.md](repo_cleanup_for_claudecode.md),
but a few items were intentionally left as compatibility-preserving exceptions.

## What Was Cleaned

- Added top-level reproducibility entrypoints:
  - [README.md](README.md)
  - [evaluate.py](evaluate.py)
  - [build_kg.py](build_kg.py)
  - [requirements.txt](requirements.txt)
  - [.env.example](.env.example)
  - [.gitignore](.gitignore)
- Added top-level reproducibility folders:
  - [data/](data/)
  - [prompts/](prompts/)
  - [results/](results/)
  - [scripts/](scripts/)
- Added standardized per-example CSV export tooling.
- Added lightweight module-alignment helpers:
  - [comagraag/context.py](comagraag/context.py)
  - [comagraag/entity_linker.py](comagraag/entity_linker.py)
  - [comagraag/kg_builder.py](comagraag/kg_builder.py)

## What Was Not Fully Migrated

### 1. `comagraag/agents.py` was not split into `comagraag/agents/`

Reason:

- The live codepath still imports `from agents import ...` from multiple places
  inside the experimental workspace.
- A hard split right before submission would require a broad import refactor
  across evaluation, baselines, rerun utilities, and diagnostics.
- That refactor is possible, but it has a higher risk of breaking the validated
  experimental pipeline.

Current compromise:

- Keep [comagraag/agents.py](comagraag/agents.py) as the authoritative implementation
- Export prompts and helper wrappers around it
- Document the gap rather than forcing a risky package split

### 2. The current GRA is not prompt-driven

Reason:

- The live `run_gra` implementation uses BM25 seed finding over KG triples plus
  BFS expansion.
- It does not currently call an LLM prompt in the same way as QDA, AGA, and VA.

Current compromise:

- [prompts/gra_prompt.txt](prompts/gra_prompt.txt) documents the archived / conceptual
  retrieval prompt specification used for supplementary-material alignment.
- The README explicitly notes that the current GRA implementation is prompt-free.

### 3. Historical experimental workspace remains under `comagraag/`

Reason:

- Existing rerun artifacts, caches, and exploratory scripts were preserved to
  avoid invalidating already generated paper results.
- The cleaned top-level surface is intended for reproducibility; the internal
  `comagraag/` workspace remains the historical record.

Current compromise:

- Use the top-level entrypoints and scripts for public-facing reproducibility.
- Treat `comagraag/` as the underlying experimental workspace.

### 4. Environment variable naming differs slightly from the target memo

Reason:

- The live code reads `LLM_MODEL` and `LLM_BASE_URL`, not `OPENAI_MODEL`.

Current compromise:

- [.env.example](.env.example) uses the actual variable names that the current code expects.

### 5. HotpotQA paper table vs surviving caches are not fully aligned

Reason:

- The surviving HotpotQA per-example caches in `comagraag/results/` currently
  recompute to:
  - `full`: `EM 0.370 / F1 0.442`
  - `no_verif`: `EM 0.374 / F1 0.445`
  - `no_decomp`: `EM 0.370 / F1 0.433`
- These do not match the paper-reported table values:
  - `full`: `0.442 / 0.522`
  - `no_verif`: `0.422 / 0.505`
  - `no_decomp`: `0.430 / 0.504`

Current compromise:

- The top-level standardized CSVs in [results/](results/) were exported from the
  surviving caches, not from unrecoverable paper-only summary numbers.
- The README keeps the paper-reported table for citation consistency, but
  explicitly notes the reproducibility mismatch.
- No hidden or synthetic per-example rows were fabricated to force agreement.

## Recommended Future Cleanup

If there is time for a non-paper-critical refactor after submission:

1. Split `comagraag/agents.py` into a package
2. Convert all internal imports to package-relative imports
3. Move data-generation logic fully to the top-level `data/`
4. Retire or archive historical root-level planning notes into a `docs/` folder
