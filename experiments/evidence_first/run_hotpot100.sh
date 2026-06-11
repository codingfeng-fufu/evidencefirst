#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

N="${N:-100}"
TAG="${TAG:-hotpotqa_ef_${N}}"
OUT_PREFIX="experiments/evidence_first/results/hotpotqa_evidence_first_${N}"

python evaluate.py \
  --mode full \
  --variant evidence_first \
  --n "$N" \
  --out "${OUT_PREFIX}.csv" \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out "${OUT_PREFIX}.jsonl" \
  --usage-log "${OUT_PREFIX}_usage.jsonl" \
  --cache-tag "$TAG"

python experiments/evidence_first/summarize_evidencefirst.py \
  "${OUT_PREFIX}.jsonl" \
  --csv "${OUT_PREFIX}.csv"
