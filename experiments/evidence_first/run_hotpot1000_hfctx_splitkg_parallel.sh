#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN="${RUN:-hotpotqa_evidence_first_hfctx_n1000_splitkg_v1}"
DATA="${DATA:-experiments/evidence_first/data/hotpotqa_1000_hf_context.json}"
KG="${KG:-comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl}"
RESULT_DIR="${RESULT_DIR:-experiments/evidence_first/results}"
CACHE_DIR="${CACHE_DIR:-experiments/evidence_first/cache}"
LOG_DIR="${LOG_DIR:-experiments/evidence_first/logs}"
WORKERS="${WORKERS:-8}"
TOTAL="${TOTAL:-1000}"
BATCH_SIZE="${BATCH_SIZE:-125}"
LAUNCH_SLEEP="${LAUNCH_SLEEP:-3}"

mkdir -p "$RESULT_DIR" "$CACHE_DIR" "$LOG_DIR"

PREFIX="$RESULT_DIR/$RUN"

echo "[$(date -Is)] starting $RUN"
echo "data=$DATA"
echo "kg=$KG"
echo "workers=$WORKERS total=$TOTAL batch_size=$BATCH_SIZE launch_sleep=$LAUNCH_SLEEP"

pids=()
for b in $(seq 0 $((WORKERS - 1))); do
  start=$((b * BATCH_SIZE))
  n=$BATCH_SIZE
  if [ "$b" -eq $((WORKERS - 1)) ]; then
    n=$((TOTAL - start))
  fi

  tag="${RUN}_b${b}"
  batch_prefix="${PREFIX}_batch${b}"
  batch_log="${LOG_DIR}/${RUN}_batch${b}.log"

  echo "[$(date -Is)] launch batch=$b start=$start n=$n tag=$tag log=$batch_log"
  python evaluate.py --mode full --variant evidence_first --n "$n" --start "$start" \
    --data "$DATA" \
    --kg "$KG" \
    --out "${batch_prefix}.csv" \
    --out-dir "$CACHE_DIR" \
    --save-pipeline-details \
    --jsonl-out "${batch_prefix}.jsonl" \
    --usage-log "${batch_prefix}_usage.jsonl" \
    --cache-tag "$tag" \
    > "$batch_log" 2>&1 &

  pids+=("$!")
  sleep "$LAUNCH_SLEEP"
done

fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    echo "[$(date -Is)] batch=$idx pid=$pid finished"
  else
    rc=$?
    echo "[$(date -Is)] batch=$idx pid=$pid failed rc=$rc"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[$(date -Is)] at least one batch failed; skipping merge"
  exit 1
fi

RUN="$RUN" RESULT_DIR="$RESULT_DIR" WORKERS="$WORKERS" python - <<'PY'
import csv
import json
import os
from pathlib import Path

run = os.environ["RUN"]
result_dir = Path(os.environ["RESULT_DIR"])
workers = int(os.environ["WORKERS"])
prefix = result_dir / run

rows = []
usage_rows = []
for b in range(workers):
    jsonl_path = result_dir / f"{run}_batch{b}.jsonl"
    usage_path = result_dir / f"{run}_batch{b}_usage.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    if not usage_path.exists():
        raise FileNotFoundError(usage_path)
    rows.extend(
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    usage_rows.extend(
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

prefix.with_suffix(".jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
(prefix.parent / f"{prefix.name}_usage.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in usage_rows),
    encoding="utf-8",
)

n = len(rows)
summary = {
    "mode": "full",
    "variant": "evidence_first",
    "max_iter": 3,
    "n": n,
    "EM": round(sum(float(row.get("em", 0.0)) for row in rows) / n, 4) if n else 0,
    "F1": round(sum(float(row.get("f1", 0.0)) for row in rows) / n, 4) if n else 0,
    "avg_iter": round(sum(float(row.get("iterations", 0.0)) for row in rows) / n, 2) if n else 0,
    "fresh_n": sum(1 for row in rows if not row.get("cache_hit")),
    "avg_llm_calls": round(sum(float(row.get("llm_calls", 0.0)) for row in rows) / n, 2) if n else 0,
    "avg_input_tokens": round(sum(float(row.get("input_tokens", 0.0)) for row in rows) / n, 2) if n else 0,
    "avg_output_tokens": round(sum(float(row.get("output_tokens", 0.0)) for row in rows) / n, 2) if n else 0,
    "avg_latency_s": round(sum(float(row.get("wall_time", 0.0)) for row in rows) / n, 4) if n else 0,
}

with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)

print("merged", summary)
PY

echo "[$(date -Is)] completed $RUN"
