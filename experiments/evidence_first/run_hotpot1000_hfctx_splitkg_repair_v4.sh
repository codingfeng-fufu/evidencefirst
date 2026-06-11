#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_RUN="${BASE_RUN:-hotpotqa_evidence_first_hfctx_n1000_splitkg_v3}"
REPAIR_RUN="${REPAIR_RUN:-hotpotqa_evidence_first_hfctx_n1000_splitkg_v4_repair}"
OUT_RUN="${OUT_RUN:-hotpotqa_evidence_first_hfctx_n1000_splitkg_v4}"
DATA="${DATA:-experiments/evidence_first/data/hotpotqa_1000_hf_context.json}"
KG="${KG:-comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl}"
RESULT_DIR="${RESULT_DIR:-experiments/evidence_first/results}"
CACHE_DIR="${CACHE_DIR:-experiments/evidence_first/cache}"
LOG_DIR="${LOG_DIR:-experiments/evidence_first/logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES

mkdir -p "$RESULT_DIR" "$CACHE_DIR" "$LOG_DIR"

echo "[$(date -Is)] starting repair run"
echo "base_run=$BASE_RUN"
echo "repair_run=$REPAIR_RUN"
echo "out_run=$OUT_RUN"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

pids=()
for pair in "5:625" "6:750" "7:875"; do
  batch="${pair%%:*}"
  start="${pair##*:}"
  tag="${REPAIR_RUN}_b${batch}"
  prefix="${RESULT_DIR}/${REPAIR_RUN}_batch${batch}"
  log="${LOG_DIR}/${REPAIR_RUN}_batch${batch}.log"
  echo "[$(date -Is)] launch repair batch=$batch start=$start tag=$tag log=$log"
  python evaluate.py --mode full --variant evidence_first --n 125 --start "$start" \
    --data "$DATA" \
    --kg "$KG" \
    --out "${prefix}.csv" \
    --out-dir "$CACHE_DIR" \
    --save-pipeline-details \
    --jsonl-out "${prefix}.jsonl" \
    --usage-log "${prefix}_usage.jsonl" \
    --cache-tag "$tag" \
    > "$log" 2>&1 &
  pids+=("$!")
  sleep 3
done

fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    echo "[$(date -Is)] repair worker=$idx pid=$pid finished"
  else
    rc=$?
    echo "[$(date -Is)] repair worker=$idx pid=$pid failed rc=$rc"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[$(date -Is)] repair batch process failed; skipping merge"
  exit 1
fi

BASE_RUN="$BASE_RUN" REPAIR_RUN="$REPAIR_RUN" OUT_RUN="$OUT_RUN" RESULT_DIR="$RESULT_DIR" python - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
base_run = os.environ["BASE_RUN"]
repair_run = os.environ["REPAIR_RUN"]
out_run = os.environ["OUT_RUN"]

rows = []
usage_rows = []

for batch in range(5):
    rows_path = result_dir / f"{base_run}_batch{batch}.jsonl"
    usage_path = result_dir / f"{base_run}_batch{batch}_usage.jsonl"
    rows.extend(json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip())
    usage_rows.extend(json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip())

repair_oom = 0
for batch in (5, 6, 7):
    rows_path = result_dir / f"{repair_run}_batch{batch}.jsonl"
    usage_path = result_dir / f"{repair_run}_batch{batch}_usage.jsonl"
    batch_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    repair_oom += sum(1 for row in batch_rows if "out of memory" in str(row.get("error", "")).lower())
    rows.extend(batch_rows)
    usage_rows.extend(json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip())

if repair_oom:
    raise RuntimeError(f"Repair run still has CUDA OOM rows: {repair_oom}")

out_prefix = result_dir / out_run
out_prefix.with_suffix(".jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
(out_prefix.parent / f"{out_prefix.name}_usage.jsonl").write_text(
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

with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)

print("merged", summary)
PY

echo "[$(date -Is)] completed repair run"
