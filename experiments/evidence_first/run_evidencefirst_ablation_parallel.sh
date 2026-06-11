#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-hotpotqa}"
RESULT_DIR="${RESULT_DIR:-experiments/evidence_first/results}"
CACHE_DIR="${CACHE_DIR:-experiments/evidence_first/cache}"
LOG_DIR="${LOG_DIR:-experiments/evidence_first/logs}"
WORKERS="${WORKERS:-2}"
LAUNCH_SLEEP="${LAUNCH_SLEEP:-3}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES-}"
ABLATIONS="${ABLATIONS:-without_verification without_repair without_reader_context without_answer_refinement}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

case "$DATASET" in
  hotpotqa|hotpot)
    DATASET="hotpotqa"
    RUN_BASE="${RUN_BASE:-hotpotqa_evidencefirst_v6_ablation}"
    DATA="${DATA:-experiments/evidence_first/data/hotpotqa_1000_hf_context.json}"
    KG="${KG:-comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl}"
    TOTAL="${TOTAL:-1000}"
    ;;
  2wiki)
    RUN_BASE="${RUN_BASE:-2wiki_evidencefirst_v6_ablation}"
    DATA="${DATA:-comagraag/data/2wiki_sample.json}"
    KG="${KG:-comagraag/data/2wiki_kgs.pkl}"
    TOTAL="${TOTAL:-500}"
    ;;
  *)
    echo "Unsupported DATASET=$DATASET; expected hotpotqa or 2wiki" >&2
    exit 2
    ;;
esac

if [ -n "$RUN_SUFFIX" ]; then
  RUN_BASE="${RUN_BASE}_${RUN_SUFFIX}"
fi

BATCH_SIZE="${BATCH_SIZE:-$(((TOTAL + WORKERS - 1) / WORKERS))}"

mkdir -p "$RESULT_DIR" "$CACHE_DIR" "$LOG_DIR"

echo "[$(date -Is)] starting ablations"
echo "dataset=$DATASET data=$DATA kg=$KG"
echo "run_base=$RUN_BASE total=$TOTAL workers=$WORKERS batch_size=$BATCH_SIZE"
echo "ablations=$ABLATIONS"
echo "run_suffix=${RUN_SUFFIX:-<none>}"
echo "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-<empty>}"

for ablation in $ABLATIONS; do
  run="${RUN_BASE}_${ablation}"
  prefix="$RESULT_DIR/$run"
  echo "[$(date -Is)] starting ablation=$ablation run=$run"

  pids=()
  batch_ids=()
  for b in $(seq 0 $((WORKERS - 1))); do
    start=$((b * BATCH_SIZE))
    if [ "$start" -ge "$TOTAL" ]; then
      continue
    fi
    n="$BATCH_SIZE"
    if [ $((start + n)) -gt "$TOTAL" ]; then
      n=$((TOTAL - start))
    fi

    tag="${run}_b${b}"
    batch_prefix="${prefix}_batch${b}"
    batch_log="${LOG_DIR}/${run}_batch${b}.log"

    echo "[$(date -Is)] launch ablation=$ablation batch=$b start=$start n=$n tag=$tag log=$batch_log"
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
      python evaluate.py --dataset "$DATASET" --mode full --variant evidence_first \
        --ablation "$ablation" --n "$n" --start "$start" \
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
    batch_ids+=("$b")
    sleep "$LAUNCH_SLEEP"
  done

  fail=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    batch="${batch_ids[$idx]}"
    if wait "$pid"; then
      echo "[$(date -Is)] ablation=$ablation batch=$batch pid=$pid finished"
    else
      rc=$?
      echo "[$(date -Is)] ablation=$ablation batch=$batch pid=$pid failed rc=$rc"
      fail=1
    fi
  done

  if [ "$fail" -ne 0 ]; then
    echo "[$(date -Is)] ablation=$ablation failed; skipping merge" >&2
    exit 1
  fi

  BATCH_IDS="$(IFS=,; echo "${batch_ids[*]}")" \
  RUN="$run" ABLATION="$ablation" RESULT_DIR="$RESULT_DIR" python - <<'PY'
import csv
import json
import os
from pathlib import Path

run = os.environ["RUN"]
ablation = os.environ["ABLATION"]
result_dir = Path(os.environ["RESULT_DIR"])
batch_ids = [int(x) for x in os.environ["BATCH_IDS"].split(",") if x]
prefix = result_dir / run

rows = []
usage_rows = []
for b in batch_ids:
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
    "ablation": ablation,
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

  echo "[$(date -Is)] completed ablation=$ablation run=$run"
done

echo "[$(date -Is)] completed all ablations for dataset=$DATASET"
