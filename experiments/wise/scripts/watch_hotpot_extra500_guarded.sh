#!/usr/bin/env bash
set -euo pipefail

cd /home/u2023312337/CoMaGRAG

CURRENT="results/wise/hotpot_extra500_comagraag_full_predictions.jsonl"
CACHE="results/cache_wise_hotpot_extra500_comagraag_full_full.json"
PARENT_PID="${1:-86263}"

while true; do
  lines=0
  if [[ -f "$CURRENT" ]]; then
    lines=$(wc -l < "$CURRENT")
  fi

  if [[ "$lines" -ge 500 ]]; then
    echo "$(date '+%F %T') current full complete: ${lines}/500; starting guarded candidates"
    env PYTHONUNBUFFERED=1 python scripts/run_guarded_adjudication_candidates.py \
      --current "$CURRENT" \
      --cache "$CACHE" \
      --out-judge-jsonl results/wise/hotpot_extra500_comagraag_candidate_judge_v2_all_predictions.jsonl \
      --out-prior-jsonl results/wise/hotpot_extra500_comagraag_contextual_prior_v2_predictions.jsonl \
      --out-guarded-jsonl results/wise/hotpot_extra500_comagraag_guarded_judge_contextual_prior_v2_predictions.jsonl \
      --out-csv results/wise_hotpot_extra500_comagraag_guarded_judge_contextual_prior_v2.csv \
      --usage-log results/wise/hotpot_extra500_comagraag_guarded_judge_contextual_prior_v2_usage.jsonl \
      --expected-n 500
    exit $?
  fi

  if ! ps -p "$PARENT_PID" >/dev/null 2>&1; then
    echo "$(date '+%F %T') comagraag parent exited before current full completed: ${lines}/500"
    exit 2
  fi

  echo "$(date '+%F %T') waiting for current full: ${lines}/500"
  sleep 60
done
