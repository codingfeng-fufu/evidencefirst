#!/bin/bash
# Monitor Evidence Augmentation (Dual-Path) parallel evaluation

echo "=========================================="
echo "  Evidence Aug Parallel Evaluation Monitor"
echo "=========================================="
echo ""

# Count running processes
running=$(ps aux | grep "evaluate.py.*dual_path_batch" | grep -v grep | wc -l)
echo "Running processes: $running/8"
echo ""

# Check each batch (mode=full, not full_dual_path)
echo "Batch Progress:"
echo "----------------------------------------"
total=0
for i in {0..7}; do
    cache="results/cache_dual_path_batch_${i}_full.json"
    if [ -f "$cache" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$cache'))))" 2>/dev/null || echo "0")
        pct=$((count * 100 / 125))
        printf "  Batch %d: %3d/125 (%3d%%) " $i $count $pct
        # Progress bar
        filled=$((pct / 5))
        empty=$((20 - filled))
        printf "["
        for ((j=0; j<filled; j++)); do printf "█"; done
        for ((j=0; j<empty; j++)); do printf " "; done
        printf "]\n"
        total=$((total + count))
    else
        printf "  Batch %d: Not started\n" $i
    fi
done

echo "----------------------------------------"
overall_pct=$((total * 100 / 1000))
printf "Overall: %d/1000 (%d%%)\n" $total $overall_pct

# Progress bar for overall
echo ""
filled=$((overall_pct / 2))
empty=$((50 - filled))
printf "Progress: ["
for ((j=0; j<filled; j++)); do printf "█"; done
for ((j=0; j<empty; j++)); do printf " "; done
printf "] %d%%\n" $overall_pct

# Estimate completion time
if [ $total -gt 0 ] && [ $running -gt 0 ]; then
    echo ""
    first_pid=$(ps aux | grep "evaluate.py.*dual_path_batch" | grep -v grep | head -1 | awk '{print $2}')
    if [ ! -z "$first_pid" ]; then
        runtime_sec=$(ps -p $first_pid -o etimes= 2>/dev/null | tr -d ' ')
        if [ ! -z "$runtime_sec" ] && [ $runtime_sec -gt 0 ]; then
            qps=$(echo "scale=4; $total / $runtime_sec" | bc)
            remaining=$((1000 - total))
            eta_sec=$(echo "scale=0; $remaining / $qps" | bc)
            eta_min=$((eta_sec / 60))
            echo "Estimated time remaining: ~${eta_min} minutes"
        fi
    fi
fi

echo ""
echo "To refresh: ./monitor_evidence_aug.sh"
echo "To follow log: tail -f /tmp/evidence_aug_parallel.log"
echo "=========================================="
