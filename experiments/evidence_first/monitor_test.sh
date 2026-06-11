#!/bin/bash
# Monitor 100-question test progress

while true; do
    clear
    echo "=== A+B Solution 100Q Test Monitor ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Check if process is running
    if ps aux | grep -q "[5]0316.*test_ab_solution_100q"; then
        echo "Status: ✅ Running (PID: 50316)"
    else
        echo "Status: ⏹️  Stopped or Completed"
    fi
    echo ""

    # Get latest progress from log
    echo "Latest Progress:"
    tail -50 test_ab_100q.log 2>/dev/null | grep "Question [0-9]*/100" | tail -1
    echo ""

    # Check for intermediate files
    echo "Intermediate Checkpoints:"
    ls -1 test_ab_100q_intermediate_*.json 2>/dev/null | tail -3 || echo "None yet"
    echo ""

    # Check if final result exists
    if [ -f test_ab_100q_results.json ]; then
        echo "Final Results:"
        python -c "
import json
with open('test_ab_100q_results.json', 'r') as f:
    data = json.load(f)
    s = data['summary']
    print(f\"  Exact Match: {s['em_exact']:.1%} ({s['correct_exact']}/100)\")
    print(f\"  Fuzzy Match: {s['em_fuzzy']:.1%} ({s['correct_fuzzy']}/100)\")
    print(f\"  Chain Rate:  {s['chain_rate']:.1%} ({s['chain_complete']}/100)\")
    print(f\"  Time: {s['elapsed_seconds']/60:.1f} minutes\")
" 2>/dev/null
    else
        echo "Final results not ready yet"
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
