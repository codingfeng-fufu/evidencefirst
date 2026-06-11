"""
Analyze 100-question EvidenceFirst test results.
"""

import re
import sys

def parse_log(log_file):
    """Parse test output log and extract metrics."""

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract final summary
    summary_match = re.search(
        r'Final Summary.*?'
        r'Total: (\d+).*?'
        r'Correct: (\d+).*?'
        r'EM: ([\d.]+)%.*?'
        r'Chain Complete: (\d+).*?'
        r'Chain Completeness: ([\d.]+)%',
        content,
        re.DOTALL
    )

    if not summary_match:
        print("❌ Failed to parse summary from log")
        return None

    total = int(summary_match.group(1))
    correct = int(summary_match.group(2))
    em = float(summary_match.group(3))
    chain_complete = int(summary_match.group(4))
    chain_completeness = float(summary_match.group(5))

    return {
        'total': total,
        'correct': correct,
        'em': em,
        'chain_complete': chain_complete,
        'chain_completeness': chain_completeness
    }

def analyze_results(metrics):
    """Analyze results and provide recommendations."""

    print("=" * 80)
    print("100-Question Test Results Analysis")
    print("=" * 80)

    print(f"\n📊 Core Metrics:")
    print(f"  Total Questions:      {metrics['total']}")
    print(f"  Correct Answers:      {metrics['correct']}")
    print(f"  Exact Match (EM):     {metrics['em']:.2f}%")
    print(f"  Chain Complete:       {metrics['chain_complete']}")
    print(f"  Chain Completeness:   {metrics['chain_completeness']:.2f}%")

    print(f"\n📈 Comparison:")
    print(f"  10-question EM:       70.00%")
    print(f"  100-question EM:      {metrics['em']:.2f}%")
    print(f"  Difference:           {metrics['em'] - 70.0:+.2f}%")

    if abs(metrics['em'] - 70.0) <= 5.0:
        print(f"  Status: ✅ Stable (within ±5%)")
    elif metrics['em'] > 70.0:
        print(f"  Status: 🎉 Better than 10-question test")
    else:
        print(f"  Status: ⚠️  Lower than 10-question test")

    print(f"\n🎯 Decision Criteria:")

    if metrics['em'] >= 60:
        print(f"  ✅ EM ≥ 60%: PASS")
        print(f"  📋 Recommendation: Proceed to Day 3 (1000-question test)")
        decision = "proceed"
    elif metrics['em'] >= 50:
        print(f"  ⚠️  EM 50-60%: MARGINAL")
        print(f"  📋 Recommendation: Optimize first, then 1000-question test")
        decision = "optimize"
    else:
        print(f"  ❌ EM < 50%: FAIL")
        print(f"  📋 Recommendation: Re-evaluate approach")
        decision = "reevaluate"

    print(f"\n🔍 Chain Completeness Analysis:")
    if metrics['chain_completeness'] < 30:
        print(f"  ⚠️  Very low ({metrics['chain_completeness']:.1f}%)")
        print(f"      - Entity extraction may need improvement")
        print(f"      - But EM is still good due to fallback mechanism")
    elif metrics['chain_completeness'] < 50:
        print(f"  📊 Moderate ({metrics['chain_completeness']:.1f}%)")
        print(f"      - Room for improvement but acceptable")
    else:
        print(f"  ✅ Good ({metrics['chain_completeness']:.1f}%)")

    print(f"\n🚀 Next Steps:")

    if decision == "proceed":
        print(f"  1. ✅ 100-question validation PASSED")
        print(f"  2. 📝 Update EVIDENCEFIRST_IMPLEMENTATION_PLAN.md")
        print(f"  3. 🔄 Proceed to Day 3: 1000-question full evaluation")
        print(f"  4. 📊 Expected time: 8-12 hours")
    elif decision == "optimize":
        print(f"  1. 🔧 Fix entity extraction (remove sentence-initial caps)")
        print(f"  2. 🔄 Re-run 100-question test")
        print(f"  3. 📊 If improved to ≥60%, proceed to 1000-question")
    else:
        print(f"  1. 🔍 Investigate why EM dropped significantly")
        print(f"  2. 🐛 Check for bugs in integration")
        print(f"  3. 📊 Analyze individual failure cases")
        print(f"  4. ❓ Consider whether 10-question was severe sample bias")

    print(f"\n{'=' * 80}\n")

    return decision

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "test_evidence_first_100q_output.log"

    print(f"Reading log: {log_file}\n")

    metrics = parse_log(log_file)

    if metrics:
        decision = analyze_results(metrics)

        # Write result to file for plan update
        with open('day2_result.txt', 'w') as f:
            f.write(f"EM: {metrics['em']:.2f}%\n")
            f.write(f"Chain Completeness: {metrics['chain_completeness']:.2f}%\n")
            f.write(f"Decision: {decision}\n")

        print(f"✅ Results saved to day2_result.txt")
    else:
        print("❌ Failed to analyze results. Check log file format.")
        sys.exit(1)
