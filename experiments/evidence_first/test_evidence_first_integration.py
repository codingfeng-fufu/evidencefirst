"""
Test EvidenceFirst integration into pipeline.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

import networkx as nx
from comagraag import pipeline

# Create a simple test KG
G = nx.DiGraph()
G.add_edge("Inception", "Christopher Nolan", relations=["directed_by"])
G.add_edge("Christopher Nolan", "British", relations=["nationality"])
G.add_edge("British", "UK", relations=["country"])
G.add_edge("Christopher Nolan", "Dark Knight", relations=["directed"])

# Test question
question = "What country is the director of Inception from?"

print("=" * 60)
print("Testing EvidenceFirst Integration")
print("=" * 60)

# Run with evidence_first variant
print("\n[1] Running with variant='evidence_first'...")
try:
    result = pipeline.run_pipeline(
        question=question,
        G=G,
        mode="full",
        passages=None,
        variant="evidence_first"
    )

    print(f"✓ Pipeline executed successfully!")
    print(f"  Answer: {result.get('answer', 'N/A')}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Converged: {result.get('converged', False)}")

    # Check history for evidence chain check
    history = result.get("history", [])
    chain_check = [h for h in history if h.get("step") == "evidence_chain_check"]
    if chain_check:
        print(f"  Evidence Chain Check: {chain_check[0]}")

    print("\n✅ Integration test PASSED!")

except Exception as e:
    print(f"\n❌ Integration test FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
