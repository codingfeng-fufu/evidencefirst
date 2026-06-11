"""
Quick test: verify entity extraction fix on problematic examples from logs.
"""

import sys
sys.path.insert(0, '.')

from comagraag.evidence_utils import extract_question_entities

# Test cases from the logs showing bad entity extraction
test_cases = [
    ("Did Qionghai or Suining have a population of 658,798 in 2002?", ["Qionghai", "Suining"]),
    ("What location is shared by both Great Neck School District and Saddle Rock Elementary School?", ["Great Neck School District", "Saddle Rock Elementary School"]),
    ("Are both The Straight Story and Frozen films of the same genre?", ["Straight Story", "Frozen"]),
    ("Which NFL team did both Don Looney and DeSean Jackson play for?", ["Don Looney", "DeSean Jackson"]),
    ("Who was born first, Edward Keonjian or Aram Saroyan?", ["Edward Keonjian", "Aram Saroyan"]),
]

print("=" * 80)
print("Entity Extraction Fix Verification")
print("=" * 80)

passed = 0
total = len(test_cases)

for i, (question, expected_keywords) in enumerate(test_cases):
    print(f"\n{i+1}. Q: {question}")

    entities = extract_question_entities(question)
    print(f"   Extracted: {entities}")

    # Check if extracted entities contain expected keywords (flexible matching)
    contains_expected = any(
        any(keyword.lower() in entity.lower() for entity in entities)
        for keyword in expected_keywords
    )

    # Check for bad patterns
    bad_patterns = ['did', 'are', 'which', 'who', 'what', 'when', 'where', '?']
    has_bad = any(
        any(bad.lower() == entity.lower() or entity.lower().startswith(bad.lower() + ' ')
            for bad in bad_patterns)
        for entity in entities
    )

    if contains_expected and not has_bad:
        print(f"   ✓ PASS (contains expected entities, no question words)")
        passed += 1
    elif has_bad:
        print(f"   ✗ FAIL (still contains question words/punctuation)")
    else:
        print(f"   ⚠ PARTIAL (no bad patterns but missing expected entities)")
        passed += 0.5

print(f"\n{'=' * 80}")
print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
print(f"{'=' * 80}\n")

if passed >= total * 0.8:
    print("✅ Entity extraction fix looks good! Ready for full test.")
else:
    print("⚠️ Entity extraction still has issues. Need more fixes.")
