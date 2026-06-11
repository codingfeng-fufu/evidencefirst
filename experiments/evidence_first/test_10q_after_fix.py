"""
End-to-end test: EvidenceFirst on 10 real HotpotQA questions (after entity extraction fix).
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, '.')

# Load modules
from comagraag import pipeline
import config

print("=" * 80)
print("Entity Extraction Fix Validation - 10 Questions")
print("=" * 80)

# Load data
data_path = Path('./comagraag/data/hotpotqa_sample.json')
kg_path = Path('./comagraag/data/hotpotqa_kgs.pkl')

print(f"\n[1] Loading data...")
with open(data_path, 'r') as f:
    questions_data = json.load(f)

with open(kg_path, 'rb') as f:
    kgs = pickle.load(f)

print(f"  ✓ Loaded {len(questions_data)} questions")

# Select first 10 questions
test_questions = questions_data[:10]

print(f"\n[2] Running EvidenceFirst on 10 questions...")
print("-" * 80)

results = []
correct = 0
chain_complete = 0

for i, item in enumerate(test_questions):
    qid = item['id']
    question = item['question']
    gold_answer = item['answer']

    # Build passages
    context = item.get('context', {})
    titles = context.get('title', [])
    sentences = context.get('sentences', [])
    passages = []
    for title, sents in zip(titles, sentences):
        if isinstance(sents, list):
            passages.append(" ".join(sents))
        else:
            passages.append(str(sents))

    # Get KG
    G = kgs.get(qid)
    if G is None:
        print(f"\n[{i+1}/10] ⚠️  No KG found for {qid}, skipping...")
        continue

    print(f"\n[{i+1}/10] Q: {question[:70]}...")

    # Run EvidenceFirst
    try:
        result = pipeline.run_pipeline(
            question=question,
            G=G,
            mode="full",
            passages=passages,
            variant="evidence_first"
        )

        pred_answer = result.get("answer", "unknown")
        is_correct = gold_answer.strip().lower() == pred_answer.strip().lower()
        is_chain_complete = result.get("evidence_chain") is not None and len(result.get("evidence_chain", [])) > 0

        if is_correct:
            correct += 1
        if is_chain_complete:
            chain_complete += 1

        status = "✓" if is_correct else "✗"
        chain_status = "Complete" if is_chain_complete else "Incomplete"
        num_triples = len(result.get("evidence_chain", []))

        print(f"         {status} {chain_status} ({num_triples} triples)")
        print(f"         Gold: {gold_answer}")
        print(f"         Pred: {pred_answer}")

        results.append({
            "qid": qid,
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
            "chain_complete": is_chain_complete,
            "chain_length": num_triples
        })

    except Exception as e:
        print(f"         ✗ Error: {str(e)[:100]}")
        results.append({
            "qid": qid,
            "question": question,
            "gold": gold_answer,
            "pred": "ERROR",
            "correct": False,
            "chain_complete": False,
            "chain_length": 0
        })

total = len(results)
em = (correct / total * 100) if total > 0 else 0
chain_rate = (chain_complete / total * 100) if total > 0 else 0

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)
print(f"Total:              {total}")
print(f"Correct:            {correct}")
print(f"Exact Match (EM):   {em:.1f}%")
print(f"Chain Complete:     {chain_complete}")
print(f"Chain Completeness: {chain_rate:.1f}%")
print("=" * 80)

# Save results
output = {
    "summary": {
        "total": total,
        "correct": correct,
        "em": em / 100,
        "chain_complete": chain_complete,
        "chain_completeness": chain_rate / 100
    },
    "results": results
}

with open('test_10q_after_fix.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to: test_10q_after_fix.json\n")
