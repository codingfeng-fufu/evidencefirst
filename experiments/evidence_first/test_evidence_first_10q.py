"""
End-to-end test: EvidenceFirst on 100 real HotpotQA questions.
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
print("Phase 2 - Day 2: End-to-End Test (100 HotpotQA Questions)")
print("=" * 80)

# Load data
data_path = Path('./comagraag/data/hotpotqa_sample.json')
kg_path = Path('./comagraag/data/hotpotqa_kgs.pkl')

print(f"\n[1] Loading data...")
print(f"  Questions: {data_path}")
print(f"  KG: {kg_path}")

with open(data_path, 'r') as f:
    questions_data = json.load(f)

with open(kg_path, 'rb') as f:
    kgs = pickle.load(f)

print(f"  ✓ Loaded {len(questions_data)} questions")
print(f"  ✓ Loaded {len(kgs)} knowledge graphs")

# Select first 100 questions
test_questions = questions_data[:100]

print(f"\n[2] Running EvidenceFirst on 100 questions...")
print("-" * 80)

results = []
correct = 0
total = 0

for i, item in enumerate(test_questions):
    qid = item['id']
    question = item['question']
    gold_answer = item['answer']

    # Build passages from context dict
    context = item.get('context', {})
    titles = context.get('title', [])
    sentences = context.get('sentences', [])
    passages = []
    for title, sents in zip(titles, sentences):
        if isinstance(sents, list):
            passages.append(" ".join(sents))
        else:
            passages.append(str(sents))

    # Get KG for this question
    G = kgs.get(qid)
    if G is None:
        print(f"\n[{i+1}/10] ⚠️  No KG found for {qid}, skipping...")
        continue

    print(f"\n[{i+1}/100] Question: {question[:80]}...")
    print(f"         Gold: {gold_answer}")
    print(f"         KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run EvidenceFirst
    try:
        result = pipeline.run_pipeline(
            question=question,
            G=G,
            mode="full",
            passages=passages,
            variant="evidence_first"
        )

        pred_answer = result.get('answer', '')
        iterations = result.get('iterations', 0)
        converged = result.get('converged', False)
        history = result.get('history', [])

        # Check evidence chain info
        chain_check = [h for h in history if h.get('step') == 'evidence_chain_check']
        chain_complete = chain_check[0].get('complete', False) if chain_check else None
        chain_length = chain_check[0].get('chain_length', 0) if chain_check else 0

        # Normalize answers for comparison
        def normalize(s):
            return s.lower().strip().replace("the ", "").replace("a ", "")

        is_correct = normalize(pred_answer) == normalize(gold_answer)
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"         Pred: {pred_answer} {status}")
        print(f"         Chain: {'Complete' if chain_complete else 'Incomplete'} ({chain_length} triples)")

        results.append({
            'qid': qid,
            'question': question,
            'gold': gold_answer,
            'pred': pred_answer,
            'correct': is_correct,
            'chain_complete': chain_complete,
            'chain_length': chain_length,
            'kg_nodes': G.number_of_nodes(),
            'kg_edges': G.number_of_edges()
        })

    except Exception as e:
        print(f"         ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

if total > 0:
    em = correct / total
    print(f"\nExact Match (EM): {correct}/{total} = {em*100:.1f}%")

    # Chain completeness analysis
    complete_chains = sum(1 for r in results if r.get('chain_complete'))
    complete_rate = complete_chains / total if total > 0 else 0
    print(f"Chain Completeness Rate: {complete_chains}/{total} = {complete_rate*100:.1f}%")

    # Correctness by chain completeness
    complete_correct = sum(1 for r in results if r.get('chain_complete') and r.get('correct'))
    incomplete_correct = sum(1 for r in results if not r.get('chain_complete') and r.get('correct'))
    complete_count = sum(1 for r in results if r.get('chain_complete'))
    incomplete_count = total - complete_count

    print(f"\nCorrectness when chain is complete: {complete_correct}/{complete_count} = {complete_correct/complete_count*100 if complete_count > 0 else 0:.1f}%")
    print(f"Correctness when chain is incomplete: {incomplete_correct}/{incomplete_count} = {incomplete_correct/incomplete_count*100 if incomplete_count > 0 else 0:.1f}%")

    # Average chain length
    avg_chain_length = sum(r.get('chain_length', 0) for r in results) / total
    print(f"\nAverage chain length: {avg_chain_length:.1f} triples")

    # Average KG size
    avg_nodes = sum(r.get('kg_nodes', 0) for r in results) / total
    avg_edges = sum(r.get('kg_edges', 0) for r in results) / total
    print(f"Average KG size: {avg_nodes:.0f} nodes, {avg_edges:.0f} edges")

    print("\n" + "=" * 80)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    for i, r in enumerate(results):
        status = "✓" if r['correct'] else "✗"
        chain_status = "Complete" if r.get('chain_complete') else "Incomplete"
        print(f"{i+1}. {status} {chain_status} ({r.get('chain_length', 0)} triples)")
        print(f"   Q: {r['question'][:60]}...")
        print(f"   Gold: {r['gold']}")
        print(f"   Pred: {r['pred']}")

    # Save results
    output_path = './evidence_first_10q_test_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'em': em,
                'chain_completeness_rate': complete_rate,
                'avg_chain_length': avg_chain_length,
                'avg_kg_nodes': avg_nodes,
                'avg_kg_edges': avg_edges
            },
            'results': results
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")

else:
    print("\n⚠️  No questions were successfully processed.")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
