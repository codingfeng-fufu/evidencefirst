"""Run 100-question test with evidence_first variant to verify fixes."""

import json
import pickle
import os
from datasets import load_dataset
from comagraag.pipeline import run_pipeline
import time

# Load dataset
dataset = load_dataset("hotpot_qa", "distractor", split="validation[:100]")

# Load KG from cache
kg_path = "data/hotpotqa_val_kgs.pkl"
if os.path.exists(kg_path):
    print("Loading knowledge graph from cache...")
    with open(kg_path, "rb") as f:
        kg_cache = pickle.load(f)
    print(f"KG cache loaded: {len(kg_cache)} graphs\n")
else:
    print("Warning: No KG cache found, will build on-the-fly\n")
    kg_cache = {}

def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
    s = ' '.join(s.split())
    return s

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def fuzzy_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm in gold_norm or gold_norm in pred_norm

def context_to_passages(context):
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles = [t for t, _ in context]
        sentences = [s for _, s in context]

    passages = []
    for title, sents in zip(titles, sentences):
        if isinstance(sents, list):
            text = ' '.join(sents)
        else:
            text = sents
        passages.append(f"{title}: {text}")
    return passages

results = []
start_time = time.time()

for i, item in enumerate(dataset, 1):
    qid = item['id']
    question = item['question']
    gold = item['answer']
    passages = context_to_passages(item['context'])

    print(f"[{i}/100] {question[:60]}...")

    # Get KG for this question
    G = kg_cache.get(qid)
    if G is None:
        # Build on-the-fly if not cached
        from comagraag.build_kg import build_kg
        G = build_kg(item['context'])

    try:
        result = run_pipeline(
            question=question,
            G=G,
            mode="full",
            passages=passages,
            variant="evidence_first"
        )

        pred = result.get('answer', 'ERROR')
        em = exact_match(pred, gold)
        fm = fuzzy_match(pred, gold)

        print(f"  Pred: {pred[:50]}")
        print(f"  Gold: {gold[:50]}")
        print(f"  EM={em}, Fuzzy={fm}")

        results.append({
            "qid": qid,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "exact_match": em,
            "fuzzy_match": fm,
            "chain_complete": result.get("chain_complete", False),
            "history": result.get("history", [])
        })

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ERROR: {error_msg}")
        traceback.print_exc()
        results.append({
            "qid": qid,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": f"ERROR: {error_msg}",
            "exact_match": False,
            "fuzzy_match": False,
            "chain_complete": False,
            "history": []
        })

    # Save checkpoint every 10 questions
    if i % 10 == 0:
        with open(f"test_fixed_intermediate_{i}.json", "w") as f:
            json.dump({"results": results}, f, indent=2)

elapsed = time.time() - start_time

# Calculate metrics
em_count = sum(1 for r in results if r["exact_match"])
fm_count = sum(1 for r in results if r["fuzzy_match"])
chain_count = sum(1 for r in results if r["chain_complete"])

summary = {
    "total": len(results),
    "correct_exact": em_count,
    "correct_fuzzy": fm_count,
    "em_exact": em_count / len(results),
    "em_fuzzy": fm_count / len(results),
    "chain_complete": chain_count,
    "chain_rate": chain_count / len(results),
    "elapsed_seconds": elapsed
}

output = {
    "summary": summary,
    "results": results
}

with open("test_fixed_100q_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"EM:           {summary['em_exact']:.1%} ({em_count}/100)")
print(f"Fuzzy Match:  {summary['em_fuzzy']:.1%} ({fm_count}/100)")
print(f"Chain Complete: {summary['chain_rate']:.1%} ({chain_count}/100)")
print(f"Time: {elapsed:.1f}s")
print(f"\nResults saved to test_fixed_100q_results.json")
