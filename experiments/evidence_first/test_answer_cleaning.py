"""Test answer cleaning improvements on a small subset."""

import json
from datasets import load_dataset
from comagraag.pipeline import run_pipeline
from comagraag.evaluate import fuzzy_match

def normalize_answer(s: str) -> str:
    """Normalize answer for EM evaluation."""
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
    s = ' '.join(s.split())
    return s

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

# Load first 20 questions
dataset = load_dataset("hotpot_qa", "distractor", split="validation[:20]")

results = []
for i, item in enumerate(dataset):
    qid = item["id"]
    question = item["question"]
    gold_answer = item["answer"]
    context = item["context"]

    print(f"\n{'='*60}")
    print(f"Q{i+1}: {question}")
    print(f"Gold: {gold_answer}")

    try:
        result = run_pipeline(
            question=question,
            graph=None,  # Will be loaded from cache
            context=context,
            variant="evidence_first"
        )
        pred_answer = result.get("answer", "unknown")

        em = exact_match(pred_answer, gold_answer)
        fm = fuzzy_match(pred_answer, gold_answer)

        print(f"Pred: {pred_answer}")
        print(f"EM={em}, Fuzzy={fm}")

        results.append({
            "id": qid,
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "em": em,
            "fuzzy": fm,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "id": qid,
            "question": question,
            "gold": gold_answer,
            "pred": "ERROR",
            "em": False,
            "fuzzy": False,
        })

# Summary
em_count = sum(1 for r in results if r["em"])
fm_count = sum(1 for r in results if r["fuzzy"])

print(f"\n{'='*60}")
print(f"RESULTS (n={len(results)})")
print(f"EM:    {em_count}/{len(results)} = {em_count/len(results)*100:.1f}%")
print(f"Fuzzy: {fm_count}/{len(results)} = {fm_count/len(results)*100:.1f}%")

# Save detailed results
with open("test_answer_cleaning_results.json", "w") as f:
    json.dump({
        "results": results,
        "summary": {
            "total": len(results),
            "em": em_count,
            "fuzzy": fm_count,
            "em_rate": em_count / len(results),
            "fuzzy_rate": fm_count / len(results),
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to test_answer_cleaning_results.json")
