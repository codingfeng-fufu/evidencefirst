"""
Test A+B Solution on 100 HotpotQA Questions
Full-scale validation to measure A+B enhancement effectiveness
"""

import json
import sys
import pickle
from pathlib import Path
from loguru import logger
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comagraag.pipeline import run_pipeline

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("test_ab_100q.log", level="DEBUG")


def load_hotpotqa_sample(n: int = 100):
    """Load first n questions from HotpotQA test set."""
    data_path = project_root / "comagraag" / "data" / "hotpotqa_sample.json"
    kg_path = project_root / "comagraag" / "data" / "hotpotqa_kgs.pkl"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return [], {}

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(kg_path, 'rb') as f:
        kgs = pickle.load(f)

    return data[:n], kgs


def exact_match(pred: str, gold: str) -> bool:
    """Exact match evaluation (case-insensitive, strip)."""
    return pred.strip().lower() == gold.strip().lower()


def normalize_answer(text: str) -> str:
    """Normalize answer for more lenient matching."""
    import re
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fuzzy_match(pred: str, gold: str) -> bool:
    """Check if prediction contains the gold answer or vice versa."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return False

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Check containment
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    # Check significant word overlap
    pred_words = set(pred_norm.split())
    gold_words = set(gold_norm.split())

    if len(gold_words) > 0:
        overlap = len(pred_words & gold_words) / len(gold_words)
        if overlap >= 0.8:
            return True

    return False


def main():
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("A+B Solution Test - 100 HotpotQA Questions")
    logger.info("=" * 80)

    # Load test questions and KGs
    logger.info("Loading test questions and KGs...")
    questions, kgs = load_hotpotqa_sample(100)
    logger.info(f"Loaded {len(questions)} questions and {len(kgs)} KGs")

    # Run tests
    results = []
    correct_exact = 0
    correct_fuzzy = 0
    chain_complete_count = 0

    for i, item in enumerate(questions, 1):
        qid = item["id"]
        question = item["question"]
        gold_answer = item["answer"]

        # Convert context from dict to list of passages
        context = item.get("context", {})
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences = context.get("sentences", [])
            passages = []
            for title, sents in zip(titles, sentences):
                if sents:
                    passage = f"{title}: {' '.join(sents)}"
                    passages.append(passage)
        else:
            passages = context if isinstance(context, list) else []

        # Get KG for this question
        G = kgs.get(qid)
        if G is None:
            logger.warning(f"No KG found for question {qid}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i}/100: {question}")
        logger.info(f"Gold answer: {gold_answer}")

        try:
            # Run EvidenceFirst with A+B solution
            result = run_pipeline(
                question=question,
                G=G,
                mode="full",
                passages=passages,
                variant="evidence_first"
            )

            pred_answer = result["answer"]
            is_exact = exact_match(pred_answer, gold_answer)
            is_fuzzy = fuzzy_match(pred_answer, gold_answer)
            is_chain_complete = result.get("chain_complete", False)

            if is_exact:
                correct_exact += 1
                correct_fuzzy += 1
            elif is_fuzzy:
                correct_fuzzy += 1

            if is_chain_complete:
                chain_complete_count += 1

            logger.info(f"Predicted: {pred_answer}")
            logger.info(f"Exact match: {is_exact}, Fuzzy match: {is_fuzzy}")
            logger.info(f"Chain complete: {is_chain_complete}")

            results.append({
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "exact_match": is_exact,
                "fuzzy_match": is_fuzzy,
                "chain_complete": is_chain_complete,
                "history": result.get("history", [])
            })

        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "ERROR",
                "exact_match": False,
                "fuzzy_match": False,
                "chain_complete": False,
                "error": str(e)
            })

        # Save intermediate results every 10 questions
        if i % 10 == 0:
            intermediate_file = project_root / f"test_ab_100q_intermediate_{i}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "progress": i,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results: {i}/100 questions")

    # Calculate metrics
    elapsed_time = time.time() - start_time
    em_exact = correct_exact / len(questions) if questions else 0
    em_fuzzy = correct_fuzzy / len(questions) if questions else 0
    chain_rate = chain_complete_count / len(questions) if questions else 0

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Correct (exact match): {correct_exact}")
    logger.info(f"Correct (fuzzy match): {correct_fuzzy}")
    logger.info(f"Exact Match (EM): {em_exact:.1%}")
    logger.info(f"Fuzzy Match: {em_fuzzy:.1%}")
    logger.info(f"Chain complete: {chain_complete_count}")
    logger.info(f"Chain completeness rate: {chain_rate:.1%}")
    logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
    logger.info("=" * 80)

    # Evaluate A+B effectiveness
    if chain_rate >= 0.45:
        logger.info("✅ SUCCESS: Chain completeness ≥ 45% - A+B solution is effective!")
    elif chain_rate >= 0.35:
        logger.info("⚠️  PARTIAL: Chain completeness 35-45% - A+B has moderate effect")
    else:
        logger.info(f"❌ BELOW TARGET: Chain completeness {chain_rate:.1%} < 35%")

    if em_exact >= 0.50:
        logger.info("✅ TARGET ACHIEVED: EM ≥ 50%!")
    elif em_fuzzy >= 0.50:
        logger.info("⚠️  CLOSE: Fuzzy match ≥ 50%, but exact match needs improvement")
    else:
        logger.info(f"❌ BELOW TARGET: EM {em_exact:.1%} < 50%")

    # Save final results
    output_file = project_root / "test_ab_100q_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": len(questions),
                "correct_exact": correct_exact,
                "correct_fuzzy": correct_fuzzy,
                "em_exact": em_exact,
                "em_fuzzy": em_fuzzy,
                "chain_complete": chain_complete_count,
                "chain_rate": chain_rate,
                "elapsed_seconds": elapsed_time
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
