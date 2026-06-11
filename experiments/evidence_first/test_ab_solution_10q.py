"""
Test A+B Solution on 10 HotpotQA Questions
Expected: Chain completeness should increase from 27% to 70%+ after A+B enhancement
"""

import json
import sys
import pickle
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comagraag.pipeline import run_pipeline

logger.remove()
logger.add(sys.stderr, level="INFO")


def load_hotpotqa_sample(n: int = 10):
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


def main():
    logger.info("=" * 80)
    logger.info("A+B Solution Test - 10 HotpotQA Questions")
    logger.info("=" * 80)

    # Load test questions and KGs
    logger.info("Loading test questions and KGs...")
    questions, kgs = load_hotpotqa_sample(10)
    logger.info(f"Loaded {len(questions)} questions and {len(kgs)} KGs")

    # Run tests
    results = []
    correct = 0
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
        logger.info(f"Question {i}/10: {question}")
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
            is_correct = exact_match(pred_answer, gold_answer)
            is_chain_complete = result.get("chain_complete", False)

            if is_correct:
                correct += 1

            if is_chain_complete:
                chain_complete_count += 1

            logger.info(f"Predicted: {pred_answer}")
            logger.info(f"Correct: {is_correct}")
            logger.info(f"Chain complete: {is_chain_complete}")

            results.append({
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct,
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
                "correct": False,
                "chain_complete": False,
                "error": str(e)
            })

    # Calculate metrics
    em = correct / len(questions) if questions else 0
    chain_rate = chain_complete_count / len(questions) if questions else 0

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Exact Match (EM): {em:.1%}")
    logger.info(f"Chain complete: {chain_complete_count}")
    logger.info(f"Chain completeness rate: {chain_rate:.1%}")
    logger.info("=" * 80)

    # Check if A+B worked
    if chain_rate >= 0.70:
        logger.info("✅ SUCCESS: Chain completeness ≥ 70% - A+B solution is working!")
    elif chain_rate >= 0.50:
        logger.info("⚠️  PARTIAL: Chain completeness 50-70% - A+B has some effect but needs tuning")
    else:
        logger.info("❌ ISSUE: Chain completeness < 50% - A+B may not be working properly")

    # Save results
    output_file = project_root / "test_ab_10q_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": len(questions),
                "correct": correct,
                "em": em,
                "chain_complete": chain_complete_count,
                "chain_rate": chain_rate
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
