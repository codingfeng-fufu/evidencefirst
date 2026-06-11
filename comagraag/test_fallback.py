"""
Quick test: run the improved pipeline (with passage fallback) on the
~91 "Cannot determine" questions from cache_full.json.
Reports EM before/after without touching the main cache.
"""

import json, pickle, re, string, sys, os
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config
from pipeline import run_pipeline

DATA_PATH = "data/hotpotqa_sample.json"
KG_PATH   = "data/hotpotqa_kgs.pkl"
CACHE_PATH = "results/cache_full.json"

BAD_KEYWORDS = ("cannot", "no information", "no relevant", "not enough",
                "cannot be determined", "cannot determine")


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def is_bad(ans: str) -> bool:
    a = ans.lower().strip()
    return not a or any(kw in a for kw in BAD_KEYWORDS)


def passages_from_context(context) -> list:
    """Flatten HotpotQA context into list of passage strings."""
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    result = []
    for title, sents in zip(titles, sentences):
        result.append(title + ": " + " ".join(sents))
    return result


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    with open(KG_PATH, "rb") as f:
        kgs = pickle.load(f)
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    # Collect questions where old answer was bad
    failed = []
    for item in data:
        qid = item.get("id", item.get("_id", ""))
        old_ans = cache.get(qid, {}).get("answer", "")
        if is_bad(old_ans):
            failed.append(item)

    print(f"Found {len(failed)} 'Cannot determine' questions")
    print(f"Testing with passage fallback (THRESH={config.PASSAGE_FALLBACK_THRESH})...\n")

    em_before, em_after = [], []
    improved, regressed = 0, 0

    for item in tqdm(failed, desc="testing"):
        qid      = item.get("id", item.get("_id", ""))
        question = item["question"]
        gold     = item["answer"]
        G        = kgs.get(qid)
        passages = passages_from_context(item["context"])
        old_ans  = cache.get(qid, {}).get("answer", "")

        try:
            result  = run_pipeline(question, G, mode="full", passages=passages)
            new_ans = result["answer"]
        except Exception as e:
            print(f"  [ERR] {qid}: {e}")
            new_ans = ""

        before = em(old_ans, gold)
        after  = em(new_ans, gold)
        em_before.append(before)
        em_after.append(after)

        if after > before:
            improved += 1
            print(f"  + Q: {question[:65]}")
            print(f"    Gold: {gold}  |  Old: {old_ans}  |  New: {new_ans}")
        elif after < before:
            regressed += 1

    n = len(em_before)
    print(f"\n=== Results on {n} previously-failed questions ===")
    print(f"EM before: {sum(em_before)/n:.4f}  ({sum(em_before):.0f}/{n})")
    print(f"EM after : {sum(em_after)/n:.4f}  ({sum(em_after):.0f}/{n})")
    print(f"Improved : {improved}  |  Regressed: {regressed}")

    # Estimate overall impact on 500 questions
    total_em_old = 0.346 * 500
    delta = sum(em_after) - sum(em_before)
    new_total_em = (total_em_old + delta) / 500
    print(f"\nEstimated new overall EM (full 500): {new_total_em:.4f}")


if __name__ == "__main__":
    main()
