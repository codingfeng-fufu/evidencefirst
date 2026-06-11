"""
Test with higher passage threshold on full vs naive-only questions.
Two sub-tests:
  A) 66 naive-only questions: how many can we rescue?
  B) 38 full-only questions: do we regress any?
"""

import json, pickle, re, string, sys, os
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH  = "data/hotpotqa_sample.json"
KG_PATH    = "data/hotpotqa_kgs.pkl"
CACHE_FULL = "results/cache_full.json"
CACHE_NAIVE = "results/cache_naive_rag.json"

BAD_KEYWORDS = ("cannot", "no information", "no relevant", "not enough", "cannot determine")


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def passages_from_context(context) -> list:
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    return [title + ": " + " ".join(sents) for title, sents in zip(titles, sentences)]


def run_test(items, kgs, full_cache, label, thresh):
    import config
    config.PASSAGE_FALLBACK_THRESH = thresh
    from pipeline import run_pipeline
    import importlib, agents
    importlib.reload(agents)
    importlib.reload(__import__('pipeline'))
    from pipeline import run_pipeline

    em_before, em_after = [], []
    improved, regressed = 0, 0

    for item in tqdm(items, desc=f"{label} (thresh={thresh})"):
        qid      = item.get("id", item.get("_id", ""))
        question = item["question"]
        gold     = item["answer"]
        G        = kgs.get(qid)
        passages = passages_from_context(item["context"])
        old_ans  = full_cache.get(qid, {}).get("answer", "")

        try:
            result  = run_pipeline(question, G, mode="full", passages=passages)
            new_ans = result["answer"]
        except Exception as e:
            print(f"  [ERR] {qid}: {e}")
            new_ans = old_ans

        b = em(old_ans, gold)
        a = em(new_ans, gold)
        em_before.append(b)
        em_after.append(a)
        if a > b:
            improved += 1
        elif a < b:
            regressed += 1
            print(f"  - REGRESSED: {question[:60]}")
            print(f"    Gold: {gold} | Old: {old_ans} | New: {new_ans}")

    n = len(em_before)
    print(f"  EM before: {sum(em_before)/n:.3f} | after: {sum(em_after)/n:.3f}")
    print(f"  Improved: {improved} | Regressed: {regressed}")
    return sum(em_after) - sum(em_before)


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    with open(KG_PATH, "rb") as f:
        kgs = pickle.load(f)
    with open(CACHE_FULL) as f:
        full = json.load(f)
    with open(CACHE_NAIVE) as f:
        naive = json.load(f)

    naive_only, full_only = [], []
    for item in data:
        qid = item.get("id", item.get("_id", ""))
        gold = item["answer"]
        f_em = em(full.get(qid, {}).get("answer", ""), gold)
        n_em = em(naive.get(qid, {}).get("answer", ""), gold)
        if not f_em and n_em:
            naive_only.append(item)
        elif f_em and not n_em:
            full_only.append(item)

    print(f"naive_only: {len(naive_only)}, full_only: {len(full_only)}\n")

    THRESH = 50  # high threshold = passage always added

    print(f"=== Test A: 66 naive-only (want improvements) ===")
    delta_a = run_test(naive_only, kgs, full, "naive_only", thresh=THRESH)

    print(f"\n=== Test B: 38 full-only (want no regression) ===")
    delta_b = run_test(full_only, kgs, full, "full_only", thresh=THRESH)

    # Estimate overall impact
    total_old = 0.346 * 500
    print(f"\n=== Summary ===")
    print(f"Net EM gain: +{delta_a:.0f} (naive_only) {delta_b:+.0f} (full_only)")
    new_em = (total_old + delta_a + delta_b) / 500
    print(f"Estimated new overall EM: {new_em:.4f}  (vs naive_rag 0.402)")


if __name__ == "__main__":
    main()
