"""
对 cache_lightrag.json 里答案过长的条目，用 LLM 重新提取简短答案。
只处理后450题（新跑的长段落），前50题保持不变。
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from agents import llm_call
from tqdm import tqdm

CACHE_PATH = "results/cache_lightrag.json"
DATA_PATH  = "data/hotpotqa_sample.json"

EXTRACT_PROMPT = """Extract the direct answer to the question from the passage below.
Output ONLY the answer itself — 1 to 6 words, no explanation, no punctuation at the end.
For yes/no questions output only "yes" or "no".

Question: {question}
Passage: {passage}
Answer:"""

cache    = json.load(open(CACHE_PATH))
data     = json.load(open(DATA_PATH))
qid2item = {item.get("id", item.get("_id", str(i))): item for i, item in enumerate(data)}

qids = list(cache.keys())
# 只处理后450题中答案>8词的
to_fix = [qid for qid in qids[50:] if len(cache[qid]["answer"].split()) > 8]
print(f"需要重提取: {len(to_fix)} 题")

fixed = 0
for qid in tqdm(to_fix, desc="re-extract"):
    item = qid2item.get(qid)
    if not item:
        continue
    passage = cache[qid]["answer"]
    question = item["question"]
    try:
        prompt = EXTRACT_PROMPT.format(question=question, passage=passage[:1000])
        ans = llm_call(prompt, max_tokens=50).strip().strip('"\'')
        cache[qid]["answer"] = ans if ans else passage[:80]
        fixed += 1
    except Exception as e:
        print(f"  [ERR] {qid}: {e}")

    if fixed % 50 == 0:
        json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False, indent=2)

json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False, indent=2)
print(f"完成，共修复 {fixed} 题，已保存 {CACHE_PATH}")
