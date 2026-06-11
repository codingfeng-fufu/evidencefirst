"""测试postprocess_context_answer对高F1错误案例的效果"""
import sys
sys.path.insert(0, './comagraag')

import json
import re
import string
from agents import postprocess_context_answer

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))

# 加载数据
with open("./data/hotpotqa_1000_test.json") as f:
    dataset1 = json.load(f)
with open("./data/hotpotqa_extra500_test.json") as f:
    dataset2 = json.load(f)

all_items = dataset1 + dataset2
qa_map = {item['id']: item for item in all_items}

# 高F1错误案例（使用正确的QID）
test_cases = [
    ("5a81d0515542990a1d231ed4", "Flamingo Las Vegas", "Las Vegas Strip in Paradise, Nevada"),
    ("5ac23f98554299636651994b", "shock cavalry", "Shock troops"),
    ("5ab2c418554299545a2cfa67", "Hotchkiss M1914 machine gun", "Mle 1914 Hotchkiss machine gun"),
    ("5a84f7255542991dd0999e33", "Chad", "Republic of Chad"),
]

print("测试postprocess_context_answer能否修复高F1错误案例\n")
fixed = 0
for i, (qid, predicted, gold) in enumerate(test_cases, 1):
    if qid not in qa_map:
        print(f"{i}. QID {qid} not found")
        continue

    item = qa_map[qid]
    question = item['question']
    context = item.get('context', [])

    # 构造passages
    passages = []
    if isinstance(context, list) and context:
        if isinstance(context[0], list):
            # [[title, [sent1, sent2]], ...]
            for title, sents in context:
                passages.append(f"{title}: {' '.join(sents)}")
        else:
            passages = context

    # 测试postprocess
    expanded = postprocess_context_answer(question, predicted, passages)

    original_em = em(predicted, gold)
    new_em = em(expanded, gold)

    print(f"{i}. {'✓' if new_em == 1.0 else '✗'}")
    print(f"   Q: {question[:70]}...")
    print(f"   Original: '{predicted}' (EM={original_em})")
    print(f"   Expanded: '{expanded}' (EM={new_em})")
    print(f"   Gold: '{gold}'")

    if new_em == 1.0 and original_em == 0.0:
        fixed += 1
    print()

print(f"修复成功: {fixed}/{len(test_cases)}")
