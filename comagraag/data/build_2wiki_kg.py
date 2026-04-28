"""
为 2WikiMultiHopQA 建知识图谱，复用 build_kg.py 的逻辑。
"""

import json, pickle, os, sys
import networkx as nx

sys.path.insert(0, ".")
from build_kg import build_kg
from data.wiki2_utils import normalize_context


def build_all_2wiki_kgs(data_path="data/2wiki_sample.json",
                        kg_path="data/2wiki_kgs.pkl"):
    with open(data_path) as f:
        data = json.load(f)

    ckpt_path = kg_path + ".ckpt"
    kgs = pickle.load(open(ckpt_path, "rb")) if os.path.exists(ckpt_path) else {}
    print(f"已有 {len(kgs)} 个 KG，共 {len(data)} 道题")

    for i, item in enumerate(data):
        qid = item.get("_id", item.get("id", str(i)))
        if qid in kgs:
            continue

        raw_context = item.get("context", item.get("supporting_facts", []))
        context = normalize_context(raw_context)

        if not context:
            print(f"  [SKIP] {qid} context 为空或格式未知")
            kgs[qid] = nx.DiGraph()
            continue

        try:
            G = build_kg(context)
            kgs[qid] = G
        except Exception as e:
            print(f"  [WARN] 建图失败 {qid}: {e}")
            kgs[qid] = nx.DiGraph()

        if (i + 1) % 20 == 0:
            pickle.dump(kgs, open(ckpt_path, "wb"))
            print(f"  进度 {i+1}/{len(data)}")

    pickle.dump(kgs, open(kg_path, "wb"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"建图完成 → {kg_path}")


if __name__ == "__main__":
    build_all_2wiki_kgs()
