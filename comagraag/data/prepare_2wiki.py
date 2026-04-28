"""
下载 2WikiMultiHopQA，采样 500 道题，保存到 data/2wiki_sample.json
"""

import json, random, os

def prepare_2wiki(n=500, seed=42):
    out_path = "data/2wiki_sample.json"
    if os.path.exists(out_path):
        print(f"已存在：{out_path}")
        return

    print("下载 2WikiMultiHopQA（可能需要几分钟）...")
    try:
        from datasets import load_dataset
        ds = load_dataset("din0s/2wikimultihopqa", split="validation",
                          trust_remote_code=True)
        data = list(ds)
    except Exception as e:
        print(f"HuggingFace 下载失败：{e}")
        print("尝试本地文件 data/2wiki_dev_raw.json ...")
        with open("data/2wiki_dev_raw.json") as f:
            data = json.load(f)

    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"采样完成：{len(sample)} 道题 → {out_path}")

    print("\n=== 样例数据结构 ===")
    item = sample[0]
    print(f"字段：{list(item.keys())}")
    print(f"question: {item['question']}")
    print(f"answer: {item['answer']}")
    ctx = item.get("context", item.get("supporting_facts", []))
    print(f"context 类型：{type(ctx)}")
    if isinstance(ctx, dict):
        titles = list(ctx.keys())[:3]
        print(f"context 前3个标题：{titles}")
    elif isinstance(ctx, list) and ctx:
        print(f"context 前1项：{ctx[0]}")

if __name__ == "__main__":
    prepare_2wiki()
