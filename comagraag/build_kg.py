"""
从 HotpotQA 的 supporting passages 建图。
每道题建一个 NetworkX DiGraph，结果 pickle 缓存。
这一步跑一次就够，之后实验直接加载。
"""

import json, pickle, os, time, re, logging
import networkx as nx
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import config

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                      # 终端
        logging.FileHandler("build_kg.log", encoding="utf-8"),  # 文件
    ]
)
log = logging.getLogger(__name__)

# 建图用主模型（coding plan 只支持这一个）
BUILD_MODEL = config.LLM_MODEL
client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)

# ---------- 采样数据 ----------

def sample_hotpotqa(n_bridge=250, n_compare=250, seed=42):
    """从 HotpotQA fullwiki dev set 采样，保存到本地。"""
    os.makedirs("data", exist_ok=True)
    out_path = "data/hotpotqa_sample.json"

    if os.path.exists(out_path):
        log.info(f"数据已存在：{out_path}")
        return

    log.info("下载 HotpotQA...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)

    random.seed(seed)
    bridge  = [x for x in ds if x["type"] == "bridge"]
    compare = [x for x in ds if x["type"] == "comparison"]

    sample = random.sample(bridge, n_bridge) + random.sample(compare, n_compare)
    random.shuffle(sample)

    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    log.info(f"采样完成：{len(sample)} 条 → {out_path}")


# ---------- Triple 抽取 ----------

EXTRACT_PROMPT = """从以下段落中抽取所有事实三元组，格式为 (主体, 关系, 客体)。
要求：
- 主体和客体是具体的命名实体或名词短语
- 关系是简洁的动词短语
- 只输出 JSON 列表，不要任何解释
- 格式：[["主体", "关系", "客体"], ...]

段落：
{passage}"""

def extract_triples(passage: str, max_retries=3) -> list:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=BUILD_MODEL,
                messages=[{"role": "user", "content": EXTRACT_PROMPT.format(passage=passage)}],
                temperature=1,
                max_tokens=1500,
            )
            text = resp.choices[0].message.content.strip()
            # 去掉 markdown 代码块
            text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
            triples = json.loads(text)
            return [(t[0].strip(), t[1].strip().lower(), t[2].strip())
                    for t in triples if len(t) == 3]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "throttling" in err_str or "quota" in err_str:
                wait = 60 * (attempt + 1)
                log.warning(f"限流，等待 {wait}s 后重试...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.warning(f"extract_triples 失败: {e}")
                return []


# ---------- 建图 ----------

def build_kg(context) -> nx.DiGraph:
    """
    context: HotpotQA 的 context 字段
    datasets 库返回的格式是 {"title": [...], "sentences": [[...], ...]}
    并行抽取各 passage 的三元组以提速。
    """
    G = nx.DiGraph()
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        passages = [str(p) for p in context if p]
    else:
        titles    = context["title"]     if isinstance(context, dict) else [t for t, _ in context]
        sentences = context["sentences"] if isinstance(context, dict) else [s for _, s in context]
        passages  = [sents if isinstance(sents, str) else " ".join(sents) for sents in sentences]

    # 并发提取三元组；正式补跑时可用 KG_BUILD_WORKERS 控制 API 压力。
    all_triples = [None] * len(passages)
    max_workers = int(os.environ.get("KG_BUILD_WORKERS", "5"))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(extract_triples, p): i for i, p in enumerate(passages)}
        for fut in as_completed(futs):
            all_triples[futs[fut]] = fut.result()

    for triples in all_triples:
        for subj, rel, obj in (triples or []):
            subj, obj = subj.strip(), obj.strip()
            if not subj or not obj:
                continue
            G.add_node(subj)
            G.add_node(obj)
            if G.has_edge(subj, obj):
                G[subj][obj]["relations"].append(rel)
            else:
                G.add_edge(subj, obj, relations=[rel])
    return G


def build_all_kgs(data_path="data/hotpotqa_sample.json",
                   kg_path="data/hotpotqa_kgs.pkl"):
    """建所有题目的 KG，带断点续跑。"""
    with open(data_path) as f:
        data = json.load(f)

    ckpt_path = kg_path + ".ckpt"
    kgs = pickle.load(open(ckpt_path, "rb")) if os.path.exists(ckpt_path) else {}
    log.info(f"已有 {len(kgs)} 个 KG，共 {len(data)} 道题，还需建 {len(data)-len(kgs)} 个")

    for i, item in enumerate(data):
        qid = item.get("id", item.get("_id", str(i)))
        if qid in kgs:
            continue
        try:
            G = build_kg(item["context"])
            kgs[qid] = G
            log.info(f"[{len(kgs)}/{len(data)}] 题 {i+1}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        except Exception as e:
            log.warning(f"建图失败 {qid}: {e}")
            kgs[qid] = nx.DiGraph()

        if (i + 1) % 20 == 0:
            pickle.dump(kgs, open(ckpt_path, "wb"))
            log.info(f"--- 进度 {i+1}/{len(data)}，已存档 ---")

    pickle.dump(kgs, open(kg_path, "wb"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log.info(f"建图完成 → {kg_path}，共 {len(kgs)} 个 KG")


if __name__ == "__main__":
    sample_hotpotqa()
    build_all_kgs()
