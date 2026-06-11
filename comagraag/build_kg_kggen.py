"""
用 KGGen 重建 HotpotQA 知识图谱。
- 每道题把所有 passage 合并成一段文本，调用 KGGen 一次
- cluster=True 自动合并实体别名（解决 Roy Khan / Roy Sætre Khantatat 问题）
- 结果存到 data/hotpotqa_kgs_kggen.pkl，原有 KG 不动
- 支持断点续跑
"""

import json, pickle, os, time, logging
import networkx as nx
from kg_gen import KGGen

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("build_kg_kggen.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

DATA_PATH = "data/hotpotqa_sample.json"
KG_PATH   = "data/hotpotqa_kgs_kggen.pkl"
CKPT_PATH = KG_PATH + ".ckpt"

kg_gen = KGGen(
    model="openai/moonshot-v1-8k",
    temperature=0.0,
    api_key=config.OPENAI_API_KEY,
    api_base=config.LLM_BASE_URL,
)


def context_to_text(context) -> str:
    """将 HotpotQA context 字段展平为一段文本。"""
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    parts = []
    for title, sents in zip(titles, sentences):
        parts.append(f"{title}: {' '.join(sents)}")
    return "\n\n".join(parts)


def kggen_to_networkx(graph) -> nx.DiGraph:
    """把 KGGen 的 graph 对象转成 NetworkX DiGraph（与现有 pipeline 兼容）。"""
    G = nx.DiGraph()

    # 如果有实体聚类，先建别名→规范名的映射
    alias_map = {}
    if hasattr(graph, 'entity_clusters') and graph.entity_clusters:
        for canonical, aliases in graph.entity_clusters.items():
            for alias in aliases:
                alias_map[alias] = canonical

    def resolve(name: str) -> str:
        return alias_map.get(name, name)

    for subj, rel, obj in graph.relations:
        subj = resolve(subj).strip()
        obj  = resolve(obj).strip()
        rel  = rel.strip().lower()
        if not subj or not obj:
            continue
        G.add_node(subj)
        G.add_node(obj)
        if G.has_edge(subj, obj):
            if rel not in G[subj][obj]["relations"]:
                G[subj][obj]["relations"].append(rel)
        else:
            G.add_edge(subj, obj, relations=[rel])
    return G


def build_all_kgs():
    with open(DATA_PATH) as f:
        data = json.load(f)

    kgs = pickle.load(open(CKPT_PATH, "rb")) if os.path.exists(CKPT_PATH) else {}
    remaining = [item for item in data
                 if item.get("id", item.get("_id", "")) not in kgs]
    log.info(f"已有 {len(kgs)} 个 KG，还需建 {len(remaining)} 个")

    for i, item in enumerate(remaining):
        qid = item.get("id", item.get("_id", ""))
        try:
            text = context_to_text(item["context"])
            graph = kg_gen.generate(
                input_data=text,
                context="Extract factual knowledge graph triples for multi-hop QA",
                cluster=True,
            )
            G = kggen_to_networkx(graph)
            kgs[qid] = G
            log.info(f"[{len(kgs)}/{len(data)}] nodes={G.number_of_nodes()}, "
                     f"edges={G.number_of_edges()}, "
                     f"clusters={len(getattr(graph, 'entity_clusters', {}) or {})}")
        except Exception as e:
            log.warning(f"[{i+1}] 建图失败 {qid}: {e}")
            kgs[qid] = nx.DiGraph()
            time.sleep(5)

        # 每 20 题存一次断点
        if (i + 1) % 20 == 0:
            pickle.dump(kgs, open(CKPT_PATH, "wb"))
            log.info(f"--- 断点存档 {len(kgs)}/{len(data)} ---")

    pickle.dump(kgs, open(KG_PATH, "wb"))
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    log.info(f"完成 → {KG_PATH}，共 {len(kgs)} 个 KG")


if __name__ == "__main__":
    build_all_kgs()
