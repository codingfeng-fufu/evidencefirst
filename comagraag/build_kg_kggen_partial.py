"""
选择性重建：只对之前 fail 的 128 道题用 KGGen 重建 KG。
结果存到 data/hotpotqa_kgs_kggen_partial.pkl（只含这 128 个）。
原始 KG 不动。
"""

import json, pickle, os, time, logging
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
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

DATA_PATH    = "data/hotpotqa_sample.json"
QIDS_PATH    = "data/rebuild_qids.json"
PARTIAL_PATH = "data/hotpotqa_kgs_kggen_partial.pkl"
CKPT_PATH    = PARTIAL_PATH + ".ckpt"

kg_gen = KGGen(
    model="openai/moonshot-v1-8k",
    temperature=0.0,
    api_key=config.OPENAI_API_KEY,
    api_base=config.LLM_BASE_URL,
)


def context_to_text(context) -> str:
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    return "\n\n".join(f"{t}: {' '.join(s)}" for t, s in zip(titles, sentences))


def kggen_to_networkx(graph) -> nx.DiGraph:
    G = nx.DiGraph()
    alias_map = {}
    if hasattr(graph, 'entity_clusters') and graph.entity_clusters:
        for canonical, aliases in graph.entity_clusters.items():
            for alias in aliases:
                alias_map[alias] = canonical

    def resolve(name):
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


def build_partial():
    with open(DATA_PATH) as f:
        data = json.load(f)
    with open(QIDS_PATH) as f:
        target_qids = set(json.load(f))

    target_items = [item for item in data
                    if item.get("id", item.get("_id", "")) in target_qids]

    kgs = pickle.load(open(CKPT_PATH, "rb")) if os.path.exists(CKPT_PATH) else {}
    remaining = [item for item in target_items
                 if item.get("id", item.get("_id", "")) not in kgs]

    log.info(f"目标 {len(target_items)} 道题，已完成 {len(kgs)}，剩余 {len(remaining)}")

    for i, item in enumerate(remaining):
        qid = item.get("id", item.get("_id", ""))
        try:
            text = context_to_text(item["context"])

            # 用 ThreadPoolExecutor 加 5 分钟超时，防止 API 挂起
            def _generate():
                return kg_gen.generate(
                    input_data=text,
                    context="Extract factual knowledge graph triples for multi-hop QA",
                    cluster=False,
                )

            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_generate)
                try:
                    graph = future.result(timeout=300)  # 5 分钟超时
                except FutureTimeout:
                    future.cancel()
                    raise RuntimeError("KGGen timeout after 300s")

            G = kggen_to_networkx(graph)
            kgs[qid] = G
            clusters = len([v for v in (getattr(graph, 'entity_clusters', {}) or {}).values() if len(v) > 1])
            log.info(f"[{len(kgs)}/{len(target_items)}] nodes={G.number_of_nodes()}, "
                     f"edges={G.number_of_edges()}, merged={clusters}")
        except Exception as e:
            log.warning(f"[{i+1}] 失败 {qid}: {e}")
            kgs[qid] = nx.DiGraph()
            time.sleep(5)

        # 每 5 题存一次断点（避免重跑损失太多）
        if (i + 1) % 5 == 0:
            pickle.dump(kgs, open(CKPT_PATH, "wb"))
            log.info(f"--- 断点存档 {len(kgs)}/{len(target_items)} ---")

    pickle.dump(kgs, open(PARTIAL_PATH, "wb"))
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    log.info(f"完成 → {PARTIAL_PATH}，共 {len(kgs)} 个 KG")


if __name__ == "__main__":
    build_partial()
