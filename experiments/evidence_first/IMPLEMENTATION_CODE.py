"""
双路径架构改进 - 代码实现模板

使用说明：
1. 将这些代码片段添加到对应的文件中
2. 测试每个组件
3. 逐步集成

预期提升：+5-7% EM (45.4% → 50-52%)
"""

# ============================================================================
# 文件1: comagraag/agents.py
# 在文件末尾添加以下代码
# ============================================================================

def run_context_answer_simple(question: str, passages: list) -> dict:
    """
    Context-based answer generation (Path 2)

    使用BM25检索相关passages，然后用LLM生成答案
    这是KG失败时的fallback路径

    Args:
        question: 问题
        passages: 所有可用的passages

    Returns:
        {
            "answer": str,
            "source": "context",
            "confidence": float,
            "used_passages": list
        }
    """
    if not passages:
        return {
            "answer": "",
            "source": "context",
            "confidence": 0.0,
            "used_passages": []
        }

    # 1. BM25检索
    from rank_bm25 import BM25Okapi
    import numpy as np

    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)

    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)

    # 取top-5
    top_k = 5
    top_indices = np.argsort(scores)[-top_k:][::-1]
    relevant_passages = [passages[i] for i in top_indices if scores[i] > 0]

    if not relevant_passages:
        return {
            "answer": "",
            "source": "context",
            "confidence": 0.0,
            "used_passages": []
        }

    # 2. 构建prompt（简洁版）
    passages_text = "\n\n".join(
        f"Passage {i+1}: {p[:500]}"  # 限制每个passage长度
        for i, p in enumerate(relevant_passages[:3])  # 只用前3个
    )

    prompt = f"""Answer the question based on the passages. Be concise and direct.

Question: {question}

{passages_text}

Answer:"""

    # 3. LLM调用
    try:
        answer_text = llm_call(prompt, max_tokens=100)
        answer_text = answer_text.strip()

        # 4. 简单的置信度估计（基于token overlap）
        answer_tokens = set(answer_text.lower().split())
        passage_tokens = set(' '.join(relevant_passages).lower().split())

        if answer_tokens:
            overlap_ratio = len(answer_tokens & passage_tokens) / len(answer_tokens)
            confidence = min(overlap_ratio, 1.0)
        else:
            confidence = 0.0

        return {
            "answer": answer_text,
            "source": "context",
            "confidence": confidence,
            "used_passages": relevant_passages[:3]
        }

    except Exception as e:
        print(f"[Context Path] Error: {e}")
        return {
            "answer": "",
            "source": "context",
            "confidence": 0.0,
            "used_passages": []
        }


# ============================================================================
# 文件2: comagraag/pipeline.py
# 在文件中间（reselect_cached_answer函数之后）添加以下代码
# ============================================================================

def _looks_bad_answer(answer: str) -> bool:
    """
    检测明显的bad answer

    Bad answer的特征：
    - 空或太短
    - 包含"unknown", "cannot"等标记
    - 过长（>20词）
    - 包含meta语言（"triple", "instruction"）
    """
    if not answer or len(answer.strip()) < 2:
        return True

    lowered = answer.lower().strip()

    # Bad markers
    bad_markers = [
        "unknown",
        "cannot", "can not",
        "insufficient",
        "not enough",
        "no information",
        "not specified",
        "provided context",
        "context does not",
        "triple", "triples",
        "instruction", "instructions",
        "graph does not"
    ]

    if any(marker in lowered for marker in bad_markers):
        return True

    # 过长
    if len(answer.split()) > 20:
        return True

    # 以奇怪的token开始
    if lowered.startswith(("however,", "but the", "check:", "-")):
        return True

    return False


def _detect_question_type(question: str) -> str:
    """
    快速问题类型检测

    返回：yesno, comparison, location, number_date, person, common, other
    """
    q = question.lower().strip()

    # Yes/No
    if q.startswith((
        "are ", "is ", "was ", "were ",
        "do ", "does ", "did ",
        "has ", "have ", "had ",
        "can ", "could ", "will ", "would "
    )):
        return "yesno"

    # Comparison
    comparison_markers = [
        "older", "younger", "first", "earlier", "later",
        "more ", "larger", "bigger", "smaller",
        "closer", "nearer", "further",
        "released first", "came first", "born first"
    ]
    if any(marker in q for marker in comparison_markers):
        return "comparison"

    # Location
    if q.startswith("where") or any(marker in q for marker in [
        "what city", "what country", "what state", "what region",
        "what province", "what county",
        "located in", "headquartered in", "based in"
    ]):
        return "location"

    # Number/Date
    if any(marker in q for marker in [
        "how many", "what year", "when was", "when did",
        "what date", "what age", "population", "number of"
    ]):
        return "number_date"

    # Person
    if q.startswith("who ") or any(marker in q for marker in [
        "which person", "which actor", "which actress",
        "which author", "which composer", "which player"
    ]):
        return "person"

    # Common/Both
    if " both " in q or "in common" in q:
        return "common"

    return "other"


def _is_yesno_answer(answer: str) -> bool:
    """检查是否是yes/no答案"""
    normalized = answer.lower().strip().rstrip(".")
    return normalized in ["yes", "no"]


def _select_answer_dual_path(kg_result: dict,
                             context_result: dict,
                             question: str) -> dict:
    """
    答案选择器：在KG和Context答案间选择

    决策规则（按优先级）：
    1. Bad answer检测：如果一个bad另一个good，选good
    2. 问题类型匹配：某些问题类型优先某条路径
    3. 分数/置信度比较
    4. 默认策略：KG优先（因为可验证）
    """
    kg_answer = kg_result.get("answer", "")
    context_answer = context_result.get("answer", "")

    kg_bad = _looks_bad_answer(kg_answer)
    context_bad = _looks_bad_answer(context_answer)

    # Rule 1: Bad answer fallback
    if kg_bad and not context_bad:
        return {
            "answer": context_answer,
            "source": "context_fallback",
            "reason": "kg_bad",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "kg_score": kg_result.get("score", 0),
            "context_confidence": context_result.get("confidence", 0)
        }

    if context_bad and not kg_bad:
        return {
            "answer": kg_answer,
            "source": "kg",
            "reason": "context_bad",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "kg_score": kg_result.get("score", 0),
            "context_confidence": context_result.get("confidence", 0)
        }

    # 两个都bad
    if kg_bad and context_bad:
        # 选一个less bad的
        return {
            "answer": kg_answer,  # 默认KG
            "source": "kg_both_bad",
            "reason": "both_bad",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }

    # Rule 2: 问题类型优化
    qtype = _detect_question_type(question)

    # Yes/No问题：优先context（更直接）
    if qtype == "yesno" and _is_yesno_answer(context_answer):
        return {
            "answer": context_answer,
            "source": "context_yesno",
            "reason": f"yesno_match",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }

    # Comparison问题：优先context（更准确）
    if qtype == "comparison" and len(context_answer.split()) <= 8:
        return {
            "answer": context_answer,
            "source": "context_comparison",
            "reason": "comparison_concise",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }

    # Rule 3: 分数比较
    kg_score = kg_result.get("score", 0)
    context_conf = context_result.get("confidence", 0)

    # KG高分：直接用
    if kg_score >= 0.6:
        return {
            "answer": kg_answer,
            "source": "kg_high_score",
            "reason": f"kg_score_{kg_score:.2f}",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "kg_score": kg_score
        }

    # Context高置信度 + KG低分：用context
    if context_conf >= 0.7 and kg_score < 0.4:
        return {
            "answer": context_answer,
            "source": "context_high_conf",
            "reason": f"context_conf_{context_conf:.2f}_kg_low",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "context_confidence": context_conf
        }

    # Rule 4: 默认策略 - KG优先（可验证）
    return {
        "answer": kg_answer,
        "source": "kg_default",
        "reason": "default_kg",
        "kg_answer": kg_answer,
        "context_answer": context_answer,
        "kg_score": kg_score,
        "context_confidence": context_conf
    }


def run_pipeline_dual_path(question: str,
                           G: nx.DiGraph,
                           mode: str = "full",
                           passages: list = None,
                           variant: str = "default") -> dict:
    """
    双路径架构 Pipeline

    Path 1: KG-based (现有pipeline)
    Path 2: Context-based (BM25 + LLM)
           ↓
       Answer Selector
           ↓
       Final Answer

    Args:
        question: 问题
        G: 知识图谱
        mode: pipeline模式 ("full", "no_verif", "no_decomp")
        passages: 原始passages（用于context path）
        variant: 变体类型

    Returns:
        dict: 包含answer和详细信息
    """
    # Path 1: 运行KG-based pipeline
    kg_result = run_pipeline(
        question=question,
        G=G,
        mode=mode,
        passages=passages,
        variant=variant
    )

    # Path 2: 运行Context-based path
    context_result = {"answer": "", "source": "context", "confidence": 0.0}

    if passages:
        from agents import run_context_answer_simple
        try:
            context_result = run_context_answer_simple(question, passages)
        except Exception as e:
            print(f"[Dual Path] Context path failed: {e}")
            context_result = {"answer": "", "source": "context", "confidence": 0.0}

    # Answer Selection
    final_result = _select_answer_dual_path(kg_result, context_result, question)

    # 保留完整trace信息（用于调试和分析）
    final_result.update({
        "iterations": kg_result.get("iterations", 1),
        "converged": kg_result.get("converged", True),
        "history": kg_result.get("history", []),
        "mode": f"{mode}_dual_path"
    })

    return final_result


# ============================================================================
# 文件3: comagraag/evaluate.py
# 在 _PIPELINE_MODES 行附近添加
# ============================================================================

# 找到这一行：
# _PIPELINE_MODES = ("full", "no_verif", "no_decomp")

# 改为：
_PIPELINE_MODES = ("full", "no_verif", "no_decomp", "full_dual_path")

# 然后在 run_eval 函数中，找到调用 run_pipeline 的地方
# 添加对 full_dual_path 的支持

# 找到类似这样的代码：
#     if mode in _PIPELINE_MODES:
#         from pipeline import run_pipeline
#         result = run_pipeline(question, G, mode, passages)

# 改为：
#     if mode in _PIPELINE_MODES:
#         if mode == "full_dual_path":
#             from pipeline import run_pipeline_dual_path
#             result = run_pipeline_dual_path(question, G, "full", passages)
#         else:
#             from pipeline import run_pipeline
#             result = run_pipeline(question, G, mode, passages)


# ============================================================================
# 测试代码
# 创建新文件: test_dual_path.py
# ============================================================================

"""
test_dual_path.py - 测试双路径架构

使用方法：
python test_dual_path.py
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "comagraag"))

def test_single_example():
    """测试单个例子"""
    import pickle
    import networkx as nx
    from pipeline import run_pipeline_dual_path

    # 加载数据
    data_file = Path("data/hotpotqa_1000_test.jsonl")
    kg_file = Path("data/hotpotqa_kgs.pkl")

    # 读取第一个问题
    with open(data_file) as f:
        item = json.loads(f.readline())

    question = item["question"]
    gold = item["answer"]

    # 加载KG
    with open(kg_file, "rb") as f:
        all_kgs = pickle.load(f)

    qid = item.get("_id") or item.get("id")
    G = all_kgs.get(qid, nx.DiGraph())

    # 获取passages
    context = item.get("context", [])
    if isinstance(context[0], list):
        passages = [title + ": " + " ".join(sents) for title, sents in context]
    else:
        passages = context

    # 运行双路径
    print(f"Question: {question}")
    print(f"Gold: {gold}")
    print("\nRunning dual path...")

    result = run_pipeline_dual_path(question, G, "full", passages)

    print(f"\nKG Answer: {result.get('kg_answer', 'N/A')}")
    print(f"Context Answer: {result.get('context_answer', 'N/A')}")
    print(f"Selected: {result['answer']}")
    print(f"Source: {result.get('source', 'N/A')}")
    print(f"Reason: {result.get('reason', 'N/A')}")

    # 检查正确性
    from evaluate import normalize
    correct = normalize(result['answer']) == normalize(gold)
    print(f"\nCorrect: {correct}")

if __name__ == "__main__":
    test_single_example()


# ============================================================================
# 运行完整评估
# ============================================================================

# 在命令行运行：
# python evaluate.py --mode full_dual_path --n 100 \
#   --data data/hotpotqa_1000_test.jsonl \
#   --kg data/hotpotqa_kgs.pkl \
#   --out results/test_dual_path_100.csv \
#   --jsonl-out results/test_dual_path_100_predictions.jsonl

# 然后分析结果：
# python -c "
# import csv
# with open('results/test_dual_path_100.csv') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         print(f'EM: {row[\"EM\"]}, F1: {row[\"F1\"]}')
# "
