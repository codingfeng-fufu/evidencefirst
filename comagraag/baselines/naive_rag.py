"""
Naive RAG 基线：BM25 检索 + LLM 生成
输入：HotpotQA 的 context（supporting passages）
输出：答案字符串
"""

from rank_bm25 import BM25Okapi
from openai import OpenAI
import time, config
import usage

client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)

PROMPT = """请根据以下上下文回答问题。
只使用上下文中的信息，不要使用外部知识。
答案要简洁（事实类问题 1-5 个词）。

上下文：
{context}

问题：{question}
答案："""

def naive_rag(question: str, context: list,
              top_k: int = 5, retries: int = 4) -> str:
    """
    context: HotpotQA 的 context 字段，格式 [[title, [sent1, sent2, ...]], ...]
    """
    # 1. 构建句子级语料库
    # context 支持两种格式：
    #   list of [title, sentences]  （原始 HotpotQA）
    #   dict {"title": [...], "sentences": [[...], ...]}  （datasets 加载后）
    passages = []
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        passages = [p for p in context if p]
    elif isinstance(context, dict):
        titles    = context.get("title", [])
        sentences = context.get("sentences", [])
        pairs = zip(titles, sentences)
        for title, sents in pairs:
            for sent in sents:
                passages.append(f"{title}: {sent}")
    else:
        pairs = context
        for title, sents in pairs:
            for sent in sents:
                passages.append(f"{title}: {sent}")

    if not passages:
        return ""

    # 2. BM25 检索 top-k 段落
    tokenized = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved = "\n".join(passages[i] for i in top_idxs)

    # 3. LLM 生成答案
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user",
                           "content": PROMPT.format(context=retrieved,
                                                    question=question)}],
                temperature=0,
                max_tokens=100,
            )
            usage.record_response(resp)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERR] naive_rag: {e}")
                return ""
