"""
LightRAG 基线。
对每道题的 supporting passages 建本地索引，用 local 模式查询。
LLM: Moonshot API（OpenAI 兼容）
Embedding: 本地 sentence-transformers (all-MiniLM-L6-v2)
"""

import os, tempfile, shutil, asyncio, time
from functools import partial
import numpy as np
import config

# ── 共享 embedding 模型（避免每次重新加载）─────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


async def _local_embedding(texts: list[str]) -> np.ndarray:
    emb = _get_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def _make_llm_func():
    from lightrag.llm.openai import openai_complete_if_cache
    # LightRAG 调用约定: llm_func(prompt, system_prompt=..., history_messages=..., **kw)
    # openai_complete_if_cache 约定: (model, prompt, system_prompt, ...)
    # 需要适配
    async def llm_wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("model", None)  # LightRAG 有时会额外传 model
        return await openai_complete_if_cache(
            model=config.LLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=config.LLM_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            **kwargs,
        )
    return llm_wrapper


def _make_embedding_func():
    from lightrag.utils import EmbeddingFunc
    return EmbeddingFunc(
        embedding_dim=384,          # all-MiniLM-L6-v2 输出维度
        max_token_size=512,
        func=_local_embedding,
    )


def _context_to_text(context) -> str:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return "\n\n".join(str(p) for p in context if p)
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    return "\n\n".join(
        f"{t}\n" + (s if isinstance(s, str) else " ".join(s))
        for t, s in zip(titles, sentences)
    )


_EXTRACT_PROMPT = """Extract the direct answer to the question from the passage below.
Output ONLY the answer itself — 1 to 6 words, no explanation, no punctuation at the end.
For yes/no questions output only "yes" or "no".

Question: {question}
Passage: {passage}
Answer:"""

def _extract_short_answer(question: str, passage: str) -> str:
    """Use LLM to extract a concise answer from a long LightRAG response."""
    from agents import llm_call
    # If already short (≤8 words), return as-is
    if len(passage.split()) <= 8:
        return passage
    try:
        prompt = _EXTRACT_PROMPT.format(question=question, passage=passage[:1000])
        ans = llm_call(prompt, max_tokens=50).strip()
        # Strip quotes if present
        ans = ans.strip('"\'')
        return ans if ans else passage[:100]
    except Exception:
        # Fallback: first sentence
        lines = [l.strip() for l in passage.split("\n") if l.strip()]
        return lines[0][:100] if lines else passage[:100]


async def _run_lightrag(question: str, context, tmp_dir: str) -> str:
    from lightrag import LightRAG, QueryParam

    rag = LightRAG(
        working_dir=tmp_dir,
        llm_model_func=_make_llm_func(),
        llm_model_name=config.LLM_MODEL,
        embedding_func=_make_embedding_func(),
        llm_model_max_async=3,
        embedding_func_max_async=4,
        enable_llm_cache=False,
    )
    await rag.initialize_storages()

    text = _context_to_text(context)
    await rag.ainsert(text)

    result = await rag.aquery(
        question,
        param=QueryParam(mode="local"),
    )

    # 从 LightRAG 输出中提取简短答案
    text_out = str(result).strip()
    return _extract_short_answer(question, text_out)


def lightrag_baseline(question: str, context, retries: int = 3) -> str:
    for attempt in range(retries):
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="lightrag_")
            # 用持久 event loop 避免每次 asyncio.run() 切换 loop 导致 Lock 失效
            loop = _get_event_loop()
            answer = loop.run_until_complete(_run_lightrag(question, context, tmp_dir))
            return answer
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERR] lightrag attempt {attempt}: {e}")
                return ""
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


_loop = None

def _get_event_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop
