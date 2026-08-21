"""
IRCoT baseline: interleave retrieval with chain-of-thought.

This is a lightweight local implementation for HotpotQA-style contexts:
- sentence-level BM25 retrieval
- iterative retrieve-reason loop
- concise final answer extraction
"""

import re
import time
from rank_bm25 import BM25Okapi
from openai import OpenAI

import config
import usage


client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)

MAX_STEPS = 3
TOP_K_PER_STEP = 5

IRCOT_PROMPT = """You are solving a multi-hop question with interleaved retrieval.

Question:
{question}

Current evidence:
{evidence}

Current reasoning:
{reasoning}

You may do one of two things:
1. If the evidence is still insufficient, write:
THOUGHT: <brief reasoning>
SEARCH_QUERY: <a short retrieval query for the next hop>

2. If you can answer, write:
THOUGHT: <brief reasoning>
FINAL_ANSWER: <short answer only>

Rules:
- Use only the evidence provided.
- Keep THOUGHT brief.
- SEARCH_QUERY must be short and concrete.
- FINAL_ANSWER must be concise, ideally 1-8 words.
- Output exactly one THOUGHT and one SEARCH_QUERY or FINAL_ANSWER.
"""


FINAL_ANSWER_PROMPT = """Answer the question using the evidence below.

Question:
{question}

Evidence:
{evidence}

Reasoning trace:
{reasoning}

Return only:
FINAL_ANSWER: <short answer>
"""


def _sentence_corpus(context) -> list[str]:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return [p for p in context if p]
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles = [t for t, _ in context]
        sentences = [s for _, s in context]

    corpus = []
    for title, sent_list in zip(titles, sentences):
        for sent in sent_list:
            sent = sent.strip()
            if sent:
                corpus.append(f"{title}: {sent}")
    return corpus


def _retrieve(bm25: BM25Okapi, corpus: list[str], query: str, seen: set[int], top_k: int) -> list[int]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    chosen = []
    for idx in ranked:
        if idx in seen:
            continue
        if scores[idx] <= 0 and chosen:
            break
        chosen.append(idx)
        if len(chosen) >= top_k:
            break
    return chosen


def _parse_step(text: str) -> tuple[str, str | None, str | None]:
    thought = ""
    query = None
    answer = None
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("THOUGHT:"):
            thought = line[len("THOUGHT:"):].strip()
        elif line.upper().startswith("SEARCH_QUERY:"):
            query = line[len("SEARCH_QUERY:"):].strip()
        elif line.upper().startswith("FINAL_ANSWER:"):
            answer = line[len("FINAL_ANSWER:"):].strip()
    return thought, query, answer


def _extract_final_answer(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("FINAL_ANSWER:"):
            return line[len("FINAL_ANSWER:"):].strip()
    return text.strip()


def ircot(question: str, context, retries: int = 4) -> str:
    corpus = _sentence_corpus(context)
    if not corpus:
        return ""

    tokenized = [c.lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized)

    seen = set()
    evidence_chunks = []
    reasoning_steps = []
    current_query = question

    for step in range(MAX_STEPS):
        new_idxs = _retrieve(bm25, corpus, current_query, seen, TOP_K_PER_STEP)
        if not new_idxs and evidence_chunks:
            break
        for idx in new_idxs:
            seen.add(idx)
            evidence_chunks.append(corpus[idx])

        evidence = "\n".join(f"- {x}" for x in evidence_chunks[-15:])
        reasoning = "\n".join(reasoning_steps) if reasoning_steps else "(none yet)"

        prompt = IRCOT_PROMPT.format(
            question=question,
            evidence=evidence,
            reasoning=reasoning,
        )

        text = ""
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=180,
                )
                usage.record_response(resp)
                text = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [ERR] ircot step: {e}")
                    text = ""

        if not text:
            break

        thought, next_query, answer = _parse_step(text)
        if thought:
            reasoning_steps.append(f"Step {step + 1}: {thought}")

        if answer:
            return answer

        if next_query:
            current_query = next_query
        else:
            break

    evidence = "\n".join(f"- {x}" for x in evidence_chunks[:20])
    reasoning = "\n".join(reasoning_steps) if reasoning_steps else "(none)"
    final_prompt = FINAL_ANSWER_PROMPT.format(
        question=question,
        evidence=evidence,
        reasoning=reasoning,
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0,
                max_tokens=100,
            )
            usage.record_response(resp)
            return _extract_final_answer(resp.choices[0].message.content.strip())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERR] ircot final: {e}")
                return ""
