"""Four Agent implementations for CoMaGRAG."""

import json, re, time
import networkx as nx
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
nltk.data.path.insert(0, '/home/u2023312337/nltk_data')

import sys
from unittest.mock import MagicMock
for _mod in ['sklearn', 'sklearn.metrics', 'sklearn.metrics.pairwise',
             'sklearn.preprocessing', 'sklearn.decomposition']:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sentence_transformers import SentenceTransformer
from openai import OpenAI

import config

client   = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)
embedder = SentenceTransformer(config.EMBED_MODEL)


def _extract_entities_nltk(text: str) -> list:
    """Extract named entities + noun phrases via NLTK."""
    entities = []
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged)
        for subtree in tree:
            if hasattr(subtree, 'label'):
                phrase = " ".join(tok for tok, _ in subtree.leaves())
                entities.append(phrase)
        chunk = []
        for tok, pos in tagged:
            if pos in ('NNP', 'NNPS', 'NN', 'NNS'):
                chunk.append(tok)
            else:
                if 2 <= len(chunk) <= 4:
                    entities.append(" ".join(chunk))
                chunk = []
        if 2 <= len(chunk) <= 4:
            entities.append(" ".join(chunk))
    except Exception:
        entities = [w for w in text.split() if w[0].isupper()]
    return list(set(entities))


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def llm_call(prompt: str, max_tokens=800, retries=4) -> str:
    """LLM call with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}] {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                raise


def parse_json_list(text: str) -> list:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


def parse_json_obj(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Agent 1: Query Decomposition Agent (QDA)
# ─────────────────────────────────────────────

QDA_FIRST = """You are a question decomposition expert.
Break the following complex multi-hop question into 2-3 simple sub-questions.

Rules:
- Each sub-question must be a standalone factual question (no placeholders like [Ans_N])
- The sub-questions together must cover all information needed to answer the original question
- If the question is already simple, return a list containing only the original question
- Output ONLY a JSON list of strings, no explanation

Example:
Question: "What is the capital of the country where the director of Inception was born?"
Output: ["Where was the director of Inception born?", "What is the capital of [that country]?"]
→ Instead write: ["Who directed Inception?", "Where was the director of Inception born?", "What is the capital of that country?"]

Question: {question}"""

QDA_RETRY = """You are a question decomposition expert.
The previous answer was rejected. Use the specific feedback below to revise the sub-questions.

Original question: {question}

Previous sub-questions:
{prev_subqueries}

Verification feedback:
{feedback}

Instructions:
- The feedback identifies SPECIFIC missing entities or facts
- Add new sub-questions that directly target the missing information
- Keep sub-questions that are still useful
- Each sub-question must be standalone (no placeholders)
Output ONLY a JSON list of strings, no explanation."""

def run_qda(ctx: dict) -> list:
    q        = ctx["question"]
    feedback = ctx.get("feedback", "")
    prev_q   = ctx.get("subqueries", [])
    t        = ctx.get("iteration", 1)

    if t == 1 or not feedback:
        prompt = QDA_FIRST.format(question=q)
    else:
        prompt = QDA_RETRY.format(
            question=q,
            prev_subqueries=json.dumps(prev_q, ensure_ascii=False),
            feedback=feedback,
        )

    text = llm_call(prompt, max_tokens=600)
    subqueries = parse_json_list(text)

    if not subqueries:
        subqueries = [q]

    ctx["subqueries"] = subqueries
    return subqueries


# ─────────────────────────────────────────────
# Agent 2: Graph Retrieval Agent (GRA)
# ─────────────────────────────────────────────

def _embed(texts: list) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)

def _entity_link_with_stage(query: str, kg_nodes: list, node_embs: np.ndarray) -> tuple[list, str]:
    """Three-stage entity linking, returns (seed nodes, stage label)."""
    if not kg_nodes:
        return [], "empty"

    entities = _extract_entities_nltk(query)

    # Stage 1: exact substring match
    exact = [n for n in kg_nodes
             if any(e.lower() in n.lower() or n.lower() in e.lower()
                    for e in entities)]
    if exact:
        return list(set(exact)), "stage1_exact"

    # Stage 2: embedding similarity
    if entities:
        ent_embs = _embed(entities)
        best = []
        for e_vec in ent_embs:
            sims = node_embs @ e_vec
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= config.ENTITY_SIM_THRESH:
                best.append(kg_nodes[best_idx])
        if best:
            return list(set(best)), "stage2_embedding"

    # Stage 3: fallback — full query embedding
    q_vec = _embed([query])[0]
    sims  = node_embs @ q_vec
    top_k = np.argsort(sims)[-config.TOP_K_FALLBACK:][::-1]
    return [kg_nodes[i] for i in top_k], "stage3_fallback"


def _entity_link(query: str, kg_nodes: list, node_embs: np.ndarray) -> list:
    """Three-stage entity linking, returns seed nodes."""
    seeds, _ = _entity_link_with_stage(query, kg_nodes, node_embs)
    return seeds


def _bfs_expand(G: nx.DiGraph, seeds: list, hops: int) -> list:
    """BFS expansion, returns list of triple strings."""
    visited  = set()
    frontier = set(seeds) & set(G.nodes())
    triples  = []

    for _ in range(hops):
        nxt = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            for _, nbr, data in G.out_edges(node, data=True):
                for rel in data.get("relations", []):
                    triples.append(f"{node} -- {rel} --> {nbr}")
                nxt.add(nbr)
            for pred, _, data in G.in_edges(node, data=True):
                for rel in data.get("relations", []):
                    triples.append(f"{pred} -- {rel} --> {node}")
                nxt.add(pred)
        frontier = nxt - visited

    return list(dict.fromkeys(triples))


def _all_kg_triples(G: nx.DiGraph) -> tuple:
    """Return (triple_strings, head_nodes, tail_nodes) for every edge in G."""
    strings, heads, tails = [], [], []
    for u, v, data in G.edges(data=True):
        for rel in data.get("relations", []):
            strings.append(f"{u} -- {rel} --> {v}")
            heads.append(u)
            tails.append(v)
    return strings, heads, tails


def _bm25_seeds(query: str, triple_strings: list, heads: list, tails: list,
                top_k: int = 20) -> set:
    """Score all triples with BM25 against query; return head+tail nodes of top-K."""
    if not triple_strings:
        return set()
    tokenized = [t.lower().split() for t in triple_strings]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[-top_k:][::-1]
    seeds = set()
    for i in top_idx:
        if scores[i] > 0:
            seeds.add(heads[i])
            seeds.add(tails[i])
    return seeds


def run_gra(ctx: dict) -> list:
    G          = ctx["graph"]
    subqueries = ctx.get("subqueries", [ctx["question"]])
    question   = ctx["question"]

    if G.number_of_nodes() == 0:
        ctx["triples"] = []
        return []

    triple_strings, heads, tails = _all_kg_triples(G)

    if not triple_strings:
        ctx["triples"] = []
        return []

    # Build combined query from question + subqueries
    combined_query = " ".join([question] + subqueries)

    # BM25 seed finding: score all triples, extract top-K head/tail nodes as seeds
    seeds = _bm25_seeds(combined_query, triple_strings, heads, tails, top_k=20)

    # BFS expand from seeds to capture multi-hop bridging edges
    triples = _bfs_expand(G, list(seeds), config.KG_HOPS)

    # Also include top BM25 triples directly (ensures we don't drop highly relevant ones)
    tokenized = [t.lower().split() for t in triple_strings]
    from rank_bm25 import BM25Okapi as _BM25
    bm25 = _BM25(tokenized)
    scores = bm25.get_scores(combined_query.lower().split())
    top_direct_idx = np.argsort(scores)[-20:][::-1]
    for i in top_direct_idx:
        if scores[i] > 0:
            triples.append(triple_strings[i])

    # Deduplicate and truncate
    triples = list(dict.fromkeys(triples))[:config.MAX_TRIPLES]

    ctx["triples"] = triples
    return triples


# ─────────────────────────────────────────────
# Agent 3: Answer Generation Agent (AGA)
# ─────────────────────────────────────────────

AGA_PROMPT = """You are a precise question-answering system.
Use ONLY the knowledge graph triples provided below. Do NOT use any external knowledge.

Knowledge Graph Triples:
{triples}

Question: {question}

Instructions:
- Reason step by step using the triples above.
- Your LAST line MUST be: "FINAL ANSWER: [answer]"
- Keep the answer concise: 1-5 words for factual questions, "yes" or "no" for yes/no questions.

Reasoning:"""

AGA_HYBRID_PROMPT = """You are a precise question-answering system.
Use ONLY the evidence provided below (knowledge graph triples and context passages).

Knowledge Graph Triples:
{triples}

Additional Context Passages:
{passages}

Question: {question}

Instructions:
- Reason step by step using all evidence above.
- Your LAST line MUST be: "FINAL ANSWER: [answer]"
- Keep the answer concise: 1-5 words for factual questions, "yes" or "no" for yes/no questions.

Reasoning:"""

def _bm25_passages(question: str, passages: list, top_k: int) -> list:
    """BM25 retrieval over raw passages; returns top-k passage strings."""
    if not passages:
        return []
    from rank_bm25 import BM25Okapi
    tokenized = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [passages[i] for i in top_idx if scores[i] > 0]


def run_aga(ctx: dict) -> str:
    triples_text = "\n".join(f"- {t}" for t in ctx.get("triples", []))
    if not triples_text:
        triples_text = "(No relevant triples retrieved)"

    # Passage fallback: when KG triples are sparse, augment with BM25 passages
    raw_passages = ctx.get("passages", [])
    n_triples = len(ctx.get("triples", []))
    if n_triples < config.PASSAGE_FALLBACK_THRESH and raw_passages:
        top_passages = _bm25_passages(ctx["question"], raw_passages, config.PASSAGE_TOP_K)
        if top_passages:
            passage_text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(top_passages))
            prompt = AGA_HYBRID_PROMPT.format(
                triples=triples_text,
                passages=passage_text,
                question=ctx["question"],
            )
        else:
            prompt = AGA_PROMPT.format(triples=triples_text, question=ctx["question"])
    else:
        prompt = AGA_PROMPT.format(triples=triples_text, question=ctx["question"])

    text = llm_call(prompt, max_tokens=800)

    # Extract final answer
    answer = ""
    for line in reversed(text.strip().split("\n")):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("FINAL ANSWER:"):
            answer = line_stripped[len("FINAL ANSWER:"):].strip()
            break
    if not answer:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        answer = lines[-1] if lines else ""

    ctx["answer"]    = answer
    ctx["reasoning"] = text
    return answer


# ─────────────────────────────────────────────
# Agent 4: Verification Agent (VA)
# ─────────────────────────────────────────────

VA_PROMPT = """You are a fact-checking expert. Evaluate whether the answer correctly answers the question.

Original Question: {question}
Candidate Answer: {answer}

Knowledge Graph Triples:
{triples}

Score on three dimensions (0.0 to 1.0):
1. factual_grounding: answer claims are supported by triples
2. completeness: answer fully addresses the question
3. consistency: no triple contradicts the answer

Then provide SPECIFIC, ACTIONABLE feedback identifying:
- The exact entity or fact that is missing (e.g., "Missing: birth country of Person X")
- A concrete sub-question to retrieve it (e.g., "Where was Person X born?")

Output ONLY valid JSON:
{{
  "factual_grounding": <float>,
  "completeness": <float>,
  "consistency": <float>,
  "feedback": "Missing: [specific entity/fact]. Suggested sub-question: [concrete question to add]"
}}"""

def run_va(ctx: dict) -> dict:
    triples_sample = ctx.get("triples", [])[:30]
    triples_text   = "\n".join(f"- {t}" for t in triples_sample) or "(No triples)"

    prompt = VA_PROMPT.format(
        question=ctx["question"],
        answer=ctx.get("answer", ""),
        triples=triples_text,
    )

    text   = llm_call(prompt, max_tokens=600)
    result = parse_json_obj(text)

    f  = max(0.0, min(1.0, float(result.get("factual_grounding", 0.5))))
    c  = max(0.0, min(1.0, float(result.get("completeness",      0.5))))
    co = max(0.0, min(1.0, float(result.get("consistency",       0.5))))

    score = config.W_FACTUAL * f + config.W_COMPLETE * c + config.W_CONSISTENT * co
    score = max(0.0, min(1.0, score))

    feedback = result.get("feedback", "Verification parse failed, please retry")

    ctx["score"]     = score
    ctx["feedback"]  = feedback
    ctx["converged"] = score >= config.THRESHOLD

    ctx.setdefault("history", []).append({
        "iteration": ctx.get("iteration", 1),
        "subqueries": ctx.get("subqueries", []),
        "n_triples":  len(ctx.get("triples", [])),
        "answer":     ctx.get("answer", ""),
        "score":      score,
        "feedback":   feedback,
    })

    return {"score": score, "feedback": feedback, "passed": score >= config.THRESHOLD}
