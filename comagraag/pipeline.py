"""
CoMaGRAG 主流程：编排四个 Agent，支持三种模式：
  - "full"      : 完整系统
  - "no_verif"  : 消融：去掉 VA
  - "no_decomp" : 消融：去掉 QDA（直接用原问题）
"""

import networkx as nx
import config
from agents import run_qda, run_gra, run_aga, run_va


def run_pipeline(question: str,
                 G: nx.DiGraph,
                 mode: str = "full",
                 passages: list = None) -> dict:
    """
    mode: "full" | "no_verif" | "no_decomp"
    passages: list of raw passage strings for fallback when KG triples are sparse
    返回: {"answer", "score", "iterations", "converged", "history"}
    """
    ctx = {
        "question":   question,
        "graph":      G,
        "passages":   passages or [],
        "subqueries": [],
        "triples":    [],
        "answer":     "",
        "reasoning":  "",
        "score":      0.0,
        "feedback":   "",
        "converged":  False,
        "history":    [],
        "iteration":  0,
    }

    best = {"answer": "", "score": 0.0, "iteration": 0}

    for t in range(1, config.MAX_ITER + 1):
        ctx["iteration"] = t

        # ── Agent 1: QDA ──────────────────────────────────
        if mode == "no_decomp":
            ctx["subqueries"] = [question]
        else:
            run_qda(ctx)

        # ── Agent 2: GRA ──────────────────────────────────
        run_gra(ctx)

        # ── Agent 3: AGA ──────────────────────────────────
        run_aga(ctx)

        # ── Agent 4: VA ──────────────────────────────────
        if mode == "no_verif":
            ctx["converged"] = True
            best = {"answer": ctx["answer"], "score": None, "iteration": t}
            break

        # no_decomp has no QDA to refine sub-questions: run once only
        if mode == "no_decomp":
            ctx["converged"] = True
            best = {"answer": ctx["answer"], "score": None, "iteration": t}
            break

        result = run_va(ctx)

        # Keep iteration-1 answer as baseline; break early if VA passes
        if t == 1:
            best = {"answer": ctx["answer"], "score": result["score"], "iteration": t}
            if result["passed"]:
                ctx["converged"] = True
                break
        elif result["passed"]:
            best = {"answer": ctx["answer"], "score": result["score"], "iteration": t}
            ctx["converged"] = True
            break

    return {
        "answer":     best["answer"],
        "score":      best["score"],
        "iterations": ctx["iteration"],
        "converged":  ctx["converged"],
        "history":    ctx["history"],
    }
