"""
CoMaGRAG 主流程：编排四个 Agent，支持多种模式：
  - "full"      : 完整系统
  - "no_verif"  : 消融：去掉 VA
  - "no_decomp" : 消融：去掉 QDA（直接用原问题）

  Variants:
  - "default"        : 原始CoMaGRAG
  - "evidence_aug"   : Evidence Augmentation
  - "evidence_first" : EvidenceFirst (pre-generation chain verification)
"""

import networkx as nx
import config
from agents import (
    run_aga,
    run_answer_postprocessor,
    run_gra,
    run_qda,
    run_context_answer_simple,
    run_va,
)

# Import EvidenceFirst components
from evidence_chain_checker import EvidenceChainChecker
from evidence_utils import extract_question_entities


_BAD_ANSWER_MARKERS = (
    "cannot",
    "cannot answer",
    "cannot determine",
    "given the constraints",
    "insufficient",
    "neither",
    "no triple",
    "no triples",
    "not enough",
    "no info",
    "no information",
    "no such",
    "none",
    "none of the provided context",
    "not specified",
    "not mentioned",  # Critical: LLM often outputs this when evidence is missing
    "not provided",
    "not stated",
    "not in passages",
    "shortest exact answer phrase",
    "unknown",
    "let's search again",
    "let’s search again",
    "search again",
    "looking again",
    "re-read",
    "reread",
    "not determinable",
    # Meta-reasoning leakage patterns
    "wait",
    "step ",
    "wait:",
    "but wait",
    "but note",
    "therefore,",
    "the question",
    "question demands",
    "but the",
    "also:",
    "re:",
    "passage",
    "draft answer",
    "instruction",
    "instructions",
    "final answer",
    "final answ",
    "未提及",
    "无法确定",
    "不能确定",
    "implying",
    "is phrased",
    "the only team matching",
    "originally located at an",
)


def _looks_bad_answer(answer: str) -> bool:
    lowered = str(answer or "").lower().strip()
    if not lowered or any(marker in lowered for marker in _BAD_ANSWER_MARKERS):
        return True
    tokens = lowered.split()
    if tokens and tokens[-1] in {"the", "a", "an", "of", "by", "to", "is", "was", "and", "or", "in"}:
        return True
    if lowered.startswith(("-", "check:", "however,", "but the instruction", "only ", "all other named")):
        return True
    if "-->" in lowered or " -- " in lowered:
        return True
    if lowered.startswith("and ") and len(lowered.split()) > 8:
        return True
    return False


def _norm_tokens(answer: str) -> set[str]:
    import re
    import string
    text = str(answer or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return set(text.split())


def _normalized_text(answer: str) -> str:
    return " ".join(sorted(_norm_tokens(answer)))


def _is_canonical_expansion(current_answer: str, post_answer: str, answer_type: str) -> bool:
    if answer_type not in {"person", "comparison", "entity"}:
        return False
    current_tokens = _norm_tokens(current_answer)
    post_tokens = _norm_tokens(post_answer)
    if not current_tokens or not current_tokens < post_tokens:
        return False
    if len(str(post_answer or "").split()) > len(str(current_answer or "").split()) + 2:
        return False
    return str(current_answer or "").lower() in str(post_answer or "").lower()


def _compact_answer_text(answer: str) -> str:
    return " ".join(str(answer or "").strip().strip("\"'`.,:;!?").lower().split())


def _is_short_non_yesno_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    compact = _compact_answer_text(answer)
    if not compact or compact in {"yes", "no"}:
        return False
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return False
    return len(text.split()) <= 6


def _should_select_postprocess_for_non_yesno_mismatch(
    current_answer: str,
    post_answer: str,
    answer_type: str,
) -> bool:
    if answer_type == "yesno":
        return False
    current = _compact_answer_text(current_answer)
    if current not in {"yes", "no", "wait", "but", "alternative"}:
        return False
    return _is_short_non_yesno_answer(post_answer)


def _should_select_answer_postprocess(
    current_answer: str,
    post_answer: str,
    answer_type: str = "",
) -> bool:
    """Decide whether EvidenceFirst should trust the canonicalized answer."""
    if not post_answer or _looks_bad_answer(post_answer):
        return False
    post_is_choice_answer = (
        answer_type == "choice"
        and post_answer.lower().strip() not in {"yes", "no"}
    )
    post_is_yesno_repair = (
        answer_type == "yesno"
        and _compact_answer_text(post_answer) in {"yes", "no"}
        and _looks_bad_answer(current_answer)
    )
    return (
        _looks_bad_answer(current_answer)
        or _normalized_text(post_answer) == _normalized_text(current_answer)
        or len(post_answer.split()) < len(str(current_answer or "").split())
        or post_is_choice_answer
        or post_is_yesno_repair
        or _is_canonical_expansion(current_answer, post_answer, answer_type)
        or _should_select_postprocess_for_non_yesno_mismatch(current_answer, post_answer, answer_type)
    )


def _is_context_expansion(current: str, context_answer: str, question: str) -> bool:
    current_tokens = _norm_tokens(current)
    context_tokens = _norm_tokens(context_answer)
    if not current_tokens or not context_tokens or not current_tokens < context_tokens:
        return False
    if current.lower() not in context_answer.lower():
        return False
    context_lower = context_answer.lower()
    if context_lower in question.lower():
        return True
    if "," in context_answer:
        return False
    if len(current_tokens) < 2:
        return False
    if " and " in context_lower or context_lower.startswith(("a ", "an ", "the ", "people's republic")):
        return False
    return context_lower.endswith(current.lower())


def _prefer_context_answer(best: dict, context_answer: str, question: str = "") -> bool:
    if not context_answer or _looks_bad_answer(context_answer):
        return False
    current = best.get("answer", "")
    if _looks_bad_answer(current):
        return True
    if _normalized_text(current) == _normalized_text(context_answer):
        return False
    if _is_context_expansion(current, context_answer, question):
        return True
    return False


def postprocess_context_answer(question: str, answer: str, passages: list | None = None) -> str:
    """Conservative compatibility hook for older caches and test scripts."""
    return str(answer or "").strip()


def run_context_answer(ctx: dict) -> str:
    """Answer directly from raw passages when KG evidence is weak or unusable."""
    result = run_context_answer_simple(ctx.get("question", ""), ctx.get("passages", []))
    return result.get("answer", "")


def build_evidence_augmentation(ctx: dict) -> dict:
    """
    Minimal disabled evidence-augmentation record.

    The previous experimental branch called this function without shipping an
    implementation, which made all fresh `variant=evidence_aug` runs fail. Keep
    the diagnostics shape but default to preserving the KG answer.
    """
    return {
        "enabled": False,
        "question_type": "",
        "passages": ctx.get("passages", []),
        "candidates": [],
        "choice_label": "KEEP_CURRENT",
        "choice_answer": ctx.get("kg_answer", ctx.get("answer", "")),
        "choice_confidence": 0.0,
        "choice_rationale": "evidence_aug disabled: implementation not available",
    }


def run_evidence_candidate_choice(ctx: dict) -> str:
    """Compatibility shim: do not override the KG answer."""
    return ctx.get("kg_answer", ctx.get("answer", ""))


def reselect_cached_answer(question: str, cached: dict, passages: list | None = None) -> str:
    history = cached.get("history") or []
    context_entry = next((h for h in history if h.get("iteration") == "context_fallback"), None)
    kg_entries = [h for h in history if h.get("iteration") != "context_fallback"]
    if not context_entry or not kg_entries:
        return cached.get("answer", "")

    best = {
        "answer": kg_entries[-1].get("answer", ""),
        "score": kg_entries[-1].get("score"),
        "iteration": kg_entries[-1].get("iteration", cached.get("iterations", 1)),
    }
    context_answer = context_entry.get("answer", "")
    if passages:
        context_answer = postprocess_context_answer(question, context_answer, passages)

    return context_answer if _prefer_context_answer(best, context_answer, question) else best["answer"]


def _pipeline_diagnostics(ctx: dict) -> dict:
    evidence_aug = ctx.get("evidence_aug") or {}
    gap_info = ctx.get("gap_info") or {}
    missing_entities = gap_info.get("missing", []) or []
    disconnected_pairs = gap_info.get("disconnected_pairs", []) or []
    evidence_chain = ctx.get("evidence_chain") or []
    answer_postprocess = ctx.get("answer_postprocess") or {}
    history = ctx.get("history") or []
    repair_steps = [
        h.get("step", "")
        for h in history
        if "repair" in str(h.get("step", "")).lower()
        or "enhancement" in str(h.get("step", "")).lower()
    ]
    fallback_steps = [
        h.get("step", "")
        for h in history
        if "fallback" in str(h.get("step", "")).lower()
        or "no_triples" in str(h.get("step", "")).lower()
    ]
    return {
        "evidence_aug_enabled": bool(evidence_aug.get("enabled")),
        "evidence_aug_question_type": evidence_aug.get("question_type", ""),
        "evidence_aug_passages": evidence_aug.get("passages", []),
        "evidence_aug_candidates": evidence_aug.get("candidates", []),
        "evidence_aug_challenger": evidence_aug.get("challenger_answer", ""),
        "evidence_aug_selected": bool(evidence_aug.get("selected", False)),
        "evidence_aug_acceptance": evidence_aug.get("acceptance", ""),
        "evidence_aug_choice_label": evidence_aug.get("choice_label", ""),
        "evidence_aug_choice_answer": evidence_aug.get("choice_answer", ""),
        "evidence_aug_choice_confidence": evidence_aug.get("choice_confidence", 0.0),
        "evidence_aug_choice_rationale": evidence_aug.get("choice_rationale", ""),
        "evidence_first_enabled": ctx.get("variant") == "evidence_first",
        "evidence_first_ablation": ctx.get("ablation", "none"),
        "evidence_first_verification_disabled": bool(ctx.get("disable_evidence_verification", False)),
        "evidence_first_repair_disabled": bool(ctx.get("disable_graph_repair", False)),
        "evidence_first_reader_context_disabled": bool(ctx.get("disable_reader_context", False)),
        "evidence_first_answer_refinement_disabled": bool(ctx.get("disable_answer_refinement", False)),
        "evidence_first_answer_type": ctx.get("evidence_first_answer_type", ""),
        "evidence_first_chain_complete": bool(ctx.get("chain_complete", False)),
        "evidence_first_chain_length": len(evidence_chain),
        "evidence_first_gap_type": gap_info.get("gap_type", ""),
        "evidence_first_missing_entities": missing_entities,
        "evidence_first_missing_entity_count": len(missing_entities),
        "evidence_first_disconnected_pair_count": len(disconnected_pairs),
        "evidence_first_repair_steps": repair_steps,
        "evidence_first_fallback_steps": fallback_steps,
        "evidence_first_postprocess_used": bool(answer_postprocess.get("used", False)),
        "evidence_first_postprocess_answer": answer_postprocess.get("answer", ""),
        "evidence_first_postprocess_selected": bool(answer_postprocess.get("selected", False)),
    }


def _candidate_grounded(answer: str, evidence_aug: dict) -> bool:
    answer_norm = _normalized_text(answer)
    if not answer_norm:
        return False
    if answer_norm in {"yes", "no"}:
        return True
    evidence_text = " ".join(evidence_aug.get("passages", []))
    evidence_tokens = _norm_tokens(evidence_text)
    answer_tokens = _norm_tokens(answer)
    if not answer_tokens:
        return False
    return len(answer_tokens & evidence_tokens) / len(answer_tokens) >= 0.75


def _should_accept_evidence_challenger(current: str, challenger: str, question: str, evidence_aug: dict) -> tuple[bool, str]:
    qtype = evidence_aug.get("question_type", "")
    choice_label = str(evidence_aug.get("choice_label", "") or "")
    if choice_label in {"", "KEEP_CURRENT"}:
        return False, "choice_keep_current"
    confidence = float(evidence_aug.get("choice_confidence", 0.0) or 0.0)
    if confidence < config.EVIDENCE_AUG_CHOICE_CONFIDENCE_THRESHOLD:
        return False, "low_choice_confidence"
    if not challenger or _looks_bad_answer(challenger):
        return False, "bad_challenger"
    if _normalized_text(current) == _normalized_text(challenger):
        return False, "same_as_current"
    if not _candidate_grounded(challenger, evidence_aug):
        return False, "not_grounded"
    if _looks_bad_answer(current):
        return True, "current_bad"

    current_tokens = _norm_tokens(current)
    challenger_tokens = _norm_tokens(challenger)
    question_tokens = _norm_tokens(question)
    is_expansion = current_tokens and current_tokens < challenger_tokens

    if qtype == "disjunctive_fact" and is_expansion:
        question_numbers = {token for token in question_tokens if any(ch.isdigit() for ch in token)}
        challenger_has_number = bool(question_numbers & challenger_tokens)
        if challenger_has_number and len(challenger_tokens) <= 16:
            return True, "disjunctive_fact_expansion"
    if qtype in {"award", "profession"} and is_expansion and len(challenger_tokens) <= 12:
        return True, f"{qtype}_expansion"
    if _is_context_expansion(current, challenger, question):
        return True, "canonical_expansion"
    if qtype in {"award", "profession", "yesno", "disjunctive_fact", "location", "common", "number_date"}:
        return True, f"{qtype}_candidate_choice"
    if confidence >= 0.85 and qtype in {"comparison", "person", "other"}:
        return True, f"{qtype}_high_confidence_choice"
    return False, "keep_current"


def run_pipeline(question: str,
                 G: nx.DiGraph,
                 mode: str = "full",
                 passages: list = None,
                 variant: str = "default",
                 ablation: str = "none") -> dict:
    """
    mode: "full" | "no_verif" | "no_decomp"
    variant: "default" | "evidence_aug" | "evidence_first"
    passages: list of raw passage strings for fallback when KG triples are sparse
    返回: {"answer", "score", "iterations", "converged", "history"}
    """
    ablation = str(ablation or "none")
    valid_ablations = {
        "none",
        "without_verification",
        "without_repair",
        "without_reader_context",
        "without_answer_refinement",
    }
    if ablation not in valid_ablations:
        raise ValueError(f"Unsupported ablation: {ablation}")

    disable_evidence_verification = ablation == "without_verification"
    disable_graph_repair = ablation in {"without_verification", "without_repair"}
    disable_reader_context = ablation == "without_reader_context"
    disable_answer_refinement = ablation in {"without_reader_context", "without_answer_refinement"}

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
        "variant":    variant,
        "ablation":   ablation,
        "disable_evidence_verification": disable_evidence_verification,
        "disable_graph_repair": disable_graph_repair,
        "disable_reader_context": disable_reader_context,
        "disable_answer_refinement": disable_answer_refinement,
        "evidence_aug": {},
        "evidence_chain": None,  # For evidence_first variant
        "chain_complete": False,
    }

    # Initialize EvidenceFirst checker if needed
    if variant == "evidence_first":
        checker = EvidenceChainChecker()
        ctx["evidence_checker"] = checker

    best = {"answer": "", "score": 0.0, "iteration": 0}
    all_candidates = []  # Track all iterations for fallback selection

    # EvidenceFirst: single iteration only
    max_iterations = 1 if variant == "evidence_first" else config.MAX_ITER

    for t in range(1, max_iterations + 1):
        ctx["iteration"] = t

        # ── Agent 1: QDA ──────────────────────────────────
        if mode == "no_decomp":
            ctx["subqueries"] = [question]
        else:
            run_qda(ctx)

        # ── Agent 2: GRA ──────────────────────────────────
        run_gra(ctx)

        # ── EvidenceFirst: Check if we have triples ───────
        if variant == "evidence_first" and not ctx.get("triples"):
            # No triples retrieved - fallback to passages if available
            if passages and not disable_reader_context:
                try:
                    from agents import run_context_answer_simple
                    context_result = run_context_answer_simple(question, passages)
                    ctx["answer"] = context_result.get("answer", "unknown")
                    ctx["converged"] = True
                    ctx["history"].append({
                        "iteration": t,
                        "step": "no_triples_fallback",
                        "fallback_to": "passages"
                    })
                    best = {"answer": ctx["answer"], "score": None, "iteration": t}
                    break
                except Exception as e:
                    # If passages fallback fails, continue with empty triples
                    ctx["answer"] = "unknown"
                    ctx["converged"] = True
                    best = {"answer": "unknown", "score": None, "iteration": t}
                    break
            else:
                # No triples and no passages - return unknown
                ctx["answer"] = "unknown"
                ctx["converged"] = True
                ctx["history"].append({
                    "iteration": t,
                    "step": "no_triples_no_passages",
                    "result": "unknown"
                })
                best = {"answer": "unknown", "score": None, "iteration": t}
                break

        # ── EvidenceFirst: Evidence Chain Check + A+B Enhancement ───────────
        if variant == "evidence_first":
            checker = ctx["evidence_checker"]
            triples = ctx.get("triples", [])

            if triples:
                if disable_evidence_verification:
                    is_complete = False
                    gap_info = {}
                    evidence_chain = []
                    ctx["chain_complete"] = False
                    ctx["gap_info"] = {}
                    ctx["history"].append({
                        "iteration": t,
                        "step": "evidence_chain_check_disabled",
                        "ablation": ablation,
                    })
                else:
                    # Check evidence chain completeness
                    is_complete, gap_info, evidence_chain = checker.check_evidence_chain_from_strings(
                        triples, question
                    )

                    ctx["evidence_first_answer_type"] = getattr(checker, "last_answer_type", "")
                    ctx["chain_complete"] = is_complete
                    ctx["gap_info"] = gap_info

                # A+B Solution: Enhance KG if chain is incomplete
                if not disable_graph_repair and not is_complete and gap_info:
                    from kg_enhancement import (
                        enhance_kg_with_missing_entities,
                        repair_disconnected_entities,
                        add_triples_to_graph
                    )
                    from evidence_utils import build_graph_from_triple_strings

                    # Build sub-graph from triples
                    sub_graph = build_graph_from_triple_strings(triples)

                    # Get context passages for LLM extraction
                    context = "\n\n".join(ctx.get("passages", []))

                    # A+ Solution: Handle missing entities first
                    if gap_info.get("gap_type") == "missing_entities":
                        missing = gap_info.get("missing", [])
                        if missing and context:
                            stats = enhance_kg_with_missing_entities(
                                missing, context, sub_graph
                            )
                            ctx["history"].append({
                                "iteration": t,
                                "step": "A+_enhancement",
                                "stats": stats
                            })

                            # Re-check after A+ to get updated gap_info with disconnected_pairs
                            enhanced_triples = []
                            for u, v, data in sub_graph.edges(data=True):
                                relation = data.get('relation', 'related_to')
                                enhanced_triples.append(f"{u} -- {relation} --> {v}")

                            is_complete_after_a, gap_info_after_a, evidence_chain_after_a = checker.check_evidence_chain_from_strings(
                                enhanced_triples, question
                            )
                            ctx["evidence_first_answer_type"] = getattr(checker, "last_answer_type", "")

                            ctx["history"].append({
                                "iteration": t,
                                "step": "recheck_after_A+",
                                "complete": is_complete_after_a,
                                "chain_length": len(evidence_chain_after_a) if evidence_chain_after_a else 0
                            })

                            # If still incomplete after A+, try B solution
                            if not is_complete_after_a and gap_info_after_a:
                                disconnected_pairs = gap_info_after_a.get("disconnected_pairs", [])
                                if disconnected_pairs and context:
                                    stats = repair_disconnected_entities(
                                        disconnected_pairs, context, sub_graph, max_pairs=5
                                    )
                                    ctx["history"].append({
                                        "iteration": t,
                                        "step": "B_repair",
                                        "stats": stats
                                    })

                                    # Final re-check after A+B
                                    enhanced_triples = []
                                    for u, v, data in sub_graph.edges(data=True):
                                        relation = data.get('relation', 'related_to')
                                        enhanced_triples.append(f"{u} -- {relation} --> {v}")

                                    is_complete, gap_info, evidence_chain = checker.check_evidence_chain_from_strings(
                                        enhanced_triples, question
                                    )

                                    ctx["evidence_first_answer_type"] = getattr(checker, "last_answer_type", "")
                                    ctx["chain_complete"] = is_complete
                                    ctx["gap_info"] = gap_info
                                    ctx["triples"] = enhanced_triples

                                    ctx["history"].append({
                                        "iteration": t,
                                        "step": "recheck_after_A+B",
                                        "complete": is_complete,
                                        "chain_length": len(evidence_chain) if evidence_chain else 0
                                    })
                                else:
                                    # No disconnected pairs or no context, use A+ result
                                    is_complete = is_complete_after_a
                                    gap_info = gap_info_after_a
                                    evidence_chain = evidence_chain_after_a
                                    ctx["chain_complete"] = is_complete
                                    ctx["gap_info"] = gap_info
                                    ctx["triples"] = enhanced_triples
                            else:
                                # A+ succeeded, update context
                                is_complete = is_complete_after_a
                                gap_info = gap_info_after_a
                                evidence_chain = evidence_chain_after_a
                                ctx["chain_complete"] = is_complete
                                ctx["gap_info"] = gap_info
                                ctx["triples"] = enhanced_triples

                    # B Solution only: Handle disconnected entities when no missing entities
                    elif gap_info.get("disconnected_pairs"):
                        disconnected_pairs = gap_info.get("disconnected_pairs", [])
                        if disconnected_pairs and context:
                            stats = repair_disconnected_entities(
                                disconnected_pairs, context, sub_graph, max_pairs=5
                            )
                            ctx["history"].append({
                                "iteration": t,
                                "step": "B_repair",
                                "stats": stats
                            })

                            # Re-check after B
                            enhanced_triples = []
                            for u, v, data in sub_graph.edges(data=True):
                                relation = data.get('relation', 'related_to')
                                enhanced_triples.append(f"{u} -- {relation} --> {v}")

                            is_complete, gap_info, evidence_chain = checker.check_evidence_chain_from_strings(
                                enhanced_triples, question
                            )
                            ctx["evidence_first_answer_type"] = getattr(checker, "last_answer_type", "")
                            ctx["chain_complete"] = is_complete
                            ctx["gap_info"] = gap_info
                            ctx["triples"] = enhanced_triples

                            ctx["history"].append({
                                "iteration": t,
                                "step": "recheck_after_B",
                                "complete": is_complete,
                                "chain_length": len(evidence_chain) if evidence_chain else 0
                            })

                if is_complete and evidence_chain:
                    # Use only the evidence chain for generation
                    ctx["evidence_chain"] = evidence_chain
                    ctx["history"].append({
                        "iteration": t,
                        "step": "evidence_chain_check",
                        "complete": True,
                        "chain_length": len(evidence_chain)
                    })
                else:
                    # Chain incomplete - use all triples as fallback
                    ctx["evidence_chain"] = ctx.get("triples", triples)
                    ctx["history"].append({
                        "iteration": t,
                        "step": "evidence_chain_check",
                        "complete": False,
                        "gap_type": gap_info.get("gap_type") if gap_info else "unknown"
                    })
            else:
                # No triples retrieved
                ctx["evidence_chain"] = []
                ctx["chain_complete"] = False

        # ── Agent 3: AGA ──────────────────────────────────
        run_aga(ctx)

        # ── EvidenceFirst: Skip VA, converge immediately ──
        if variant == "evidence_first":
            ctx["converged"] = True
            best = {"answer": ctx["answer"], "score": None, "iteration": t}
            break

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

        # Track this iteration's candidate
        all_candidates.append({
            "answer": ctx["answer"],
            "score": result["score"],
            "iteration": t
        })

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

    # If not converged, select the best scoring answer from all iterations
    if not ctx["converged"] and all_candidates:
        best_candidate = max(all_candidates, key=lambda x: x["score"] if x["score"] is not None else 0.0)
        if best_candidate["score"] is not None and best_candidate["score"] > best.get("score", 0.0):
            best = best_candidate

    if mode == "full" and variant == "evidence_aug" and ctx.get("passages"):
        kg_answer = best.get("answer", "")
        ctx["kg_answer"] = kg_answer
        ctx["answer"] = kg_answer
        ctx["evidence_aug"] = build_evidence_augmentation(ctx)
        challenger = run_evidence_candidate_choice(ctx)
        ctx["answer"] = challenger
        selected, acceptance = _should_accept_evidence_challenger(kg_answer, challenger, question, ctx["evidence_aug"])
        ctx["evidence_aug"]["challenger_answer"] = challenger
        ctx["evidence_aug"]["selected"] = selected
        ctx["evidence_aug"]["acceptance"] = acceptance
        ctx.setdefault("history", []).append({
            "iteration": "evidence_aug",
            "answer": challenger,
            "selected": selected,
            "acceptance": acceptance,
            "question_type": ctx["evidence_aug"].get("question_type", ""),
            "n_passages": len(ctx["evidence_aug"].get("passages", [])),
            "n_candidates": len(ctx["evidence_aug"].get("candidates", [])),
            "candidates": ctx["evidence_aug"].get("candidates", []),
            "choice_label": ctx["evidence_aug"].get("choice_label", ""),
            "choice_confidence": ctx["evidence_aug"].get("choice_confidence", 0.0),
            "choice_rationale": ctx["evidence_aug"].get("choice_rationale", ""),
        })
        if selected:
            best = {
                "answer": challenger,
                "score": best.get("score"),
                "iteration": best.get("iteration", ctx["iteration"]),
            }

    if mode == "full" and variant != "evidence_aug" and ctx.get("passages") and not disable_reader_context:
        context_answer = run_context_answer(ctx)
        selected = _prefer_context_answer(best, context_answer, question)
        ctx.setdefault("history", []).append({
            "iteration": "context_fallback",
            "answer": context_answer,
            "selected": selected,
        })
        if selected:
            best = {
                "answer": context_answer,
                "score": best.get("score"),
                "iteration": best.get("iteration", ctx["iteration"]),
            }

    if (
        mode == "full"
        and variant == "evidence_first"
        and config.EVIDENCE_FIRST_POSTPROCESS
        and ctx.get("passages")
        and not disable_answer_refinement
    ):
        post = run_answer_postprocessor(
            question,
            best.get("answer", ""),
            ctx.get("passages", []),
            top_k=config.EVIDENCE_FIRST_POSTPROCESS_TOP_K,
            answer_type=ctx.get("evidence_first_answer_type", "entity"),
        )
        post_answer = post.get("answer", "")
        current_answer = best.get("answer", "")
        answer_type = ctx.get("evidence_first_answer_type", "")
        selected = _should_select_answer_postprocess(
            current_answer=current_answer,
            post_answer=post_answer,
            answer_type=answer_type,
        )
        post["selected"] = selected
        ctx["answer_postprocess"] = post
        ctx.setdefault("history", []).append({
            "iteration": "answer_postprocess",
            "draft_answer": best.get("answer", ""),
            "answer": post_answer,
            "selected": selected,
        })
        if selected:
            best = {
                "answer": post_answer,
                "score": best.get("score"),
                "iteration": best.get("iteration", ctx["iteration"]),
            }

    return {
        "answer":     best["answer"],
        "score":      best["score"],
        "iterations": ctx["iteration"],
        "converged":  ctx["converged"],
        "history":    ctx["history"],
        "diagnostics": _pipeline_diagnostics(ctx),
        "evidence_chain": ctx.get("evidence_chain"),  # EvidenceFirst
        "chain_complete": ctx.get("chain_complete", False),  # EvidenceFirst
    }


# ============================================================================
# Dual Path Architecture - Answer Selection Logic
# ============================================================================

def _looks_bad_answer_dual_path(answer: str) -> bool:
    """
    Detect obviously bad answers.
    
    Bad answer indicators:
    - Empty or too short
    - Contains markers like "unknown", "cannot"
    - Too long (>20 words)
    - Contains meta-language about triples/instructions
    """
    if not answer or len(answer.strip()) < 2:
        return True
    
    lowered = answer.lower().strip()
    
    bad_markers = [
        "unknown", "cannot", "can not", "insufficient",
        "not enough", "no information", "not specified",
        "provided context", "context does not",
        "triple", "triples", "instruction", "instructions",
        "graph does not", "given the constraints"
    ]
    
    if any(marker in lowered for marker in bad_markers):
        return True
    
    if len(answer.split()) > 20:
        return True
    
    if lowered.startswith(("however,", "but the", "check:", "-")):
        return True
    
    return False


def _detect_question_type(question: str) -> str:
    """
    Detect question type for path selection.
    
    Returns: yesno, comparison, location, number_date, person, common, other
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
    """Check if answer is yes/no"""
    normalized = answer.lower().strip().rstrip(".")
    return normalized in ["yes", "no"]


def _select_answer_dual_path(kg_result: dict,
                             context_result: dict,
                             question: str) -> dict:
    """
    Smart answer selector for dual-path architecture.
    
    Selection rules (priority order):
    1. Bad answer detection: prefer good over bad
    2. Question type matching: certain types prefer certain paths
    3. Score/confidence comparison
    4. Default: KG preferred (because verifiable)
    """
    kg_answer = kg_result.get("answer", "")
    context_answer = context_result.get("answer", "")
    
    kg_bad = _looks_bad_answer_dual_path(kg_answer)
    context_bad = _looks_bad_answer_dual_path(context_answer)
    
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
    
    # Both bad
    if kg_bad and context_bad:
        return {
            "answer": kg_answer,
            "source": "kg_both_bad",
            "reason": "both_bad",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }
    
    # Rule 2: Question type optimization
    qtype = _detect_question_type(question)
    
    # Yes/No: prefer context (more direct)
    if qtype == "yesno" and _is_yesno_answer(context_answer):
        return {
            "answer": context_answer,
            "source": "context_yesno",
            "reason": "yesno_match",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }
    
    # Comparison: prefer context (more accurate)
    if qtype == "comparison" and len(context_answer.split()) <= 8:
        return {
            "answer": context_answer,
            "source": "context_comparison",
            "reason": "comparison_concise",
            "kg_answer": kg_answer,
            "context_answer": context_answer
        }
    
    # Rule 3: Score comparison
    kg_score = float(kg_result.get("score") or 0.0)
    context_conf = float(context_result.get("confidence") or 0.0)
    
    # KG high score
    if kg_score >= 0.6:
        return {
            "answer": kg_answer,
            "source": "kg_high_score",
            "reason": f"kg_score_{kg_score:.2f}",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "kg_score": kg_score
        }
    
    # Context high confidence + KG low score
    if context_conf >= 0.7 and kg_score < 0.4:
        return {
            "answer": context_answer,
            "source": "context_high_conf",
            "reason": f"context_conf_{context_conf:.2f}_kg_low",
            "kg_answer": kg_answer,
            "context_answer": context_answer,
            "context_confidence": context_conf
        }
    
    # Rule 4: Default - KG preferred (verifiable)
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
                           variant: str = "default",
                           ablation: str = "none") -> dict:
    """
    Dual-path architecture pipeline.
    
    Path 1: KG-based (existing pipeline)
    Path 2: Context-based (BM25 + LLM)
           ↓
       Answer Selector
           ↓
       Final Answer
    
    This architecture provides redundancy: if KG retrieval fails,
    context-based path can still provide a good answer.
    
    Expected improvement: +5-7% EM (45.4% → 50-52%)
    """
    # Path 1: Run KG-based pipeline
    kg_result = run_pipeline(
        question=question,
        G=G,
        mode=mode,
        passages=passages,
        variant=variant,
        ablation=ablation,
    )
    
    # Path 2: Run Context-based path
    context_result = {"answer": "", "source": "context", "confidence": 0.0}

    if passages:
        try:
            context_result = run_context_answer_simple(question, passages)
        except Exception as e:
            print(f"[Dual Path] Context path failed: {e}")
            context_result = {"answer": "", "source": "context", "confidence": 0.0}
    
    # Answer Selection
    final_result = _select_answer_dual_path(kg_result, context_result, question)
    
    # Preserve trace information
    final_result.update({
        "iterations": kg_result.get("iterations", 1),
        "converged": kg_result.get("converged", True),
        "history": kg_result.get("history", []),
        "mode": f"{mode}_dual_path"
    })
    
    return final_result
