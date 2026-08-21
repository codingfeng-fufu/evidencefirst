"""
Batch evaluation for CoMaGRAG and local baselines.

The legacy defaults are preserved, while WISE reruns can isolate cache files by
max-iteration, variant, and cache tags and export per-example JSONL artifacts.
"""

import argparse
import json
import os
import pickle
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config
import usage


_BAD_KEYWORDS = (
    "cannot",
    "can not",
    "no information",
    "no relevant",
    "not enough",
    "cannot determine",
    "unknown",
    "insufficient",
    "not specified",
    "not stated",
    "not mentioned",
    "not provided",
    "no info",
    "none of the provided",
    "shortest exact answer phrase",
    "draft answer",
    "passage [",
    "instruction",
    "instructions",
    "format demands",
    "per instructions",
    "provided context",
    "context does not",
)
_PIPELINE_MODES = ("full", "no_verif", "no_decomp", "full_dual_path")
_NO_KG_MODES = ("naive_rag", "ircot", "lightrag", "ms_graphrag")
_ABLATIONS = (
    "none",
    "without_verification",
    "without_repair",
    "without_reader_context",
    "without_answer_refinement",
)


def _strip_citations_and_markup(text: str) -> str:
    text = re.sub(r"\[data:[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>`]", " ", text)
    return " ".join(text.split())


def _postprocess_ms_graphrag_answer(question: str, answer: str) -> str:
    text = _strip_citations_and_markup(answer).strip()
    if not text:
        return ""

    lowered = text.lower()
    q_lower = question.lower()

    if q_lower.startswith(("did ", "does ", "do ", "is ", "are ", "was ", "were ", "has ", "have ", "had ", "can ")):
        yn_match = re.search(r"\b(yes|no)\b", lowered)
        if yn_match:
            return yn_match.group(1)

    comp_patterns = [
        r"\b([A-Z][A-Za-z0-9'&.\-]*(?:\s+[A-Z][A-Za-z0-9'&.\-]*){0,5})\s+is\s+(?:older|younger|earlier|later)\b",
        r"\b([A-Z][A-Za-z0-9'&.\-]*(?:\s+[A-Z][A-Za-z0-9'&.\-]*){0,5})\s+(?:predates|preceded)\b",
    ]
    for pattern in comp_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip(" .,:;")

    if q_lower.startswith(("which ", "who ", "what ")):
        explicit = re.search(
            r"\b(?:therefore|thus|so|hence)\b[:,]?\s*([A-Z][^.!?\n]{0,120})",
            text,
            flags=re.I,
        )
        if explicit:
            candidate = explicit.group(1).strip(" .,:;")
            if candidate:
                return candidate

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if first_sentence:
        text = first_sentence

    text = re.sub(r"^(the answer is|answer:|therefore|thus|so)\s+", "", text, flags=re.I).strip()
    return text.strip(" .,:;")


def _is_bad_answer(ans: str) -> bool:
    a = ans.lower().strip()
    return not a or any(kw in a for kw in _BAD_KEYWORDS)


def _has_pipeline_details(cached: dict | None) -> bool:
    return bool(cached and "history" in cached and "converged" in cached)


def _should_reuse_cache(
    mode: str,
    cached: dict | None,
    rerun_bad_cache: bool,
    save_pipeline_details: bool,
) -> bool:
    if not cached:
        return False
    if rerun_bad_cache and _is_bad_answer(cached.get("answer", "")):
        return False
    if save_pipeline_details and mode in _PIPELINE_MODES and not _has_pipeline_details(cached):
        return False
    return True


def _load_items(data_path: str) -> list[dict]:
    if data_path.endswith(".jsonl"):
        with open(data_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported data format in {data_path}")
    return data


def _qid(item: dict, fallback: str) -> str:
    return str(item.get("id", item.get("_id", fallback)))


def _passages_from_context(context) -> list:
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        return context
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles = [t for t, _ in context]
        sentences = [s for _, s in context]
    return [t + ": " + " ".join(s) for t, s in zip(titles, sentences)]


def _build_global_passage_ranker(items: list[dict]) -> tuple[list[str], BM25Okapi | None]:
    passages = []
    for item in items:
        context = item.get("context", [])
        try:
            passages.extend(_passages_from_context(context))
        except Exception:
            continue
    passages = list(dict.fromkeys(p for p in passages if p))
    if not passages:
        return [], None
    return passages, BM25Okapi([p.lower().split() for p in passages])


def _global_passages_for(question: str, passages: list[str], bm25: BM25Okapi | None) -> list[str]:
    if not passages or bm25 is None:
        return []
    scores = bm25.get_scores(question.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:config.GLOBAL_CONTEXT_TOP_K]
    return [passages[i] for i in top_idx if scores[i] > 0]


def _pipeline_passages_for(
    mode: str,
    dataset: str,
    variant: str,
    question: str,
    context,
    global_passages: list[str],
    global_bm25: BM25Okapi | None,
    use_global_context: bool = True,
) -> list[str] | None:
    if mode not in _PIPELINE_MODES:
        return None

    local_passages = _passages_from_context(context)
    if dataset == "2wiki" or variant == "evidence_aug":
        return local_passages
    if not use_global_context:
        return local_passages

    global_extra = _global_passages_for(question, global_passages, global_bm25)
    return list(dict.fromkeys([*local_passages, *global_extra]))


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if not n:
        return 0.0
    prec = n / len(p_toks)
    rec = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _cache_path(
    out_dir: str,
    dataset: str,
    mode: str,
    variant: str,
    max_iter: int | None,
    cache_tag: str | None,
    ablation: str = "none",
) -> str:
    ablation_part = "" if ablation in {None, "", "none"} else f"_{ablation}"
    if cache_tag:
        return os.path.join(out_dir, f"cache_{cache_tag}{ablation_part}_{mode}.json")

    cache_prefix = f"{dataset}_" if dataset != "hotpotqa" else ""
    if variant == "default" and max_iter is None and not ablation_part:
        return os.path.join(out_dir, f"cache_{cache_prefix}{mode}.json")

    iter_part = f"n{max_iter}" if max_iter is not None else f"n{config.MAX_ITER}"
    variant_part = "" if variant == "default" else f"_{variant}"
    return os.path.join(out_dir, f"cache_{cache_prefix}{mode}_{iter_part}{variant_part}{ablation_part}.json")


def _load_naive_fallback_cache(out_dir: str, dataset: str, cache_tag: str | None) -> dict:
    """Load a compatible Naive RAG cache for bad-answer fallback."""
    candidates = []
    if cache_tag:
            candidates.append(_cache_path(out_dir, dataset, "naive_rag", "default", None, cache_tag))
    if dataset == "2wiki":
        candidates.append(_cache_path(out_dir, dataset, "naive_rag", "default", None, "wise_2wiki_naive_rag"))
    candidates.append(_cache_path(out_dir, dataset, "naive_rag", "default", None, None))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _prepare_output_file(path: str | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    open(path, "w", encoding="utf-8").close()


def _append_jsonl(path: str | None, record: dict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pipeline_diagnostics(result: dict | None, cached: dict | None = None) -> dict:
    diagnostics = {}
    if result:
        diagnostics = result.get("diagnostics", {}) or {}
    if not diagnostics and cached:
        diagnostics = cached.get("diagnostics", {}) or {}
    candidates = diagnostics.get("evidence_aug_candidates", []) or []
    passages = diagnostics.get("evidence_aug_passages", []) or []
    missing_entities = diagnostics.get("evidence_first_missing_entities", []) or []
    repair_steps = diagnostics.get("evidence_first_repair_steps", []) or []
    fallback_steps = diagnostics.get("evidence_first_fallback_steps", []) or []
    return {
        "evidence_aug_enabled": bool(diagnostics.get("evidence_aug_enabled", False)),
        "evidence_aug_question_type": diagnostics.get("evidence_aug_question_type", ""),
        "evidence_aug_candidate_count": len(candidates),
        "evidence_aug_candidates": " | ".join(str(candidate) for candidate in candidates),
        "evidence_aug_passage_count": len(passages),
        "evidence_aug_passages": " || ".join(str(passage) for passage in passages[:5]),
        "evidence_aug_challenger": diagnostics.get("evidence_aug_challenger", ""),
        "evidence_aug_selected": bool(diagnostics.get("evidence_aug_selected", False)),
        "evidence_aug_acceptance": diagnostics.get("evidence_aug_acceptance", ""),
        "evidence_aug_choice_label": diagnostics.get("evidence_aug_choice_label", ""),
        "evidence_aug_choice_answer": diagnostics.get("evidence_aug_choice_answer", ""),
        "evidence_aug_choice_confidence": diagnostics.get("evidence_aug_choice_confidence", 0.0),
        "evidence_aug_choice_rationale": diagnostics.get("evidence_aug_choice_rationale", ""),
        "evidence_first_enabled": bool(diagnostics.get("evidence_first_enabled", False)),
        "evidence_first_ablation": diagnostics.get("evidence_first_ablation", "none"),
        "evidence_first_verification_disabled": bool(diagnostics.get("evidence_first_verification_disabled", False)),
        "evidence_first_repair_disabled": bool(diagnostics.get("evidence_first_repair_disabled", False)),
        "evidence_first_reader_context_disabled": bool(diagnostics.get("evidence_first_reader_context_disabled", False)),
        "evidence_first_answer_refinement_disabled": bool(diagnostics.get("evidence_first_answer_refinement_disabled", False)),
        "evidence_first_answer_type": diagnostics.get("evidence_first_answer_type", ""),
        "evidence_first_chain_complete": bool(diagnostics.get("evidence_first_chain_complete", False)),
        "evidence_first_chain_length": diagnostics.get("evidence_first_chain_length", 0),
        "evidence_first_gap_type": diagnostics.get("evidence_first_gap_type", ""),
        "evidence_first_missing_entity_count": diagnostics.get("evidence_first_missing_entity_count", len(missing_entities)),
        "evidence_first_missing_entities": " | ".join(str(entity) for entity in missing_entities),
        "evidence_first_disconnected_pair_count": diagnostics.get("evidence_first_disconnected_pair_count", 0),
        "evidence_first_repair_steps": " | ".join(str(step) for step in repair_steps),
        "evidence_first_fallback_steps": " | ".join(str(step) for step in fallback_steps),
        "evidence_first_postprocess_used": bool(diagnostics.get("evidence_first_postprocess_used", False)),
        "evidence_first_postprocess_answer": diagnostics.get("evidence_first_postprocess_answer", ""),
        "evidence_first_postprocess_selected": bool(diagnostics.get("evidence_first_postprocess_selected", False)),
    }


def _summarize_usage(records: list[dict]) -> dict:
    fresh = [r for r in records if not r.get("cache_hit")]
    source = fresh if fresh else records
    if not source:
        return {
            "fresh_n": 0,
            "avg_llm_calls": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_latency_s": 0.0,
        }
    return {
        "fresh_n": len(fresh),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in source) / len(source), 2),
        "avg_input_tokens": round(sum(r.get("input_tokens", 0) for r in source) / len(source), 2),
        "avg_output_tokens": round(sum(r.get("output_tokens", 0) for r in source) / len(source), 2),
        "avg_latency_s": round(sum(r.get("wall_time", 0.0) for r in source) / len(source), 4),
    }


def _load_kgs(kg_path: str | None, modes: tuple[str, ...]) -> dict:
    needs_kg = any(mode not in _NO_KG_MODES for mode in modes)
    if kg_path:
        paths = [path.strip() for path in re.split(r"[,;]", kg_path) if path.strip()]
        merged = {}
        missing = []
        for path in paths:
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path, "rb") as f:
                merged.update(pickle.load(f))
        if merged:
            if missing:
                print(f"  [WARN] missing KG paths ignored: {missing}")
            return merged
    if needs_kg:
        raise FileNotFoundError(f"KG file is required for modes {modes}: {kg_path}")
    return {}


def run_eval(data_path="data/hotpotqa_sample.json",
             kg_path="data/hotpotqa_kgs.pkl",
             modes=("full", "no_verif", "no_decomp"),
             n=None,
             start=0,
             out_dir="results",
             dataset="hotpotqa",
             out_csv=None,
             rerun_bad_cache=False,
             save_pipeline_details=False,
             max_iter=None,
             variant="default",
             jsonl_out=None,
             usage_log=None,
             cache_tag=None,
             ablation="none",
             use_global_context=True):

    os.makedirs(out_dir, exist_ok=True)
    _prepare_output_file(jsonl_out)
    _prepare_output_file(usage_log)

    modes = tuple(modes)
    ablation = str(ablation or "none")
    if ablation not in _ABLATIONS:
        raise ValueError(f"Unsupported ablation: {ablation}")
    old_max_iter = config.MAX_ITER
    if max_iter is not None:
        config.MAX_ITER = int(max_iter)

    try:
        all_data = _load_items(data_path)
        global_passages, global_bm25 = (
            _build_global_passage_ranker(all_data)
            if use_global_context and any(mode in _PIPELINE_MODES for mode in modes)
            else ([], None)
        )
        data = all_data
        end = None if n is None else start + n
        data = data[start:end]
        kgs = _load_kgs(kg_path, modes)

        print(f"评估 {len(data)} 道题（start={start}），模式：{modes}")
        print(f"variant={variant}, ablation={ablation}, max_iter={config.MAX_ITER}, out_dir={out_dir}")

        rows = []

        for mode in modes:
            print(f"\n{'='*40}")
            print(f"模式：{mode}")
            print(f"{'='*40}")

            cache_path = _cache_path(out_dir, dataset, mode, variant, max_iter, cache_tag, ablation)
            cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

            preds, golds, iters, usage_records = [], [], [], []

            for i, item in enumerate(tqdm(data, desc=mode)):
                qid = _qid(item, str(i))
                question = item["question"]
                gold = item["answer"]
                G = kgs.get(qid)

                if dataset == "2wiki":
                    from data.wiki2_utils import normalize_context
                    raw_ctx = item.get("context", item.get("supporting_facts", []))
                    context = normalize_context(raw_ctx)
                else:
                    context = item["context"]

                if G is None and mode not in _NO_KG_MODES:
                    print(f"  [SKIP] {qid} 无 KG")
                    continue

                pipeline_passages = _pipeline_passages_for(
                    mode=mode,
                    dataset=dataset,
                    variant=variant,
                    question=question,
                    context=context,
                    global_passages=global_passages,
                    global_bm25=global_bm25,
                    use_global_context=use_global_context,
                )

                cached = cache.get(qid)
                result = None
                error = None
                cache_hit = _should_reuse_cache(mode, cached, rerun_bad_cache, save_pipeline_details)

                if cache_hit:
                    pred = cached["answer"]
                    if mode in _PIPELINE_MODES and _has_pipeline_details(cached):
                        from pipeline import reselect_cached_answer
                        pred = reselect_cached_answer(question, cached, pipeline_passages)
                    n_iter = cached.get("iterations", 1)
                    usage_record = {
                        "qid": qid,
                        "mode": mode,
                        "cache_hit": True,
                        "llm_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "wall_time": 0.0,
                    }
                else:
                    usage.start_case(qid)
                    try:
                        if mode == "naive_rag":
                            from baselines.naive_rag import naive_rag
                            pred = naive_rag(question, context)
                            n_iter = 1
                        elif mode == "ircot":
                            from baselines.ircot import ircot
                            pred = ircot(question, context)
                            n_iter = 1
                        elif mode == "lightrag":
                            from baselines.lightrag_baseline import lightrag_baseline
                            pred = lightrag_baseline(question, context)
                            n_iter = 1
                        elif mode == "ms_graphrag":
                            from baselines.ms_graphrag import ms_graphrag
                            pred = ms_graphrag(question, context, case_id=qid)
                            n_iter = 1
                        else:
                            if mode == "full_dual_path":
                                from pipeline import run_pipeline_dual_path
                                result = run_pipeline_dual_path(
                                    question,
                                    G,
                                    mode="full",
                                    passages=pipeline_passages,
                                    variant=variant,
                                    ablation=ablation,
                                )
                            else:
                                from pipeline import run_pipeline
                                result = run_pipeline(
                                    question,
                                    G,
                                    mode=mode,
                                    passages=pipeline_passages,
                                    variant=variant,
                                    ablation=ablation,
                                )
                            pred = result["answer"]
                            n_iter = result["iterations"]

                        usage_record = usage.finish_case({
                            "qid": qid,
                            "mode": mode,
                            "ablation": ablation,
                            "cache_hit": False,
                        })
                        payload = {
                            "answer": pred,
                            "iterations": n_iter,
                            "ablation": ablation,
                            "usage": usage_record,
                        }
                        if mode in _PIPELINE_MODES and result is not None:
                            payload["diagnostics"] = result.get("diagnostics", {})
                        if save_pipeline_details and mode in _PIPELINE_MODES and result is not None:
                            payload.update({
                                "score": result.get("score"),
                                "converged": result.get("converged"),
                                "history": result.get("history", []),
                            })
                        cache[qid] = payload
                        if (i + 1) % 10 == 0:
                            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                    except Exception as e:
                        error = str(e)
                        print(f"  [ERR] {qid}: {e}")
                        pred, n_iter = "", 1
                        usage_record = usage.finish_case({
                            "qid": qid,
                            "mode": mode,
                            "ablation": ablation,
                            "cache_hit": False,
                            "error": error,
                        })
                        cache[qid] = {
                            "answer": pred,
                            "iterations": n_iter,
                            "ablation": ablation,
                            "error": error,
                            "usage": usage_record,
                        }

                if mode == "ms_graphrag":
                    pred = _postprocess_ms_graphrag_answer(question, pred)

                if mode in _PIPELINE_MODES and _is_bad_answer(pred):
                    naive_cache = _load_naive_fallback_cache(out_dir, dataset, cache_tag)
                    fallback = naive_cache.get(qid, {}).get("answer", "")
                    if fallback:
                        pred = fallback

                em_score = em(pred, gold)
                f1_score = f1(pred, gold)
                usage_records.append(usage_record)
                preds.append(pred)
                golds.append(gold)
                iters.append(n_iter)

                example_record = {
                    "_id": qid,
                    "id": qid,
                    "mode": mode,
                    "variant": variant,
                    "ablation": ablation,
                    "max_iter": config.MAX_ITER,
                    "question": question,
                    "answer": pred,
                    "prediction": pred,
                    "gold": gold,
                    "type": item.get("type", ""),
                    "em": em_score,
                    "f1": round(f1_score, 4),
                    "iterations": n_iter,
                    "cache_hit": usage_record.get("cache_hit", False),
                    "error": error,
                    "llm_calls": usage_record.get("llm_calls", 0),
                    "input_tokens": usage_record.get("input_tokens", 0),
                    "output_tokens": usage_record.get("output_tokens", 0),
                    "total_tokens": usage_record.get("total_tokens", 0),
                    "wall_time": usage_record.get("wall_time", 0.0),
                }
                if mode in _PIPELINE_MODES:
                    example_record.update(_pipeline_diagnostics(result, cached))
                _append_jsonl(jsonl_out, example_record)
                _append_jsonl(usage_log, {
                    "_id": qid,
                    "mode": mode,
                    "variant": variant,
                    "ablation": ablation,
                    "max_iter": config.MAX_ITER,
                    "iterations": n_iter,
                    **usage_record,
                })

            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

            em_scores = [em(p, g) for p, g in zip(preds, golds)]
            f1_scores = [f1(p, g) for p, g in zip(preds, golds)]
            usage_summary = _summarize_usage(usage_records)

            row = {
                "mode": mode,
                "variant": variant,
                "ablation": ablation,
                "max_iter": config.MAX_ITER,
                "n": len(preds),
                "EM": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
                "F1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
                "avg_iter": round(sum(iters) / len(iters), 2) if iters else 0,
                **usage_summary,
            }
            rows.append(row)

            print(
                f"  EM={row['EM']:.4f}  F1={row['F1']:.4f}  "
                f"avg_iter={row['avg_iter']:.2f}  "
                f"avg_calls={row['avg_llm_calls']:.2f}"
            )

        df = pd.DataFrame(rows)
        csv_path = out_csv if out_csv else f"{out_dir}/core_results.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*40}")
        print("汇总结果：")
        print(df.to_string(index=False))
        print(f"\n结果已保存 → {csv_path}")
        return df
    finally:
        config.MAX_ITER = old_max_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="评估前 N 道题（默认全部）")
    parser.add_argument("--start", type=int, default=0, help="从第几道题开始评估（0-based）")
    parser.add_argument("--mode", nargs="+", default=["full", "no_verif", "no_decomp"], help="评估哪些模式")
    parser.add_argument("--quick", action="store_true", help="快速测试（只跑 config.QUICK_N 道题）")
    parser.add_argument("--data", type=str, default="data/hotpotqa_sample.json", help="数据集路径")
    parser.add_argument("--kg", type=str, default="data/hotpotqa_kgs.pkl", help="KG pickle 路径（兼容旧 --kg_path）")
    parser.add_argument("--kg_path", type=str, default=None, help="KG pickle 路径（旧参数，优先于 --kg）")
    parser.add_argument("--out", type=str, default=None, help="结果 CSV 路径（默认 results/core_results.csv）")
    parser.add_argument("--out-dir", type=str, default="results", help="缓存与默认结果输出目录")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "2wiki"], help="数据集类型")
    parser.add_argument("--rerun-bad-cache", action="store_true", help="遇到空答案或坏答案缓存时重新跑该题")
    parser.add_argument("--save-pipeline-details", action="store_true", help="对 pipeline 模式额外保存 score/converged/history 到 cache")
    parser.add_argument("--max-iter", type=int, default=None, help="运行时覆盖 config.MAX_ITER")
    parser.add_argument("--variant", type=str, default="default", choices=["default", "sparql_cot", "evidence_aug", "evidence_first"], help="实验变体")
    parser.add_argument("--ablation", type=str, default="none", choices=_ABLATIONS, help="EvidenceFirst 消融开关")
    parser.add_argument("--jsonl-out", type=str, default=None, help="逐题预测 JSONL 输出路径")
    parser.add_argument("--usage-log", type=str, default=None, help="逐题 usage JSONL 输出路径")
    parser.add_argument("--cache-tag", type=str, default=None, help="实验缓存标签，避免不同实验复用同一 cache")
    parser.add_argument("--no-global-context", action="store_true", help="对 pipeline 模式只使用当前样本上下文，不追加全局 BM25 passages")
    args = parser.parse_args()

    kg = args.kg_path if args.kg_path else args.kg
    n = config.QUICK_N if args.quick else args.n
    run_eval(
        data_path=args.data,
        kg_path=kg,
        modes=args.mode,
        n=n,
        start=args.start,
        out_dir=args.out_dir,
        dataset=args.dataset,
        out_csv=args.out,
        rerun_bad_cache=args.rerun_bad_cache,
        save_pipeline_details=args.save_pipeline_details,
        max_iter=args.max_iter,
        variant=args.variant,
        jsonl_out=args.jsonl_out,
        usage_log=args.usage_log,
        cache_tag=args.cache_tag,
        ablation=args.ablation,
        use_global_context=not args.no_global_context,
    )
