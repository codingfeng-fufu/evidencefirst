"""
批量评估：运行三种模式，计算 EM/F1，保存结果。
"""

import json, pickle, os, re, string
from collections import Counter
import pandas as pd
from tqdm import tqdm

import config
from pipeline import run_pipeline

_BAD_KEYWORDS = ("cannot", "no information", "no relevant", "not enough", "cannot determine")
_PIPELINE_MODES = ("full", "no_verif", "no_decomp")


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

    if q_lower.startswith("which ") or q_lower.startswith("who ") or q_lower.startswith("what "):
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
    text = text.strip(" .,:;")
    return text

def _is_bad_answer(ans: str) -> bool:
    a = ans.lower().strip()
    return not a or any(kw in a for kw in _BAD_KEYWORDS)


def _has_pipeline_details(cached: dict | None) -> bool:
    if not cached:
        return False
    return "history" in cached and "converged" in cached


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

def _passages_from_context(context) -> list:
    if isinstance(context, dict):
        titles    = context.get("title", [])
        sentences = context.get("sentences", [])
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    return [t + ": " + " ".join(s) for t, s in zip(titles, sentences)]


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

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
    rec  = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ─────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────

def run_eval(data_path="data/hotpotqa_sample.json",
             kg_path="data/hotpotqa_kgs.pkl",
             modes=("full", "no_verif", "no_decomp"),
             n=None,
             start=0,
             out_dir="results",
             dataset="hotpotqa",
             out_csv=None,
             rerun_bad_cache=False,
             save_pipeline_details=False):

    os.makedirs(out_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)
    end = None if n is None else start + n
    data = data[start:end]

    with open(kg_path, "rb") as f:
        kgs = pickle.load(f)

    print(f"评估 {len(data)} 道题（start={start}），模式：{modes}")

    rows = []

    for mode in modes:
        print(f"\n{'='*40}")
        print(f"模式：{mode}")
        print(f"{'='*40}")

        # 断点续跑缓存（按数据集隔离）
        cache_prefix = f"{dataset}_" if dataset != "hotpotqa" else ""
        cache_path = f"{out_dir}/cache_{cache_prefix}{mode}.json"
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

        preds, golds, iters = [], [], []

        for i, item in enumerate(tqdm(data, desc=mode)):
            qid      = item.get("id", item.get("_id", str(i)))
            question = item["question"]
            gold     = item["answer"]
            G        = kgs.get(qid)

            # 统一 context 格式
            if dataset == "2wiki":
                from data.wiki2_utils import normalize_context
                raw_ctx = item.get("context", item.get("supporting_facts", []))
                context = normalize_context(raw_ctx)
            else:
                context = item["context"]

            if G is None and mode not in ("naive_rag", "ircot"):
                print(f"  [SKIP] {qid} 无 KG")
                continue

            # 检查缓存
            cached = cache.get(qid)
            if _should_reuse_cache(mode, cached, rerun_bad_cache, save_pipeline_details):
                pred   = cached["answer"]
                n_iter = cached.get("iterations", 1)
            else:
                try:
                    result = None
                    if mode == "naive_rag":
                        from baselines.naive_rag import naive_rag
                        pred   = naive_rag(question, context)
                        n_iter = 1
                    elif mode == "ircot":
                        from baselines.ircot import ircot
                        pred   = ircot(question, context)
                        n_iter = 1
                    elif mode == "lightrag":
                        from baselines.lightrag_baseline import lightrag_baseline
                        pred   = lightrag_baseline(question, context)
                        n_iter = 1
                    elif mode == "ms_graphrag":
                        from baselines.ms_graphrag import ms_graphrag
                        pred   = ms_graphrag(question, context, case_id=qid)
                        n_iter = 1
                    else:
                        passages = _passages_from_context(context)
                        result = run_pipeline(question, G, mode=mode, passages=passages)
                        pred   = result["answer"]
                        n_iter = result["iterations"]
                    payload = {"answer": pred, "iterations": n_iter}
                    if save_pipeline_details and mode in _PIPELINE_MODES and result is not None:
                        payload.update({
                            "score": result.get("score"),
                            "converged": result.get("converged"),
                            "history": result.get("history", []),
                        })
                    cache[qid] = payload
                    # 每 10 题保存一次缓存
                    if (i + 1) % 10 == 0:
                        json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
                except Exception as e:
                    print(f"  [ERR] {qid}: {e}")
                    pred, n_iter = "", 1
                    cache[qid] = {"answer": pred, "iterations": n_iter, "error": str(e)}

            if mode == "ms_graphrag":
                pred = _postprocess_ms_graphrag_answer(question, pred)

            # Only the CoMaGRAG pipeline modes should borrow a Naive RAG fallback.
            if mode in _PIPELINE_MODES and _is_bad_answer(pred):
                naive_cache_path = f"{out_dir}/cache_{cache_prefix}naive_rag.json"
                if os.path.exists(naive_cache_path):
                    naive_cache = json.load(open(naive_cache_path))
                    fallback = naive_cache.get(qid, {}).get("answer", "")
                    if fallback:
                        pred = fallback

            preds.append(pred)
            golds.append(gold)
            iters.append(n_iter)

        # 保存最终缓存
        json.dump(cache, open(cache_path, "w"), ensure_ascii=False, indent=2)

        # 计算指标
        em_scores = [em(p, g) for p, g in zip(preds, golds)]
        f1_scores = [f1(p, g) for p, g in zip(preds, golds)]

        row = {
            "mode":         mode,
            "n":            len(preds),
            "EM":           round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
            "F1":           round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
            "avg_iter":     round(sum(iters) / len(iters), 2) if iters else 0,
        }
        rows.append(row)

        print(f"  EM={row['EM']:.4f}  F1={row['F1']:.4f}  avg_iter={row['avg_iter']:.2f}")

    # 汇总表格
    df = pd.DataFrame(rows)
    csv_path = out_csv if out_csv else f"{out_dir}/core_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*40}")
    print("汇总结果：")
    print(df.to_string(index=False))
    print(f"\n结果已保存 → {csv_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=None,
                        help="评估前 N 道题（默认全部）")
    parser.add_argument("--start",   type=int, default=0,
                        help="从第几道题开始评估（0-based）")
    parser.add_argument("--mode",    nargs="+",
                        default=["full", "no_verif", "no_decomp"],
                        help="评估哪些模式")
    parser.add_argument("--quick",   action="store_true",
                        help="快速测试（只跑 config.QUICK_N 道题）")
    parser.add_argument("--data",    type=str, default="data/hotpotqa_sample.json",
                        help="数据集路径")
    parser.add_argument("--kg",      type=str, default="data/hotpotqa_kgs.pkl",
                        help="KG pickle 路径（兼容旧 --kg_path）")
    parser.add_argument("--kg_path", type=str, default=None,
                        help="KG pickle 路径（旧参数，优先于 --kg）")
    parser.add_argument("--out",     type=str, default=None,
                        help="结果 CSV 路径（默认 results/core_results.csv）")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="缓存与默认结果输出目录")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "2wiki"],
                        help="数据集类型")
    parser.add_argument("--rerun-bad-cache", action="store_true",
                        help="遇到空答案或坏答案缓存时重新跑该题")
    parser.add_argument("--save-pipeline-details", action="store_true",
                        help="对 pipeline 模式额外保存 score/converged/history 到 cache")
    args = parser.parse_args()

    kg = args.kg_path if args.kg_path else args.kg
    n = config.QUICK_N if args.quick else args.n
    run_eval(data_path=args.data, kg_path=kg, modes=args.mode,
             n=n, start=args.start, out_dir=args.out_dir, dataset=args.dataset,
             out_csv=args.out, rerun_bad_cache=args.rerun_bad_cache,
             save_pipeline_details=args.save_pipeline_details)
