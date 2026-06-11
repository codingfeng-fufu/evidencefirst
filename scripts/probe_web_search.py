"""
Real Web-Search Probe (N=100) using Tavily API.
Replaces benchmark context with live web search results.
All fetched pages are archived for replayability.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, k: int = 5) -> list[dict]:
    """Search Tavily, return list of {title, url, content}."""
    import requests
    r = requests.post(TAVILY_URL, json={
        "api_key": TAVILY_KEY,
        "query": query,
        "max_results": k,
        "search_depth": "basic",
    }, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def load_gold_states(artifact_path: str) -> dict[str, str]:
    """Load per-question state from an EvidenceFirst JSONL artifact."""
    states = {}
    with open(artifact_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec.get("id") or rec.get("_id", "")
            chain_ok = rec.get("evidence_first_chain_complete", False)
            repair_str = rec.get("evidence_first_repair_steps", "") or ""
            fallback_str = rec.get("evidence_first_fallback_steps", "") or ""
            try:
                repair = json.loads(repair_str) if repair_str.strip().startswith("[") else []
            except Exception:
                repair = []
            try:
                fallback = json.loads(fallback_str) if fallback_str.strip().startswith("[") else []
            except Exception:
                fallback = []
            if fallback:
                s = "fallback"
            elif chain_ok and repair:
                s = "repaired"
            elif chain_ok:
                s = "checked"
            else:
                s = "residual"
            states[qid] = s
    return states


def run_probe(
    questions_file: str,
    n: int = 100,
    seed: int = 123,
    outdir: str = "probe_web",
    gold_artifact: str | None = None,
) -> dict:
    from comagraag.build_kg import build_kg
    from comagraag.pipeline import run_pipeline

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    archive_dir = out / "pages"
    archive_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "probe_results.jsonl"

    # Load questions
    if questions_file.endswith(".jsonl"):
        with open(questions_file, encoding="utf-8") as f:
            all_items = [json.loads(line) for line in f if line.strip()]
    else:
        with open(questions_file, encoding="utf-8") as f:
            data = json.load(f)
        all_items = data if isinstance(data, list) else data.get("data", [])

    # Fixed-seed sample
    rng = random.Random(seed)
    idx_pool = list(range(len(all_items)))
    sampled_idx = sorted(rng.sample(idx_pool, min(n, len(idx_pool))))
    sample = [all_items[i] for i in sampled_idx]

    # Load gold states
    gold_states = load_gold_states(gold_artifact) if gold_artifact else {}
    print(f"Gold states: {len(gold_states)} questions")

    jsonl_path.write_text("")
    archive_manifest = []

    results = []
    state_counts = {"checked": 0, "repaired": 0, "residual": 0, "fallback": 0}
    fallback_examples = []
    transitions: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(sample):
        qid = str(item.get("_id") or item.get("id") or sampled_idx[idx])
        question = item["question"]
        gold_s = gold_states.get(qid, "unknown")
        print(f"\n[{idx + 1}/{len(sample)}] {qid}: {question[:80]}...")

        # ── Step 1: Tavily search ──
        try:
            search_results = tavily_search(question, k=5)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [WARN] search failed: {e}")
            search_results = []

        passages = []
        for r in search_results:
            content = (r.get("content") or "").strip()
            if content and len(content) > 20:
                passages.append(content)
            # Archive
            h = hashlib.sha256(content.encode()).hexdigest()
            archive_manifest.append({"qid": qid, "url": r.get("url", ""),
                                     "title": r.get("title", ""),
                                     "sha256": h})

        print(f"  Retrieved: {len(search_results)} results, {len(passages)} passages")

        # ── Step 2: Build KG ──
        import networkx as nx
        if passages:
            try:
                G = build_kg(passages)
            except Exception as e:
                print(f"  [WARN] KG failed: {e}")
                G = nx.DiGraph()
        else:
            G = nx.DiGraph()

        # ── Step 3: EvidenceFirst pipeline ──
        try:
            result = run_pipeline(
                question=question, G=G, mode="full",
                passages=passages, variant="evidence_first",
            )
        except Exception as e:
            print(f"  [ERR] pipeline: {e}")
            result = {"answer": "", "diagnostics": {}}

        diag = result.get("diagnostics", {}) or {}
        chain = bool(diag.get("evidence_first_chain_complete", False))
        repair = diag.get("evidence_first_repair_steps", []) or []
        fallback_steps = diag.get("evidence_first_fallback_steps", []) or []
        gap = diag.get("evidence_first_gap_type", "")

        if fallback_steps:
            ws = "fallback"
        elif chain and repair:
            ws = "repaired"
        elif chain:
            ws = "checked"
        else:
            ws = "residual"

        state_counts[ws] += 1
        if ws == "fallback":
            fallback_examples.append({"qid": qid, "question": question[:120],
                                      "gap_type": gap, "answer": result.get("answer", "")})

        transitions.setdefault(gold_s, {})
        transitions[gold_s][ws] = transitions[gold_s].get(ws, 0) + 1

        record = {
            "qid": qid, "question": question, "gold_answer": item.get("answer", ""),
            "pred_answer": result.get("answer", ""),
            "web_state": ws, "gold_state": gold_s,
            "chain_complete": chain, "chain_length": diag.get("evidence_first_chain_length", 0),
            "gap_type": gap,
            "missing_entity_count": diag.get("evidence_first_missing_entity_count", 0),
            "repair_steps": len(repair), "fallback_steps": len(fallback_steps),
            "answer_selected": diag.get("evidence_first_postprocess_selected", False),
            "kg_nodes": G.number_of_nodes(), "kg_edges": G.number_of_edges(),
            "passage_count": len(passages),
        }
        results.append(record)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  {gold_s} -> {ws} | chain={chain} gap={gap} "
              f"ans='{result.get('answer', '')[:50]}'")

    # ── Summary ──
    n_total = len(results)
    summary = {
        "n_total": n_total, "state_counts": state_counts,
        "checked_pct": state_counts["checked"] / n_total * 100,
        "repaired_pct": state_counts["repaired"] / n_total * 100,
        "residual_pct": state_counts["residual"] / n_total * 100,
        "fallback_pct": state_counts["fallback"] / n_total * 100,
        "fallback_examples": fallback_examples,
        "transitions": {str(k): v for k, v in transitions.items()},
        "seed": seed,
    }
    # Gold distribution for same questions
    gold_dist = {}
    for idx, item in enumerate(sample):
        qid = str(item.get("_id") or item.get("id") or sampled_idx[idx])
        gs = gold_states.get(qid, "unknown")
        gold_dist[gs] = gold_dist.get(gs, 0) + 1
    summary["gold_state_counts"] = gold_dist

    out_summary = out / "probe_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    out_manifest = out / "archive_manifest.json"
    out_manifest.write_text(json.dumps(archive_manifest, ensure_ascii=False, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Real Web-Search Probe (Tavily)")
    parser.add_argument("--data", type=str, default="data/hotpotqa_combined1000_test.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="probe_web")
    parser.add_argument("--gold-artifact", type=str,
                        default="experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl")
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    data_path = ROOT / args.data if not os.path.isabs(args.data) else Path(args.data)
    gold_path = ROOT / args.gold_artifact if not os.path.isabs(args.gold_artifact) else Path(args.gold_artifact)
    n = args.smoke if args.smoke > 0 else args.n

    summary = run_probe(
        questions_file=str(data_path), n=n, seed=args.seed,
        outdir=args.out, gold_artifact=str(gold_path),
    )

    print("\n" + "=" * 60)
    print("PROBE SUMMARY — Real Web Search (Tavily)")
    print("=" * 60)
    print(f"Total: {summary['n_total']}")
    print(f"\n{'State':<12} {'Count':>6} {'Pct':>8}")
    print("-" * 28)
    for s in ["checked", "repaired", "residual", "fallback"]:
        print(f"{s:<12} {summary['state_counts'][s]:>6} {summary[f'{s}_pct']:>7.1f}%")
    if summary.get("gold_state_counts"):
        print(f"\nGold-context baseline (same questions):")
        for s in ["checked", "repaired", "residual", "fallback"]:
            cnt = summary["gold_state_counts"].get(s, 0)
            print(f"  {s:<12} {cnt:>6} ({cnt / summary['n_total'] * 100:.1f}%)")
    if summary.get("transitions"):
        print(f"\nPaired transitions (gold -> web):")
        for gs, nm in sorted(summary["transitions"].items()):
            for ws, cnt in sorted(nm.items()):
                print(f"  {gs} -> {ws}: {cnt}")
    print(f"\nFallback examples: {len(summary['fallback_examples'])}")
    for ex in summary["fallback_examples"][:5]:
        print(f"  {ex['qid']}: {ex['question'][:80]}...")


if __name__ == "__main__":
    main()
