"""
Noise-retrieval probe (N=50).
Replaces per-question curated context with BM25 retrieval from the full
HotpotQA-1000 distractor passage pool.  The EvidenceFirst pipeline is
otherwise unchanged.  Paired per-question state comparison with the
gold-context artifact shows how the evidence-state contract degrades under
noisy retrieval.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ────────────────────────────────────────────────────

def _passage_text(title: str, sents) -> str:
    body = " ".join(sents) if isinstance(sents, list) else str(sents)
    return f"{title}: {body}" if body else title


def _all_passages(items: list[dict]) -> list[str]:
    """Extract all unique passage strings from the full dataset."""
    seen = set()
    passages = []
    for item in items:
        ctx = item.get("context", [])
        if isinstance(ctx, list) and ctx and isinstance(ctx[0], str):
            for p in ctx:
                p = str(p).strip()
                if p and p not in seen:
                    seen.add(p)
                    passages.append(p)
        else:
            titles = ctx.get("title", []) if isinstance(ctx, dict) else [t for t, _ in ctx]
            sents = ctx.get("sentences", []) if isinstance(ctx, dict) else [s for _, s in ctx]
            for t, s in zip(titles, sents):
                p = _passage_text(t, s)
                if p and p not in seen:
                    seen.add(p)
                    passages.append(p)
    return passages


def _top_k(query: str, corpus: list[str], bm25: BM25Okapi, k: int = 5) -> list[str]:
    scores = bm25.get_scores(query.lower().split())
    idx = sorted(range(len(corpus)), key=lambda i: -scores[i])[:k]
    return [corpus[i] for i in idx]


def _qid(item: dict, fallback: int) -> str:
    return str(item.get("_id") or item.get("id") or fallback)


# ── Load gold-context states ───────────────────────────────────

def load_gold_states(artifact_jsonl: str | None) -> dict[str, dict]:
    """Load per-question state from an existing EvidenceFirst artifact JSONL."""
    if not artifact_jsonl or not Path(artifact_jsonl).exists():
        return {}
    states = {}
    with open(artifact_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("id") or rec.get("qid") or rec.get("_id", "")
            chain_ok = rec.get("evidence_first_chain_complete", False)
            repair_str = rec.get("evidence_first_repair_steps", "") or ""
            repair = json.loads(repair_str) if isinstance(repair_str, str) and repair_str.strip().startswith("[") else (repair_str if isinstance(repair_str, list) else [])
            fallback_str = rec.get("evidence_first_fallback_steps", "") or ""
            fallback = json.loads(fallback_str) if isinstance(fallback_str, str) and fallback_str.strip().startswith("[") else (fallback_str if isinstance(fallback_str, list) else [])
            if fallback:
                s = "fallback"
            elif chain_ok and repair:
                s = "repaired"
            elif chain_ok:
                s = "checked"
            else:
                s = "residual"
            states[qid] = {
                "state": s,
                "chain_complete": chain_ok,
                "gap_type": rec.get("gap_type", ""),
            }
    return states


# ── Main probe ─────────────────────────────────────────────────

def run_probe(
    questions_file: str,
    n: int = 50,
    seed: int = 42,
    outdir: str = "probe_noise",
    gold_artifact: str | None = None,
) -> dict:
    from comagraag.build_kg import build_kg
    from comagraag.pipeline import run_pipeline

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "probe_results.jsonl"

    # Load all items
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

    # Build BM25 index from ALL passages
    all_pass = _all_passages(all_items)
    tokenized = [p.lower().split() for p in all_pass]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index: {len(all_pass)} unique passages from {len(all_items)} questions")

    # Load gold-context states for paired comparison
    gold_states = load_gold_states(gold_artifact)
    print(f"Gold states loaded: {len(gold_states)} questions")

    jsonl_path.write_text("")

    results = []
    state_counts = {"checked": 0, "repaired": 0, "residual": 0, "fallback": 0}
    fallback_examples = []
    # Paired transition matrix
    transitions: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(sample):
        qid = _qid(item, sampled_idx[idx])
        question = item["question"]
        gold_s = gold_states.get(qid, {})

        print(f"\n[{idx + 1}/{len(sample)}] {qid}: {question[:80]}...")

        # ── Noisy retrieval ──
        noise_passages = _top_k(question, all_pass, bm25, k=5)
        print(f"  Retrieved {len(noise_passages)} passages from pool of {len(all_pass)}")

        # ── Build KG ──
        if noise_passages:
            try:
                G = build_kg(noise_passages)
            except Exception as e:
                print(f"  [WARN] KG build failed: {e}")
                import networkx as nx
                G = nx.DiGraph()
        else:
            import networkx as nx
            G = nx.DiGraph()

        # ── Run EvidenceFirst pipeline ──
        try:
            result = run_pipeline(
                question=question,
                G=G,
                mode="full",
                passages=noise_passages,
                variant="evidence_first",
            )
        except Exception as e:
            print(f"  [ERR] pipeline failed: {e}")
            result = {"answer": "", "diagnostics": {}}

        diagnostics = result.get("diagnostics", {}) or {}

        # ── Determine state ──
        chain_complete = bool(diagnostics.get("evidence_first_chain_complete", False))
        repair_steps = diagnostics.get("evidence_first_repair_steps", []) or []
        fallback_steps = diagnostics.get("evidence_first_fallback_steps", []) or []
        gap_type = diagnostics.get("evidence_first_gap_type", "")

        if fallback_steps:
            noise_state = "fallback"
        elif chain_complete and repair_steps:
            noise_state = "repaired"
        elif chain_complete:
            noise_state = "checked"
        else:
            noise_state = "residual"

        state_counts[noise_state] += 1
        if noise_state == "fallback":
            fallback_examples.append({
                "qid": qid, "question": question[:100],
                "gap_type": gap_type, "answer": result.get("answer", ""),
            })

        # Paired transition
        gold_label = gold_s.get("state", "unknown")
        transitions.setdefault(gold_label, {})
        transitions[gold_label][noise_state] = transitions[gold_label].get(noise_state, 0) + 1

        # ── Save record ──
        record = {
            "qid": qid,
            "question": question,
            "gold_answer": item.get("answer", ""),
            "pred_answer": result.get("answer", ""),
            "noise_state": noise_state,
            "gold_state": gold_label,
            "chain_complete": chain_complete,
            "chain_length": diagnostics.get("evidence_first_chain_length", 0),
            "gap_type": gap_type,
            "missing_entity_count": diagnostics.get("evidence_first_missing_entity_count", 0),
            "disconnected_pair_count": diagnostics.get("evidence_first_disconnected_pair_count", 0),
            "repair_steps": repair_steps,
            "fallback_steps": fallback_steps,
            "answer_selected": diagnostics.get("evidence_first_postprocess_selected", False),
            "kg_nodes": G.number_of_nodes(),
            "kg_edges": G.number_of_edges(),
        }
        results.append(record)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        arrow = f"{gold_label} -> {noise_state}"
        print(f"  {arrow} | chain={chain_complete} gap={gap_type} "
              f"ans='{result.get('answer', '')[:60]}'")

    # ── Summary ──
    n_total = len(results)
    summary = {
        "n_total": n_total,
        "state_counts": state_counts,
        "checked_pct": state_counts["checked"] / n_total * 100,
        "repaired_pct": state_counts["repaired"] / n_total * 100,
        "residual_pct": state_counts["residual"] / n_total * 100,
        "fallback_pct": state_counts["fallback"] / n_total * 100,
        "fallback_examples": fallback_examples,
        "transitions": transitions,
        "seed": seed,
        "sampled_indices": sampled_idx,
    }

    # Add gold-baseline distribution for the same 50 questions
    gold_dist = {}
    for idx, item in enumerate(sample):
        qid = _qid(item, sampled_idx[idx])
        gs = gold_states.get(qid, {})
        gold_dist[gs.get("state", "unknown")] = gold_dist.get(gs.get("state", "unknown"), 0) + 1
    summary["gold_state_counts"] = gold_dist

    summary_path = out / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Save SHA256 of BM25 index for verifier
    index_hash = hashlib.sha256(
        json.dumps({"corpus": all_pass, "seed": seed, "n": n}, sort_keys=True).encode()
    ).hexdigest()
    (out / "index_fingerprint.txt").write_text(index_hash + "\n")

    return summary


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Noise-Retrieval Probe")
    parser.add_argument("--data", type=str, default="data/hotpotqa_1000_test.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="probe_noise")
    parser.add_argument("--gold-artifact", type=str, default=None,
                        help="Existing EF JSONL for paired state comparison")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test: run only N questions")
    args = parser.parse_args()

    data_path = ROOT / args.data if not os.path.isabs(args.data) else Path(args.data)
    n = args.smoke if args.smoke > 0 else args.n

    summary = run_probe(
        questions_file=str(data_path),
        n=n,
        seed=args.seed,
        outdir=args.out,
        gold_artifact=args.gold_artifact,
    )

    print("\n" + "=" * 60)
    print("PROBE SUMMARY — Noise Retrieval (BM25 from distractor pool)")
    print("=" * 60)
    print(f"Total: {summary['n_total']}")
    print(f"\n{'State':<12} {'Count':>6} {'Pct':>8}")
    print("-" * 28)
    for s in ["checked", "repaired", "residual", "fallback"]:
        print(f"{s:<12} {summary['state_counts'][s]:>6} {summary[f'{s}_pct']:>7.1f}%")
    if summary.get("gold_state_counts"):
        print(f"\nGold-context baseline (same {summary['n_total']} questions):")
        for s in ["checked", "repaired", "residual", "fallback"]:
            cnt = summary["gold_state_counts"].get(s, 0)
            print(f"  {s:<12} {cnt:>6} ({cnt / summary['n_total'] * 100:.1f}%)")
    if summary.get("transitions"):
        print(f"\nPaired transitions (gold -> noise):")
        for gold_s, noise_map in sorted(summary["transitions"].items()):
            for noise_s, cnt in sorted(noise_map.items()):
                print(f"  {gold_s} -> {noise_s}: {cnt}")
    print(f"\nFallback examples: {len(summary['fallback_examples'])}")
    for ex in summary["fallback_examples"][:3]:
        print(f"  {ex['qid']}: {ex['question'][:70]}...")


if __name__ == "__main__":
    main()
