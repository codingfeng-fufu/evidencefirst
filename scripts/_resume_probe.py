import json, os, sys, time, hashlib, random
import requests
import networkx as nx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comagraag.build_kg import build_kg
from comagraag.pipeline import run_pipeline

results_file = str(ROOT / "probe_web/probe_results.jsonl")
data_file = str(ROOT / "data/hotpotqa_combined1000_test.jsonl")
gold_file = str(ROOT / "experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl")

# Load already-completed qids
done = set()
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            done.add(d["qid"])
print(f"Already done: {len(done)}", flush=True)

# Load questions
with open(data_file) as f:
    all_items = [json.loads(line) for line in f if line.strip()]

# Sample
rng = random.Random(123)
sampled_idx = sorted(rng.sample(range(len(all_items)), 100))
sample = [all_items[i] for i in sampled_idx]

# Load gold states
gold_states = {}
with open(gold_file) as f:
    for line in f:
        rec = json.loads(line)
        qid = rec.get("id") or rec.get("_id", "")
        chain_ok = rec.get("evidence_first_chain_complete", False)
        repair_str = rec.get("evidence_first_repair_steps", "") or ""
        fb_str = rec.get("evidence_first_fallback_steps", "") or ""
        try:
            repair = json.loads(repair_str) if repair_str.strip().startswith("[") else []
        except Exception:
            repair = []
        try:
            fb = json.loads(fb_str) if fb_str.strip().startswith("[") else []
        except Exception:
            fb = []
        if fb: s = "fallback"
        elif chain_ok and repair: s = "repaired"
        elif chain_ok: s = "checked"
        else: s = "residual"
        gold_states[qid] = s

counts = {"checked": 0, "repaired": 0, "residual": 0, "fallback": 0}
total_new = 0

for idx, item in enumerate(sample):
    qid = str(item.get("_id") or item.get("id") or sampled_idx[idx])
    if qid in done:
        continue
    total_new += 1
    question = item["question"]
    gold_s = gold_states.get(qid, "unknown")

    print(f"[{idx+1}/100] {qid}: {question[:80]}...", flush=True)

    # Tavily
    try:
        r = requests.post("https://api.tavily.com/search", json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": question, "max_results": 5, "search_depth": "basic",
        }, timeout=30)
        r.raise_for_status()
        search_results = r.json().get("results", [])
        time.sleep(0.3)
    except Exception as e:
        print(f"  [WARN] search failed: {e}", flush=True)
        search_results = []

    passages = []
    for sr in search_results:
        c = (sr.get("content") or "").strip()
        if c and len(c) > 20:
            passages.append(c)

    # KG
    if passages:
        try:
            G = build_kg(passages)
        except Exception as e:
            print(f"  [WARN] KG: {e}", flush=True)
            G = nx.DiGraph()
    else:
        G = nx.DiGraph()

    # Pipeline
    try:
        result = run_pipeline(question=question, G=G, mode="full",
                              passages=passages, variant="evidence_first")
    except Exception as e:
        print(f"  [ERR] pipeline: {e}", flush=True)
        result = {"answer": "", "diagnostics": {}}

    diag = result.get("diagnostics", {}) or {}
    chain = bool(diag.get("evidence_first_chain_complete", False))
    repair = diag.get("evidence_first_repair_steps", []) or []
    fb_steps = diag.get("evidence_first_fallback_steps", []) or []
    gap = diag.get("evidence_first_gap_type", "")

    if fb_steps: ws = "fallback"
    elif chain and repair: ws = "repaired"
    elif chain: ws = "checked"
    else: ws = "residual"

    counts[ws] += 1

    rec = {"qid": qid, "question": question, "gold_answer": item.get("answer", ""),
           "pred_answer": result.get("answer", ""), "web_state": ws, "gold_state": gold_s,
           "chain_complete": chain, "gap_type": gap}
    with open(results_file, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {gold_s} -> {ws} | chain={chain} gap={gap} ans='{result.get('answer', '')[:50]}'", flush=True)

print(f"\nDone! New: {total_new}, counts={counts}", flush=True)
