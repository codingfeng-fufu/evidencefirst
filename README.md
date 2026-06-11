# EvidenceFirst

Replayable evidence-state auditing and routing for Web Knowledge-RAG.

EvidenceFirst records a pre-reader evidence-graph state (checked path, repaired
path, residual gap, or passage fallback) before generation, enabling post-hoc
failure attribution without new LLM calls. This repository contains the
implementation, baseline scripts, and saved artifacts for the paper
"EvidenceFirst: Replayable Evidence-State Auditing and Routing for Web
Knowledge-RAG" (submitted for double-blind review).

## Directory Structure

```
comagraag/          Core library (pipeline, agents, baselines, evidence checker)
scripts/            Evaluation, analysis, and probe scripts
experiments/        Experiment configurations and test scripts
data/               Dataset samples (HotpotQA, 2WikiMultiHopQA)
external_repos/     Cloned baseline repositories (IRCoT, A-RAG, HopRAG, etc.)
external_runs/      Saved baseline run configurations
```

## Environment Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your LLM API key and optional search API key
   ```

## Reproducing Results

### Quick Verification

```bash
python scripts/verify_paper_artifacts.py --check-only
```

### Main Evaluation

```bash
# EvidenceFirst end-to-end (HotpotQA-1000)
python evaluate.py --data data/hotpotqa_combined1000_test.jsonl \
  --variant evidence_first --jsonl-out results/predictions.jsonl

# EvidenceFirst (2WikiMultiHopQA-500)
python evaluate.py --dataset 2wiki --variant evidence_first \
  --jsonl-out results/2wiki_predictions.jsonl
```

### Retrieval-Noise Probe (§5.5)

```bash
python scripts/probe_web_search.py --n 100 --seed 123 --out probe_web
```

## License

License to be added upon de-anonymization.
