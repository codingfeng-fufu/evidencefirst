# CoMaGRAG: A Collaborative Multi-Agent Architecture with Surgical Sub-Query Repair

CoMaGRAG is a training-free, four-agent collaborative framework for multi-hop
question answering over knowledge graphs, featuring sub-query-level verification
and targeted repair without full pipeline restarts.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{feng2026comagraag,
  author    = {Zonglin Feng and Shouzhen Xu},
  title     = {{CoMaGRAG}: A Collaborative Multi-Agent Architecture
               with Surgical Sub-Query Repair for Multi-Hop Question
               Answering over Knowledge Graphs},
  booktitle = {Proc. CollaborateCom 2026},
  publisher = {Springer LNICST},
  year      = {2026}
}
```

## Setup

**Requirements:** Python 3.9+, OpenAI-compatible API key

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Optional environment variables:

- `LLM_MODEL` (default in code: `qwen-plus`)
- `LLM_BASE_URL` (default in code: DashScope-compatible endpoint)

## Repository Layout

This repository now exposes a reproducibility-oriented top-level interface:

- [evaluate.py](evaluate.py) - experiment entrypoint
- [build_kg.py](build_kg.py) - KG construction entrypoint
- [data/](data/) - dataset preparation scripts
- [prompts/](prompts/) - prompt templates / archived prompt material
- [results/](results/) - standardized per-example CSV outputs
- [scripts/](scripts/) - helper scripts for exporting, scoring, and ID verification

The historical experimental workspace remains under [comagraag/](comagraag/). See [repo_cleanup_gap_note.md](repo_cleanup_gap_note.md) for items that were only partially migrated.

## Data Preparation

### HotpotQA

```bash
python data/prepare_hotpot.py
# Downloads HotpotQA full-wiki dev split and samples 500 questions
# Output: data/hotpotqa_sample.json + data/hotpotqa_val.json
```

### 2WikiMultiHopQA

```bash
python data/prepare_2wiki.py
# Downloads 2WikiMultiHopQA dev split and samples 500 questions (seed=42)
# All four experimental conditions use the same fixed question list
# Output: data/2wiki_sample_fixed.json + data/2wiki_fixed_ids.json
```

### Build Knowledge Graphs

```bash
# HotpotQA
python build_kg.py --dataset hotpotqa --data data/hotpotqa_sample.json --out data/hotpotqa_kgs.pkl

# 2WikiMultiHopQA
python build_kg.py --dataset 2wiki --data data/2wiki_sample_fixed.json --out data/2wiki_kgs_fixed.pkl
```

## Running Experiments

### HotpotQA

```bash
# Full system
python evaluate.py --mode full --data data/hotpotqa_sample.json --kg data/hotpotqa_kgs.pkl --out results/hotpotqa_full.csv

# Ablation: No Verification
python evaluate.py --mode no_verif --data data/hotpotqa_sample.json --kg data/hotpotqa_kgs.pkl --out results/hotpotqa_no_verif.csv

# Ablation: No Decomposition
python evaluate.py --mode no_decomp --data data/hotpotqa_sample.json --kg data/hotpotqa_kgs.pkl --out results/hotpotqa_no_decomp.csv

# Baseline: Naive RAG
python evaluate.py --mode naive_rag --data data/hotpotqa_sample.json --kg data/hotpotqa_kgs.pkl --out results/hotpotqa_naive_rag.csv
```

### 2WikiMultiHopQA

```bash
for MODE in full no_verif no_decomp naive_rag; do
    python evaluate.py \
        --mode $MODE \
        --data data/2wiki_sample_fixed.json \
        --kg data/2wiki_kgs_fixed.pkl \
        --dataset 2wiki \
        --out results/2wiki_${MODE}.csv
done
```

### Compute Metrics

```bash
python scripts/eval_results.py
```

### Verify Question IDs

```bash
python scripts/verify_ids.py
```

## Expected Results

### HotpotQA (500 questions)

Paper-reported results:

| Method | EM | F1 |
|---|---:|---:|
| Naive RAG | 0.402 | 0.501 |
| Microsoft GraphRAG | 0.286 | 0.379 |
| LightRAG | 0.412 | 0.507 |
| CoMaGRAG-NoVerif | 0.422 | 0.505 |
| CoMaGRAG-NoDecomp | 0.430 | 0.504 |
| **CoMaGRAG (full)** | **0.442** | **0.522** |

Repository note:

- The standardized top-level `results/hotpotqa_*.csv` files currently reproduce
  the surviving workspace caches, which yield `EM/F1 = 0.370/0.442` for the
  full system, `0.374/0.445` for `NoVerif`, and `0.370/0.433` for `NoDecomp`.
- This mismatch between the paper-reported HotpotQA table and the surviving
  per-example caches is documented in
  [repo_cleanup_gap_note.md](repo_cleanup_gap_note.md).
- The 2Wiki rerun and GraphRAG/LightRAG numbers are backed by surviving
  result artifacts; the HotpotQA main-table caches are not fully aligned.

### 2WikiMultiHopQA (500 questions, fixed question set)

| Method | EM | F1 |
|---|---:|---:|
| Naive RAG | 0.302 | 0.336 |
| CoMaGRAG-NoVerif | 0.336 | 0.386 |
| CoMaGRAG-NoDecomp | 0.304 | 0.353 |
| **CoMaGRAG (full)** | **0.318** | **0.377** |

## Prompts

All agent prompts are in [prompts/](prompts/). Key files:

- [qda_prompt.txt](prompts/qda_prompt.txt) - Query Decomposition Agent prompt template
- [gra_prompt.txt](prompts/gra_prompt.txt) - archived / conceptual retrieval prompt note for the GRA
- [aga_prompt.txt](prompts/aga_prompt.txt) - Answer Generation Agent prompt template
- [va_scoring_prompt.txt](prompts/va_scoring_prompt.txt) - Verification scoring rubric
- [va_feedback_prompt.txt](prompts/va_feedback_prompt.txt) - Verification feedback prompt template

## Notes

- Generated data files and KG caches are intentionally gitignored. Use the scripts in [data/](data/) and [build_kg.py](build_kg.py) to recreate them.
- The top-level [results/](results/) directory contains standardized per-example CSVs suitable for review.
- Historical experiment notes remain in the repository for paper-tracking purposes and are not required to reproduce the main reported numbers.
