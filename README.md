# EvidenceFirst

EvidenceFirst is an evidence-state monitoring layer for fixed-context GraphRAG
pipelines. Before answer generation, it checks whether retrieved triples form a
candidate multi-hop path, records the structural state, and uses localized
repair when a bridge or entity is missing. The saved diagnostic fields support
offline inspection without rerunning generation.

This repository contains source code and data-preparation utilities. Generated
datasets, knowledge-graph caches, model outputs, and paper artifacts are not
included.

## Setup

```bash
git clone https://github.com/codingfeng-fufu/evidencefirst.git
cd evidencefirst
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` and, when needed, configure an OpenAI-compatible
endpoint with `LLM_MODEL` and `LLM_BASE_URL`. Download the NLTK resources once:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words
```

## Prepare Data

```bash
python data/prepare_hotpot.py
python build_kg.py --dataset hotpotqa \
  --data data/hotpotqa_sample.json \
  --out data/hotpotqa_kgs.pkl
```

The generated inputs and KG cache stay local by design. See
[data/README.md](data/README.md) for the expected paths.

## Run EvidenceFirst

```bash
python evaluate.py \
  --mode full \
  --variant evidence_first \
  --data data/hotpotqa_sample.json \
  --kg data/hotpotqa_kgs.pkl \
  --out results/evidencefirst.csv \
  --save-pipeline-details \
  --jsonl-out results/evidencefirst_details.jsonl
```

Available EvidenceFirst ablations are `without_verification`,
`without_repair`, `without_reader_context`, and
`without_answer_refinement`.

## Tests

```bash
pytest tests
```

## Layout

- `comagraag/`: pipeline, checker, graph utilities, and baseline adapters
- `data/`: dataset sampling and preparation scripts
- `prompts/`: prompt templates
- `scripts/`: lightweight scoring and export helpers
- `tests/`: unit tests for pipeline behavior

## Citation

Citation details will be added after the camera-ready record is available.
