# Current Direction: EvidenceFirst

EvidenceFirst is the active direction for this repository.

## Core Claim

Existing KG-RAG pipelines usually retrieve evidence, generate an answer, and
then verify the answer. EvidenceFirst reverses that order: it verifies whether
the retrieved KG subgraph contains a complete multi-hop evidence chain before
answer generation.

## Method Thesis

The method combines three properties:

1. Pre-generation verification: reliability control happens before the answer is generated.
2. Evidence-chain verification: the verified object is the evidence path, not the final answer.
3. Graph-structural analysis: completeness is checked with graph connectivity and gap diagnostics after triples are extracted.

## Pipeline

```text
Question + passages
  -> per-question KG construction
  -> sub-query retrieval over the KG
  -> evidence graph construction
  -> graph-structural chain completeness check
  -> targeted gap repair when the chain is incomplete
  -> grounded generation from the verified chain
```

## Gap Types

- Missing entity: a question or candidate answer entity is not present in the evidence graph.
- Disconnected pair: both endpoints are present but lie in disconnected components.
- Bridge gap: a partial path exists, but a specific edge or bridge entity is missing.
- Short chain: a path exists but is below the minimum support length for the inferred question type.
- Empty evidence: no usable triples were retrieved.

## Evaluation Focus

EvidenceFirst should be evaluated on both answer quality and evidence quality:

- EM/F1
- chain completeness rate
- missing entity rate
- disconnected pair rate
- repair attempt rate
- repair success rate
- percentage of answers generated from verified chains
- fallback/insufficient-evidence rate

## Current Result

The current short-term target is met on the repaired-context HotpotQA first
100-question slice:

- Input: `experiments/evidence_first/data/hotpotqa_sample_hf_context.json`
- Result: `EM 0.5800 / F1 0.6495`
- Result files: `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n100_v1.*`

This repaired input preserves the local sample IDs but restores each context
from the HuggingFace HotpotQA distractor validation split. The original local
sample is context-incomplete and should not be used as the primary quality
claim for EvidenceFirst.

The current repaired-context 1000-question result boundary is:

- Frozen fresh end-to-end baseline: v4, `EM 0.5500 / F1 0.6484`
- First deterministic offline clean: v5_offline_clean, `EM 0.5570 / F1 0.6507`
- Second deterministic offline selector guard: v6_offline_selector_guard,
  `EM 0.5620 / F1 0.6564`

Use v4 for fresh-run claims. Use v5/v6 only as posthoc passes over the same v4
predictions.

## Legacy Boundary

The old CoMaGRAG four-agent surgical repair system remains useful as a baseline,
but it is no longer the primary method claim.
