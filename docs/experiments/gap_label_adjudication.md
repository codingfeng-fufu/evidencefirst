# Structural Gap-State Proxy Plausibility Audit

Date: 2026-06-08

## Purpose

This audit addresses the reviewer question: do EvidenceFirst structural gap
labels correspond to semantically meaningful evidence gaps?

Because no external expert annotators are available, the repository includes a
no-LLM three-deterministic-proxy-view audit over saved 2Wiki artifacts. This is
not a professional human annotation study and should not be cited as one. It is
a reproducible reviewer-facing substitute for an initial inspection packet: it
applies three independent rule-based perspectives over 100 stratified examples
and exposes their rationales and disagreements.

## Command

```bash
python scripts/analyze_2wiki_evidence_support.py \
    --out-dir results/analysis/2wiki_support

python scripts/analyze_2wiki_chain_validity.py \
    --out-dir results/analysis/2wiki_chain_validity

python scripts/run_semantic_gap_adjudication.py \
    --chain results/analysis/2wiki_chain_validity/2wiki_chain_validity_per_example.csv \
    --support results/analysis/2wiki_support/2wiki_support_per_example.csv \
    --out-dir results/analysis/semantic_gap_adjudication \
    --sample-size 100 \
    --seed 20260608
```

The same command is run by:

```bash
python scripts/verify_paper_artifacts.py \
    --out-dir results/analysis/reviewer_verify
```

## Outputs

- `results/analysis/semantic_gap_adjudication/semantic_gap_adjudication_2wiki100.csv`
- `results/analysis/semantic_gap_adjudication/semantic_gap_adjudication_2wiki100.jsonl`
- `results/analysis/semantic_gap_adjudication/semantic_gap_adjudication_summary.json`

Each CSV row includes the question id, question, prediction, system gap label,
gold-chain proxy metrics, reader-support visibility metrics, three proxy labels,
majority outcome, confidence, disagreement flag, and a short rationale. The
JSONL file additionally includes support snippets and the per-perspective
annotation protocol so a reviewer can inspect individual cases without rerunning
LLM generation.

## Sampling

The audit uses a stratified 2Wiki sample because 2Wiki provides gold evidence
triples and saved per-question KGs. The fixed seed sample contains:

| System label | N |
| --- | ---: |
| complete | 34 |
| missing_entities | 35 |
| short_chain | 20 |
| disconnected | 11 |
| **Total** | **100** |

`disconnected` is rare in the full 2Wiki-500 artifact, so all 11 examples are
included. `short_chain` is oversampled relative to its full-sample frequency to
make the audit useful for reviewer inspection.

## Proxy Adjudicators

The script uses three deterministic reviewer-role perspectives:

| Proxy adjudicator | Evidence used | Judgment focus |
| --- | --- | --- |
| `strict_chain_reviewer` | Saved KG gold-entity, pair, relation, and connectivity proxies | Whether the structural label is supported by gold-chain proxies |
| `support_sufficiency_reviewer` | Reader-visible support-title recall and evidence-entity coverage | Whether the gap is isolated to graph state rather than missing context |
| `conservative_meta_reviewer` | Answer quality, reader support, and graph proxies | Whether the label is defensible to a skeptical reviewer |

Each proxy returns `match`, `mismatch`, or `ambiguous`. The final audit outcome
is the majority label; a row is flagged as disputed when the proxy perspectives
do not all agree. These labels are rule outputs over saved artifacts, not human
semantic judgments.

## Results

| Outcome | N | Rate |
| --- | ---: | ---: |
| Match | 40 | 0.4000, Wilson 95% CI [0.3094, 0.4980] |
| Mismatch | 31 | 0.3100 |
| Ambiguous | 29 | 0.2900 |

By system label:

| System label | N | Match | Mismatch | Ambiguous | Match rate | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| complete | 34 | 4 | 10 | 20 | 0.1176 | [0.0467, 0.2662] |
| disconnected | 11 | 2 | 0 | 9 | 0.1818 | [0.0514, 0.4770] |
| missing_entities | 35 | 26 | 9 | 0 | 0.7429 | [0.5793, 0.8584] |
| short_chain | 20 | 8 | 12 | 0 | 0.4000 | [0.2188, 0.6134] |

Inter-agent agreement is low:

| Agreement metric | Value |
| --- | ---: |
| strict vs. support | 0.5300 |
| strict vs. conservative | 0.4700 |
| support vs. conservative | 0.4000 |
| Fleiss kappa | 0.1836 |

## Interpretation

This audit supports a narrow claim: `missing_entities` is the most semantically
plausible gap label under the saved 2Wiki gold proxies, while `complete`,
`disconnected`, and `short_chain` remain mostly structural states rather than
semantic correctness certificates.

The low agreement and high ambiguous rate are important negative evidence. They
show that the paper should not claim semantic evidence-chain verification or
manual-level label correctness. The safe claim is that EvidenceFirst exposes
auditable structural strata whose proxy plausibility can be inspected, and whose
limitations are visible in saved artifacts.

## If Converted To A Human Study

A true human study should use a blind annotation packet that excludes `gold`,
`em`, `f1`, baseline outputs, and gold supporting facts. Annotators should label:

- whether the visible evidence is sufficient for the answer,
- whether the system gap type correctly explains the insufficiency,
- whether repair adds relevant evidence,
- whether repair changes evidence sufficiency,
- whether the final answer is supported by visible evidence.

The paper should then report pre-adjudication agreement, Cohen kappa or
Krippendorff alpha, adjudicated per-label precision with Wilson confidence
intervals, and semantic repair success rate. The current proxy audit is a
reproducible substitute for reviewer inspection, not a replacement for that
future human study.
