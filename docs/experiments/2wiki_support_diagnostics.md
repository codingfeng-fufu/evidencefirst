# 2Wiki Evidence Availability and Diagnostics

Date: 2026-06-08

## Goal

Run an offline support/evidence analysis for the completed 2Wiki-500 runs. This
analysis does not call an LLM. It uses the gold `supporting_facts` and
`evidences` fields in `comagraag/data/2wiki_sample.json` plus existing
prediction/cache artifacts.

## Artifacts

- Script: `scripts/analyze_2wiki_evidence_support.py`
- Gold-chain script: `scripts/analyze_2wiki_chain_validity.py`
- Per-example output: `results/analysis/2wiki_support/2wiki_support_per_example.csv`
- Summary output: `results/analysis/2wiki_support/2wiki_support_summary.json`
- Grouped outputs:
  - `results/analysis/2wiki_support/2wiki_support_by_chain_complete.csv`
  - `results/analysis/2wiki_support/2wiki_support_by_gap_type.csv`
  - `results/analysis/2wiki_support/2wiki_support_by_type.csv`
  - `results/analysis/2wiki_chain_validity/2wiki_chain_validity_summary.json`
  - `results/analysis/2wiki_chain_validity/2wiki_chain_validity_by_gap_type.csv`
  - `results/analysis/2wiki_chain_validity/2wiki_chain_validity_by_repair_outcome.csv`
  - `results/analysis/2wiki_chain_validity/2wiki_gap_label_gold_audit.csv`

## Main Numbers

| Input / condition | N | Support-title recall | Evidence-entity coverage | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EvidenceFirst v6 reader-full input | 500 | 1.0000 | 0.9468 | 0.6900 | 0.7581 |
| Naive BM25 top-5 input | 500 | 0.6020 | 0.6685 | - | - |
| A-RAG actual read chunks | 500 | 0.2955 | 0.3326 | - | - |
| A-RAG keyword-search found upper bound | 500 | 0.5885 | 0.6085 | - | - |

The A-RAG actual-read row uses `chunks_read_ids` from
`results/wise/2wiki_arag_full_post_predictions.jsonl` and maps chunk ids through
`external_runs/2wiki500/arag/chunks.json`. The upper-bound row uses every chunk
id returned by A-RAG keyword search logs, not only chunks actually read by the
agent.

## Diagnostic Split

| EvidenceFirst diagnostic | N | EM | F1 |
| --- | ---: | ---: | ---: |
| Chain complete | 239 | 0.7029 | 0.7681 |
| Chain incomplete | 261 | 0.6782 | 0.7490 |
| Gap: disconnected | 11 | 0.4545 | 0.5351 |
| Gap: missing entities | 217 | 0.6959 | 0.7667 |
| Gap: short chain | 33 | 0.6364 | 0.7035 |

## Gold-Chain Validity Audit

This posthoc audit uses the gold `evidences` triples only after prediction. It
does not change any prediction and must not be described as inference-time
oracle access.

| Condition | N | Gold entity coverage | Gold pair recall | Gold relation recall | Gold pair complete | Gold nodes connected | EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall saved KG | 500 | 0.7657 | 0.4778 | 0.0420 | 0.1920 | 0.2060 | 0.6900 |
| Chain complete | 239 | 0.7580 | 0.4753 | 0.0439 | 0.2176 | 0.2259 | 0.7029 |
| Chain incomplete | 261 | 0.7726 | 0.4801 | 0.0402 | 0.1686 | 0.1877 | 0.6782 |
| Repair recheck complete | 66 | 0.6876 | 0.3674 | 0.0341 | 0.1061 | 0.1212 | 0.6818 |

## Gold-Derived Gap-Label Proxy Audit

The table below asks whether each structural label selects a corresponding
posthoc gold-derived proxy target. These are imperfect proxies, not human
semantic adjudications.

| Label | Proxy target | Label N | Target N | Precision | Recall | Lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| complete | Gold pairs complete and gold nodes connected | 239 | 70 | 0.1506 | 0.5143 | 1.0759 |
| missing_entities | At least one gold evidence/answer entity missing | 217 | 316 | 0.6866 | 0.4715 | 1.0864 |
| disconnected | All gold entities present but gold nodes disconnected | 11 | 81 | 0.0909 | 0.0123 | 0.5612 |
| short_chain | Partial but incomplete gold endpoint-pair recall | 33 | 269 | 0.4545 | 0.0558 | 0.8449 |

## Three-Proxy-Reviewer Semantic Gap-Label Adjudication

The semantic proxy adjudication samples 100 stratified 2Wiki examples and uses
three no-LLM proxy reviewers: a strict gold-chain reviewer, a support
sufficiency reviewer, and a conservative meta reviewer. The audit is
reproducible and reviewer-facing, but it is not a professional human annotation
study.

| System label | N | Match | Mismatch | Ambiguous | Match rate | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| complete | 34 | 4 | 10 | 20 | 0.1176 | [0.0467, 0.2662] |
| disconnected | 11 | 2 | 0 | 9 | 0.1818 | [0.0514, 0.4770] |
| missing_entities | 35 | 26 | 9 | 0 | 0.7429 | [0.5793, 0.8584] |
| short_chain | 20 | 8 | 12 | 0 | 0.4000 | [0.2188, 0.6134] |
| Overall | 100 | 40 | 31 | 29 | 0.4000 | [0.3094, 0.4980] |

Inter-agent agreement is low: strict-vs-support agreement is 0.5300,
strict-vs-conservative agreement is 0.4700, support-vs-conservative agreement
is 0.4000, and Fleiss kappa is 0.1836. These numbers support a conservative
interpretation: the missing-entity label is often semantically plausible under
2Wiki gold proxies, while complete/disconnected/short-chain labels remain
mostly structural audit states.

## Interpretation

This result is favorable as an auxiliary analysis. It shows that the final
EvidenceFirst reader-full setting preserves gold support-title visibility for
all 2Wiki examples, while Naive BM25 and A-RAG frequently expose only a subset
of the gold support titles to the answer reader.

The diagnostic split supports a bounded traceability claim: cases marked
`chain_complete=True` are slightly more accurate than incomplete cases, and
`disconnected` / `short_chain` cases are visibly harder. The gold-chain and
proxy-label audits are deliberately less flattering: relation recall is low,
and labels align only weakly with gold-derived semantic proxies. The
three-proxy-reviewer adjudication reaches the same conclusion with explicit
per-example rationales. This should be framed as evidence availability and
diagnostic selectivity, not as a strict retrieval leaderboard or semantic chain
correctness, because LightRAG does not expose comparable per-query retrieved
support artifacts in the saved outputs.

## Paper Use

Safe placement:

- Appendix table, or
- short main-text paragraph after the 2Wiki result table.

Safe claim:

> On 2Wiki-500, EvidenceFirst v6 keeps the full gold-support passage set visible
> to the reader, while Naive BM25 top-5 and A-RAG's actual read chunks expose
> substantially less of the annotated support. EvidenceFirst diagnostics further
> identify harder cases: disconnected and short-chain gaps have lower answer
> EM/F1 than complete-chain cases. A separate posthoc gold-chain audit shows the
> limitation: structural complete/repaired states are not semantic proof of the
> gold evidence chain. A 100-example three-proxy-reviewer adjudication further
> confirms that only some structural labels, especially missing-entity gaps, are
> semantically supported under the saved 2Wiki proxies.

Do not claim:

- that this is a full retrieval benchmark against LightRAG;
- that support-title visibility alone proves graph evidence correctness;
- that every diagnostic component monotonically improves final EM/F1.
- that repair recheck completion proves semantic repair correctness.
- that the current structural gap labels have high gold-derived semantic
  precision.
- that the three-proxy-reviewer adjudication is a professional human annotation
  study.
