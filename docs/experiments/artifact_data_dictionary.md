# EvidenceFirst Artifact Data Dictionary

Date: 2026-06-08

This file defines the reviewer-facing fields used by the EvidenceFirst paper
artifacts. It covers saved per-example prediction JSONL/CSV files, usage logs,
cache histories, and offline audit outputs.

## Identity And Answer Fields

| Field | Meaning |
| --- | --- |
| `_id`, `id`, `qid` | Dataset question identifier. Scripts normalize these to `qid` for joins. |
| `question` | Input multi-hop question. |
| `type` | Dataset question type, such as HotpotQA `bridge` or 2Wiki `compositional`. |
| `gold` | Gold answer used only for evaluation. Not used by inference-time EvidenceFirst. |
| `answer` | Raw or cache-level answer string produced by the pipeline. |
| `prediction` | Final normalized prediction used for EM/F1 scoring. |
| `raw_prediction` | Pre-canonicalization prediction when a posthoc canonicalizer is applied. |
| `em` | Exact match against `gold`, saved per example and rechecked by analysis scripts. |
| `f1` | Token-level answer F1 against `gold`, saved per example and summarized by scripts. |

## EvidenceFirst Diagnostic Fields

| Field | Meaning |
| --- | --- |
| `evidence_first_enabled` | Whether the EvidenceFirst path was active for this row. |
| `evidence_first_answer_type` | Coarse expected answer type inferred before answer generation. |
| `evidence_first_chain_complete` | Boolean structural state from the evidence-chain checker. It is not semantic proof of a gold evidence chain. |
| `evidence_first_chain_length` | Number of triples in the candidate structural chain when available. |
| `evidence_first_gap_type` | Structural diagnostic label. Known values are empty/`complete`, `missing_entities`, `disconnected`, `short_chain`, and `empty_evidence`. |
| `evidence_first_missing_entities` | Serialized list of entities that the checker could not map into the saved graph. |
| `evidence_first_missing_entity_count` | Count corresponding to `evidence_first_missing_entities`. |
| `evidence_first_disconnected_pair_count` | Number of entity pairs that were present but lacked a structural path. |
| `evidence_first_repair_steps` | Serialized repair steps attempted, usually `B_repair` for disconnected/bridge repair. |
| `evidence_first_fallback_steps` | Serialized fallback steps used when graph evidence was empty or unusable. |

Important boundary: these fields describe operational graph state. They do not
claim that the graph state is a complete semantic proof of the gold answer.

## Selector And Post-Processing Fields

| Field | Meaning |
| --- | --- |
| `evidence_first_postprocess_used` | Whether answer post-processing was run. |
| `evidence_first_postprocess_selected` | Whether the post-processor selected the refined evidence-grounded answer. |
| `evidence_first_postprocess_selected_v6` | Selector-guard version of the same selection flag for the v6 HotpotQA artifact. |
| `evidence_first_postprocess_answer` | Candidate answer from the post-processing or refinement stage. |
| `posthoc_clean_changed` | Whether deterministic answer cleaning changed the prediction. |
| `posthoc_selector_guard_changed` | Whether the v6 selector guard changed the prediction relative to the source artifact. |
| `canonicalized` | Whether the 2Wiki canonicalizer changed the prediction. |
| `canonicalizer` | Canonicalizer rule set name for 2Wiki. |
| `posthoc_source_variant` | Source artifact for deterministic posthoc rows. |

Posthoc selector and canonicalizer rows are deterministic transformations over
saved predictions. They should not be described as fresh retrieval or generation
runs.

## Usage Fields

| Field | Meaning |
| --- | --- |
| `llm_calls` | LLM calls charged to the current row or posthoc row. |
| `input_tokens` | Input tokens charged to the current row or posthoc row. |
| `output_tokens` | Output tokens charged to the current row or posthoc row. |
| `total_tokens` | Input plus output tokens. |
| `wall_time` | Runtime in seconds for the row when available. |
| `cache_hit` | Whether the result came from a cache rather than a fresh call. |
| `source_llm_calls`, `source_total_tokens` | Source-artifact usage carried forward by deterministic posthoc rows. |
| `avg_end_to_end_*` | Usage-summary fields combining source and posthoc costs. |

Baseline usage artifacts are incomplete and many saved `total_cost` fields are
zero, so the paper does not claim comparable monetary cost.

## Cache History Fields

Cache files such as `results/cache_wise_2wiki_evidencefirst_v6_readerfull_*.json`
store one object per `qid`.

| Field | Meaning |
| --- | --- |
| `history` | Ordered pipeline trace for the question. |
| `history[].step` | Step name, such as `evidence_chain_check`, `B_repair`, `recheck_after_B`, `context_fallback`, or `answer_postprocess`. |
| `history[].complete` | Structural recheck result for checker/recheck steps. |
| `history[].chain_length` | Chain length observed at that step. |
| `history[].gap_type` | Gap label emitted by an evidence-chain check. |
| `history[].stats.triples_extracted` | Triples extracted during repair. |
| `history[].stats.triples_added` | Triples added to the saved graph during repair. |
| `history[].stats.llm_calls` | Repair-stage LLM calls. |

`repair_to_complete` in analysis outputs means that a post-repair structural
recheck returned complete. It is not gold-path precision or causal answer
success.

## 2Wiki Support And Gold-Chain Audit Fields

The following fields are produced by `scripts/analyze_2wiki_evidence_support.py`
and `scripts/analyze_2wiki_chain_validity.py`. Gold 2Wiki evidence is used only
after prediction.

| Field | Meaning |
| --- | --- |
| `support_title_recall` | Fraction of gold supporting-fact titles visible in the reader input. |
| `evidence_entity_coverage` | Fraction of gold evidence entities visible in the reader input. |
| `gold_entity_coverage` | Fraction of gold evidence/answer entities matched in the saved KG. |
| `gold_pair_recall` | Fraction of gold evidence endpoint pairs present as graph edges. |
| `gold_relation_recall` | Fraction of gold evidence relations matched by graph edge labels. |
| `gold_pair_complete` | Whether every gold endpoint pair is present. |
| `gold_relation_complete` | Whether every gold endpoint pair and relation is present. |
| `gold_evidence_nodes_connected` | Whether all matched gold evidence nodes are connected in the saved KG. |
| `matched_gold_entities` | Human-readable list of matched gold entities. |
| `missing_gold_entities` | Human-readable list of missing gold entities. |

## Semantic Gap-Label Proxy Adjudication Fields

Produced by `scripts/run_semantic_gap_adjudication.py`.

| Field | Meaning |
| --- | --- |
| `system_gap_type` | System structural label under adjudication. |
| `strict_chain_judge` | Proxy label from the strict graph-chain reviewer. |
| `support_sufficiency_judge` | Proxy label from the support-visibility reviewer. |
| `conservative_meta_judge` | Proxy label from the conservative meta reviewer. |
| `majority_adjudication` | Majority of the three proxy labels: `match`, `mismatch`, or `ambiguous`. |
| `match_votes`, `mismatch_votes`, `ambiguous_votes` | Vote counts. |
| `adjudicated_correct` | True when the majority label is `match`. |
| `adjudication_confidence` | `high`, `medium`, or `low`, based on vote concentration. |
| `primary_disagreement` | Whether at least two proxy reviewers disagreed. |
| `rationale` | Concise per-agent rationale string. |

This proxy adjudication is a reproducible audit over saved artifacts. It is not
a professional human annotation study and should not be reported as one.
