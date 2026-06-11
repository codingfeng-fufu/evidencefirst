# Blind Multi-Agent Gap-Label Adjudication

Date: 2026-06-09

## Purpose

This supplemental audit approximates a small reviewer-style semantic inspection
when professional annotators are unavailable. It is not a human annotation study
and should not be reported as semantic ground truth.

## Packet Boundary

The 100-case packet is derived from the stratified 2Wiki gap-label audit. Each
case includes the question, prediction, system structural gap label, structural
fields, and reader-visible snippets. It intentionally excludes gold answers,
EM/F1, baseline outputs, and the earlier deterministic proxy judgments.

Packet files:

- `results/analysis/blind_multiagent_gap_adjudication/protocol.md`
- `results/analysis/blind_multiagent_gap_adjudication/packet_01.jsonl`
- `results/analysis/blind_multiagent_gap_adjudication/packet_02.jsonl`
- `results/analysis/blind_multiagent_gap_adjudication/packet_03.jsonl`
- `results/analysis/blind_multiagent_gap_adjudication/packet_04.jsonl`

## Protocol

Four isolated adjudicators reviewed disjoint 25-case packets. Each case was
judged as:

- `match`: the structural label is a plausible explanation of the visible
  evidence state.
- `mismatch`: the visible evidence strongly contradicts the structural label.
- `ambiguous`: the packet is insufficient or mixed.

## Results

| Outcome | N | Rate |
| --- | ---: | ---: |
| Match | 53 | 0.5300 |
| Mismatch | 36 | 0.3600 |
| Ambiguous | 11 | 0.1100 |

By system gap type:

| System label | N | Match | Mismatch | Ambiguous | Match rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| complete | 34 | 23 | 5 | 6 | 0.6765 |
| disconnected | 11 | 6 | 4 | 1 | 0.5455 |
| missing_entities | 35 | 20 | 12 | 3 | 0.5714 |
| short_chain | 20 | 4 | 15 | 1 | 0.2000 |

## Interpretation

The blind packet gives a more semantically direct inspection than the
deterministic proxy audit, but it remains mixed. It supports the narrow claim
that structural labels are inspectable operational routing tags. It does not
support treating the labels as human-validated evidence sufficiency judgments.

The weakest label remains `short_chain`, which is often contradicted by visible
snippets. The safest paper claim is that these states expose audit strata and
repair/routing causes, not semantic correctness certificates.

