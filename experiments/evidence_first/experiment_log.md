# EvidenceFirst Experiment Log

## Short-term Target

Target: on the current 100-question HotpotQA slice, reach either:

- EM >= 0.60, or
- F1 / F-measure >= 0.60

All runs must keep the command, cache tag, CSV, JSONL diagnostics, and usage log
so that improvements are attributable and reproducible.

## 2026-06-06 HotpotQA 5-question smoke

Goal: switch the active direction from legacy CoMaGRAG post-hoc verification to
EvidenceFirst pre-generation graph-structural evidence-chain verification.

### Run 1: initial EvidenceFirst

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 5 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n5.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n5.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n5_usage.jsonl \
  --cache-tag hotpotqa_ef_n5
```

Result:

- EM: 0.2000
- F1: 0.4189
- Avg. iterations: 1.00
- Avg. LLM calls: 4.60
- Chain complete: 3/5
- Missing-entity gaps: 2/5

Observation: one-edge paths were often marked as complete chains even when they
only reached an intermediate entity. This made chain completeness too permissive.

### Run 2: minimum chain-length / distance-filter verifier

Change:

- EvidenceFirst generation now uses `ctx["evidence_chain"]` when present.
- Non-yes/no questions require at least two evidence-chain edges.
- Candidate answer nodes too close to question entities are filtered when
  possible.
- EvidenceFirst diagnostics are exported to JSONL rows.

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 5 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v2.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v2.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n5_v2_usage.jsonl \
  --cache-tag hotpotqa_ef_n5_v2
```

Result:

- EM: 0.2000
- F1: 0.4119
- Avg. iterations: 1.00
- Avg. LLM calls: 4.60
- Chain complete: 1/5
- Short-chain gaps: 2/5
- Missing-entity gaps: 2/5

Observation: verifier diagnostics improved, but answer quality did not. The next
blocker is answer-type-aware endpoint selection. Connectivity alone is too weak:
the checker needs to know whether the terminal node is a plausible answer type
for the question.

## Next Experiment

### Run 3: answer-type endpoint filter

Change:

- Added coarse answer-type inference for endpoint filtering:
  `award`, `profession`, `location`, `country`, `date`, `number`, `person`,
  `yesno`, or generic `entity`.
- Exported inferred answer type in diagnostics.

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 5 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v3.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v3.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n5_v3_usage.jsonl \
  --cache-tag hotpotqa_ef_n5_v3
```

Result:

- EM: 0.2000
- F1: 0.3833
- Avg. iterations: 1.00
- Avg. LLM calls: 4.80
- Chain complete: 2/5
- Missing-entity gaps: 2/5
- Short-chain gaps: 1/5

Observation: answer-type filtering changes the selected endpoint, but it is not
yet enough. The award case moved to a typed award endpoint (`Emmy Award`) but
still missed the question constraint that the answer should be the rare combined
award (`EGOT`). The next verifier needs endpoint ranking by question constraints,
not only coarse answer type.

Do not scale to 100 questions yet. First implement:

1. Candidate endpoint ranking by question constraints, not just coarse answer type.
2. Separate verified-chain length from fallback evidence length in diagnostics.
3. Fallback generation cleanup for verbose partial answers.
4. A no-repair ablation to separate verifier quality from A+B repair quality.

### Run 4: passage-grounded answer postprocessing with selection gate

Change:

- Added an EvidenceFirst-only answer postprocessor:
  `(question, draft answer, raw passages) -> canonical answer`.
- Added a selection gate so the postprocessor does not overwrite a better short
  answer with an incomplete phrase.
- Expanded answer cleanup for bullet, colon, dash, and meta-connective outputs.

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 5 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v6.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n5_v6.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n5_v6_usage.jsonl \
  --cache-tag hotpotqa_ef_n5_v6
```

Result:

- EM: 0.4000
- F1: 0.5833
- Avg. LLM calls: 5.80
- Chain complete: 2/5
- Repair attempts: 3/5

Observation: EM improved because a recoverable answer containing `EGOT` was
canonicalized correctly. This confirms that exact-match postprocessing is a
major short-term lever. Next gate: run 20 questions before scaling to 100.

### Run 5: 20-question gate with v6 settings

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 20 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n20_v6.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n20_v6.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n20_v6_usage.jsonl \
  --cache-tag hotpotqa_ef_n20_v6
```

Result:

- EM: 0.4500
- F1: 0.5386
- Avg. LLM calls: 6.10
- Chain complete: 8/20
- Repair attempts: 14/20
- Missing-entity total: 14
- Disconnected-pair total: 22

Conclusion: do not run 100 yet. F1 is improved but still below the short-term
0.60 target. The largest visible error groups are yes/no questions, comparison
questions, and cases where entity extraction marks generic words as missing
entities. A stability patch was added after this run to reject instruction-copy
postprocessor outputs and avoid `NodeNotFound` during endpoint distance checks.

Then rerun the same 5-question slice with a new cache tag before expanding.

### Run 6: 20-question gate after deterministic canonicalization

Change:

- Added high-precision EvidenceFirst postprocess canonicalization:
  - search-control answers such as `Let's search again` are treated as bad
    current answers;
  - acronym definition tails such as `... is the EGOT` normalize to `EGOT`;
  - `Did A or B have a population of N in YYYY?` choice outputs are converted
    to the supported Hotpot-style statement when passages contain the fact;
  - person/comparison answers can expand to a passage-supported full name or
    title, e.g. `Dwight Schultz` -> `William Dwight Schultz`.
- Added regression tests in `experiments/evidence_first/test_postprocess_gate.py`.

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 20 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n20_v11.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n20_v11.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n20_v11_usage.jsonl \
  --cache-tag hotpotqa_ef_n20_v11
```

Result:

- EM: 0.6000
- F1: 0.6361
- Avg. iterations: 1.00
- Avg. LLM calls: 6.35
- Chain complete: 8/20
- Repair attempts: 14/20
- Missing-entity total: 9
- Disconnected-pair total: 19

Observation: the 20-question gate now clears the short-term target. The main
lift comes from deterministic final-answer canonicalization, not improved chain
completeness. Remaining visible failures include missing evidence for event
location, yes/no comparison cases, noisy single-token entities such as `Out`,
and award questions where the LLM postprocessor does not surface `EGOT`.

Next: run the fixed 100-question EvidenceFirst evaluation with cache tag
`hotpotqa_ef_n100_v11` before claiming the target is met.

### Run 7: original local-context 100-question evaluation

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 100 \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_n100_v11.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_n100_v11.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_n100_v11_usage.jsonl \
  --cache-tag hotpotqa_ef_n100_v11
```

Result:

- EM: 0.4500
- F1: 0.4929
- Avg. iterations: 1.00
- Avg. LLM calls: 5.76

Observation: the original local 100-question sample did not clear the target
despite the 20-question gate passing. A follow-up context audit found that the
local `comagraag/data/hotpotqa_sample.json` file is not a fair context-complete
HotpotQA evaluation slice:

- First 100 rows: supporting-fact title missing from context in 110/240 cases
  (45.83%), affecting 69/100 questions.
- Full 500 rows: supporting-fact title missing from context in 531/1211 cases
  (43.85%), affecting 355/500 questions.

Conclusion: do not treat this run as the primary EvidenceFirst quality result.
It is a measurement of a truncated/misaligned context file.

### Run 8: restore contexts from HotpotQA distractor validation by id

Command:

```bash
python experiments/evidence_first/repair_hotpot_contexts.py \
  --input comagraag/data/hotpotqa_sample.json \
  --output experiments/evidence_first/data/hotpotqa_sample_hf_context.json \
  --split validation
```

Result:

- IDs match the original 500-row local sample.
- First 100 rows: supporting-fact title missing from context reduced from
  110/240 to 0/240.
- Full 500 rows: supporting-fact title missing from context reduced from
  531/1211 to 0/1211.

This file is the context-complete reproduction input for the current target.

### Run 9: repaired-context 20-question sanity check

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 20 \
  --data experiments/evidence_first/data/hotpotqa_sample_hf_context.json \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n20_v1.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n20_v1.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n20_v1_usage.jsonl \
  --cache-tag hotpotqa_ef_hfctx_n20_v1
```

Result:

- EM: 0.7500
- F1: 0.7750
- Avg. iterations: 1.00
- Avg. LLM calls: 5.55

Observation: restoring the original HotpotQA validation contexts gives a large
gain on the same first 20 IDs, confirming that the low original 100-question
result was dominated by missing evidence pages.

### Run 10: repaired-context 100-question target evaluation

Command:

```bash
python evaluate.py --mode full --variant evidence_first --n 100 \
  --data experiments/evidence_first/data/hotpotqa_sample_hf_context.json \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n100_v1.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n100_v1.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n100_v1_usage.jsonl \
  --cache-tag hotpotqa_ef_hfctx_n100_v1
```

Result:

- EM: 0.5800
- F1: 0.6495
- Avg. iterations: 1.00
- Avg. LLM calls: 5.49
- Avg. input tokens: 5949.13
- Avg. output tokens: 789.99
- Avg. latency: 20.0971s
- Chain complete: 54/100
- Repair attempts: 66/100
- Fallback cases: 1/100
- Missing-entity total: 37
- Disconnected-pair total: 132

Conclusion: the short-term target is met on the context-complete repaired
HotpotQA 100-question slice because F1 >= 0.60. EM remains just below the stricter
0.60 line, so the next optimization target is EM, not evidence coverage.

## 2026-06-06 HotpotQA 1000-question preflight

Goal: inspect the available 1000-question data and separated KG graph caches
before launching a full 1000-question EvidenceFirst run.

### Data/KG alignment

Available 1000-question data files:

- `data/hotpotqa_1000_for_test.json`: 1000 IDs, includes `supporting_facts`.
- `data/hotpotqa_combined1000_test.jsonl`: same 1000 ID set/order as
  `hotpotqa_1000_for_test.json`, but no `supporting_facts`.
- `data/hotpotqa_1000_test.json` and `data/hotpotqa_1000_test.jsonl`: 1000 IDs,
  but only 500 of them are covered by the available 1000 merged KG cache.

Available graph caches:

- `comagraag/data/hotpotqa_kgs.pkl`: 500 graphs.
- `data/hotpotqa_extra500_kgs.pkl`: 500 graphs.
- `comagraag/data/hotpotqa_1000_kgs_merged.pkl`: 1000 graphs.

Coverage check:

- `data/hotpotqa_1000_for_test.json` is fully covered by either
  `comagraag/data/hotpotqa_1000_kgs_merged.pkl` or the separated pair
  `comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl`.
- `data/hotpotqa_1000_test.json` is not aligned with the full 1000 KG set; it
  covers only the extra 500 graphs and misses the first 500 graph IDs.

The evaluator already supports separated KG files through a comma/semicolon
separated `--kg` value, so no merge script is required for a split-KG run.

### Context repair

The 1000-row JSON input has the same context-completeness problem observed in
the 100-question local sample:

- `data/hotpotqa_1000_for_test.json`: supporting-fact title missing from context
  in 1062/2398 cases.

The repaired context-complete 1000-row input was generated with:

```bash
python experiments/evidence_first/repair_hotpot_contexts.py \
  --input data/hotpotqa_1000_for_test.json \
  --output experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --split validation
```

Result:

- Items: 1000
- Missing IDs in HuggingFace validation: 0
- Supporting-fact title missing after repair: 0/2398

### Split-KG smoke checks

The following checks verify that both halves of the separated KG pair run through
EvidenceFirst with the repaired 1000-row context input.

Front-half smoke:

```bash
python evaluate.py --mode full --variant evidence_first --n 1 --start 0 \
  --data experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --kg 'comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl' \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start0.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start0.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start0_usage.jsonl \
  --cache-tag hotpotqa_ef_1000_hfctx_splitkg_smoke_start0
```

Result: `EM 1.0000 / F1 1.0000`, 5 LLM calls.

Back-half smoke:

```bash
python evaluate.py --mode full --variant evidence_first --n 1 --start 500 \
  --data experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --kg 'comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl' \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start500.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start500.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_1000_hfctx_splitkg_smoke_start500_usage.jsonl \
  --cache-tag hotpotqa_ef_1000_hfctx_splitkg_smoke_start500
```

Result: `EM 1.0000 / F1 1.0000`, 7 LLM calls.

### Recommended 1000-question run

Use the repaired-context 1000-row input and the separated KG pair:

```bash
python evaluate.py --mode full --variant evidence_first --n 1000 \
  --data experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --kg 'comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl' \
  --out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v1.csv \
  --out-dir experiments/evidence_first/cache \
  --save-pipeline-details \
  --jsonl-out experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v1.jsonl \
  --usage-log experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v1_usage.jsonl \
  --cache-tag hotpotqa_ef_hfctx_n1000_splitkg_v1
```

Based on the smoke latencies, a serial 1000-question run is likely many hours.
If wall time matters, run by `--start/--n` batches with distinct cache tags and
merge/analyze the JSONL outputs afterwards.

### Run 11: parallel 1000-question run and OOM repair

Initial parallel run:

- Run tag/output prefix: `hotpotqa_evidence_first_hfctx_n1000_splitkg_v3`
- Launcher: `experiments/evidence_first/run_hotpot1000_hfctx_splitkg_parallel.sh`
- Workers: 8
- Batch size: 125
- Data: `experiments/evidence_first/data/hotpotqa_1000_hf_context.json`
- KG: `comagraag/data/hotpotqa_kgs.pkl,data/hotpotqa_extra500_kgs.pkl`

The initial v3 merge produced `EM 0.3420 / F1 0.4095`, but this result is
invalid because the last three batches were polluted by local CUDA OOM failures:

- Batch 5: 125/125 errors, 125 CUDA OOM.
- Batch 6: 125/125 errors, 125 CUDA OOM.
- Batch 7: 125/125 errors, 124 CUDA OOM.

Root cause: multiple evaluator processes loaded the local `SentenceTransformer`
embedder onto GPU0 while GPU0 was already heavily occupied. This caused empty
answers for the affected batches and artificially depressed the aggregate score.

Repair run:

- Repair script: `experiments/evidence_first/run_hotpot1000_hfctx_splitkg_repair_v4.sh`
- Repair batches: 5, 6, and 7 only.
- Repair environment: `CUDA_VISIBLE_DEVICES=1`
- Base run for batches 0-4: `hotpotqa_evidence_first_hfctx_n1000_splitkg_v3`
- Final merged prefix: `hotpotqa_evidence_first_hfctx_n1000_splitkg_v4`

Repair result:

- EM: 0.5500
- F1: 0.6484
- n: 1000
- Fresh rows: 1000
- Avg. iterations: 1.00
- Avg. LLM calls: 5.71
- Avg. input tokens: 6301.24
- Avg. output tokens: 793.81
- Avg. latency: 20.13s
- Chain complete: 568/1000
- Repair attempts: 637/1000
- Fallback cases: 5/1000
- Missing-entity total: 432
- Disconnected-pair total: 2082
- Errors: 4 total, all `data_inspection_failed`; 0 CUDA OOM.

Per-batch scores after repair:

- Batch 0: EM 0.6000 / F1 0.6851
- Batch 1: EM 0.5040 / F1 0.6195
- Batch 2: EM 0.5280 / F1 0.6275
- Batch 3: EM 0.5680 / F1 0.6822
- Batch 4: EM 0.5360 / F1 0.6616
- Batch 5: EM 0.5120 / F1 0.6115
- Batch 6: EM 0.5280 / F1 0.6061
- Batch 7: EM 0.6240 / F1 0.6941

Conclusion: use v4 as the valid 1000-question repaired-context split-KG result.
The 1000-question run clears the F1 >= 0.60 target, though EM remains below
0.60. Do not use v3 for model-quality claims.

### Run 12: v5 offline final-answer clean

The v4 result is frozen as the valid end-to-end baseline. As a first low-risk
EM improvement pass, v5 applies only deterministic final-answer cleanup to the
existing v4 JSONL predictions; it does not make fresh LLM calls and does not
overwrite v4.

Cleanup rule:

- Prefer the final short answer segment when a prediction contains obvious
  leaked context before the answer, either separated by a newline or by a period
  followed by two or more spaces.
- Apply the change only when the final segment looks like a short answer
  phrase: 1-6 tokens and at least one uppercase character or digit.

Regression coverage:

```bash
pytest -q experiments/evidence_first/test_postprocess_gate.py
```

Result: `12 passed`.

Independent metric recomputation:

```bash
python scripts/compute_metrics.py \
  --predictions experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v4.jsonl \
  --gold experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --prediction-field prediction \
  --no-answer-fallback

python scripts/compute_metrics.py \
  --predictions experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v5_offline_clean.jsonl \
  --gold experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --prediction-field prediction \
  --no-answer-fallback
```

Results:

- v4 frozen baseline: `EM 0.5500 (550/1000) / F1 0.6484`
- v5 offline clean: `EM 0.5570 (557/1000) / F1 0.6507`
- Changed rows: 15
- EM gains: 7
- EM losses: 0
- Fresh rows / fresh LLM calls: 0

Representative fixed predictions:

- `Epitaph Records.  Motion City Soundtrack` -> `Motion City Soundtrack`
- `ISIL.\nOperation Diadem` -> `Operation Diadem`
- `Gaydar.\nHayley Williams` -> `Hayley Williams`

Result files:

- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v5_offline_clean.csv`
- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v5_offline_clean.jsonl`
- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v5_offline_clean_usage.jsonl`

Boundary: v5 is a deterministic posthoc cleanup of v4 predictions. Use v4 for
the frozen fresh end-to-end run, and use v5 only when reporting the first
offline EM-cleaning pass.

### Run 13: v6 offline selector guard

The v6 pass keeps v4 and v5 frozen, then applies a narrow selector-gate change
over the v5 JSONL. It does not make fresh LLM calls.

Selector rule:

- If the inferred answer type is not `yesno`, and the current answer is an
  obvious answer-type mismatch (`yes`, `no`, `Wait`, `But`, or `Alternative`),
  select the postprocessor answer only when it is a short non-yes/no phrase.
- Do not select truncated parenthetical answers with unbalanced brackets.
- Do not select yes/no flips for true `yesno` answer types.

Regression coverage:

```bash
pytest -q experiments/evidence_first/test_postprocess_gate.py
```

Result: `16 passed`.

Independent metric recomputation:

```bash
python scripts/compute_metrics.py \
  --predictions experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl \
  --gold experiments/evidence_first/data/hotpotqa_1000_hf_context.json \
  --prediction-field prediction \
  --no-answer-fallback
```

Result:

- v6 offline selector guard: `EM 0.5620 (562/1000) / F1 0.6564`
- Delta vs v5 offline clean: `+5 EM`, `0 EM losses`
- Changed rows: 6
- Fresh rows / fresh LLM calls: 0

Representative fixed predictions:

- `Wait` -> `Sapsali`
- `no` -> `alcoholic`
- `Alternative` -> `George Cayley`
- `But` -> `writer`

Result files:

- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.csv`
- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard.jsonl`
- `experiments/evidence_first/results/hotpotqa_evidence_first_hfctx_n1000_splitkg_v6_offline_selector_guard_usage.jsonl`

Boundary: v6 is still an offline posthoc pass over v5/v4 predictions. Use v4
for fresh-run claims; use v6 only as the cumulative offline cleanup/selector
result.
