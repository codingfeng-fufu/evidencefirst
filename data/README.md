# Data Preparation

The repository does not track generated dataset samples or KG caches.
Use the preparation scripts in this directory to recreate the evaluation files:

- [prepare_hotpot.py](prepare_hotpot.py) - sample HotpotQA and derive the validation subset
- [prepare_2wiki.py](prepare_2wiki.py) - sample a fixed 500-question 2Wiki set with `seed=42`

Expected generated files:

- `data/hotpotqa_sample.json`
- `data/hotpotqa_val.json`
- `data/2wiki_sample_fixed.json`
- `data/2wiki_fixed_ids.json`
- `data/hotpotqa_kgs.pkl`
- `data/2wiki_kgs_fixed.pkl`
