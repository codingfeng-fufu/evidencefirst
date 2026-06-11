import json
import random
from pathlib import Path


SEED = 42
N = 500


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "2wiki_sample_fixed.json"
    ids_path = root / "data" / "2wiki_fixed_ids.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading 2WikiMultiHopQA validation split...")
    from datasets import load_dataset

    ds = load_dataset("din0s/2wikimultihopqa", split="validation", trust_remote_code=True)
    data = list(ds)

    random.seed(SEED)
    sampled = random.sample(data, N)
    ids = [item.get("_id", item.get("id", str(i))) for i, item in enumerate(sampled)]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print(f"Wrote {N} fixed 2Wiki questions to {out_path}")
    print(f"Wrote ID list to {ids_path}")


if __name__ == "__main__":
    main()

