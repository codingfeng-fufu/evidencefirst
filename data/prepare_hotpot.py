import json
import random
from pathlib import Path
from datasets import load_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data" / "hotpotqa_sample.json"
    val_path = root / "data" / "hotpotqa_val.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_path.exists():
        data = json.load(open(sample_path, encoding="utf-8"))
    else:
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)
        bridge = [x for x in ds if x["type"] == "bridge"]
        comp = [x for x in ds if x["type"] == "comparison"]

        random.seed(42)
        data = random.sample(bridge, 250) + random.sample(comp, 250)
        random.shuffle(data)

        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    bridge = [x for x in data if x.get("type") == "bridge"]
    comp = [x for x in data if x.get("type") == "comparison"]

    random.seed(2026)
    val = random.sample(bridge, 50) + random.sample(comp, 50)
    random.shuffle(val)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} HotpotQA samples to {sample_path}")
    print(f"Wrote {len(val)} validation samples to {val_path}")


if __name__ == "__main__":
    main()
