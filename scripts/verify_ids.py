import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

GROUPS = {
    "hotpotqa": [
        "hotpotqa_full.csv",
        "hotpotqa_no_verif.csv",
        "hotpotqa_no_decomp.csv",
        "hotpotqa_naive_rag.csv",
    ],
    "2wiki": [
        "2wiki_full.csv",
        "2wiki_no_verif.csv",
        "2wiki_no_decomp.csv",
        "2wiki_naive_rag.csv",
    ],
}


def load_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [row["qid"] for row in csv.DictReader(f)]


def main() -> None:
    for group, files in GROUPS.items():
        print(f"=== {group} ===")
        existing = [RESULTS_DIR / name for name in files if (RESULTS_DIR / name).exists()]
        if len(existing) != len(files):
            missing = [name for name in files if not (RESULTS_DIR / name).exists()]
            print(f"missing files: {missing}")
            continue

        id_lists = {path.name: load_ids(path) for path in existing}
        ref_name = files[0]
        ref_ids = id_lists[ref_name]
        print(f"{ref_name}: {len(ref_ids)} rows")
        for name in files[1:]:
            same = ref_ids == id_lists[name]
            print(f"{ref_name} == {name}: {same}")


if __name__ == "__main__":
    main()

