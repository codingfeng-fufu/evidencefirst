import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MAIN_RESULT_PREFIXES = ("hotpotqa_", "2wiki_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include local archived/experimental CSVs in results/.",
    )
    args = parser.parse_args()

    rows = []
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        if not args.all and not csv_path.name.startswith(MAIN_RESULT_PREFIXES):
            continue
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        if not data or "em" not in data[0] or "f1" not in data[0]:
            continue
        em_avg = sum(float(row["em"]) for row in data) / len(data)
        f1_avg = sum(float(row["f1"]) for row in data) / len(data)
        rows.append((csv_path.name, len(data), em_avg, f1_avg))

    print("file,n,EM,F1")
    for name, n, em_avg, f1_avg in rows:
        print(f"{name},{n},{em_avg:.4f},{f1_avg:.4f}")


if __name__ == "__main__":
    main()
