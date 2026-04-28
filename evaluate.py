import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "comagraag"
sys.path.insert(0, str(PKG))


def _load_impl(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _default_data(dataset: str) -> str:
    if dataset == "2wiki":
        fixed = ROOT / "data" / "2wiki_sample_fixed.json"
        return str(fixed if fixed.exists() else ROOT / "data" / "2wiki_sample.json")
    return str(ROOT / "data" / "hotpotqa_sample.json")


def _default_kg(dataset: str) -> str:
    if dataset == "2wiki":
        fixed = ROOT / "data" / "2wiki_kgs_fixed.pkl"
        return str(fixed if fixed.exists() else ROOT / "data" / "2wiki_kgs.pkl")
    return str(ROOT / "data" / "hotpotqa_kgs.pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Evaluate the first N questions")
    parser.add_argument("--start", type=int, default=0, help="0-based start index")
    parser.add_argument("--mode", nargs="+", default=["full", "no_verif", "no_decomp"], help="Evaluation modes")
    parser.add_argument("--quick", action="store_true", help="Use the repository quick sample size")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "2wiki"])
    parser.add_argument("--data", type=str, default=None, help="Dataset path")
    parser.add_argument("--kg", type=str, default=None, help="KG pickle path")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--out-dir", type=str, default="results", help="Cache/output directory")
    parser.add_argument("--rerun-bad-cache", action="store_true", help="Rerun empty or bad cached answers")
    parser.add_argument("--save-pipeline-details", action="store_true", help="Save score/converged/history for pipeline modes")
    args = parser.parse_args()

    impl = _load_impl("comagraag_evaluate_impl", PKG / "evaluate.py")
    data_path = args.data or _default_data(args.dataset)
    kg_path = args.kg or _default_kg(args.dataset)
    n = 50 if args.quick else args.n

    impl.run_eval(
        data_path=data_path,
        kg_path=kg_path,
        modes=tuple(args.mode),
        n=n,
        start=args.start,
        out_dir=args.out_dir,
        dataset=args.dataset,
        out_csv=args.out,
        rerun_bad_cache=args.rerun_bad_cache,
        save_pipeline_details=args.save_pipeline_details,
    )


if __name__ == "__main__":
    main()
