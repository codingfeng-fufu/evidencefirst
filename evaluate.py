import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "comagraag"
sys.path.insert(0, str(PKG))

_ABLATIONS = (
    "none",
    "without_verification",
    "without_repair",
    "without_reader_context",
    "without_answer_refinement",
)


def _load_impl(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prefer_root_data(name: str) -> str:
    primary = ROOT / "data" / name
    legacy = PKG / "data" / name
    return str(primary if primary.exists() or not legacy.exists() else legacy)


def _default_data(dataset: str) -> str:
    if dataset == "2wiki":
        fixed = _prefer_root_data("2wiki_sample_fixed.json")
        return fixed if Path(fixed).exists() else _prefer_root_data("2wiki_sample.json")
    return _prefer_root_data("hotpotqa_sample.json")


def _default_kg(dataset: str) -> str:
    if dataset == "2wiki":
        fixed = _prefer_root_data("2wiki_kgs_fixed.pkl")
        return fixed if Path(fixed).exists() else _prefer_root_data("2wiki_kgs.pkl")
    return _prefer_root_data("hotpotqa_kgs.pkl")


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
    parser.add_argument("--max-iter", type=int, default=None, help="Override config.MAX_ITER for this run")
    parser.add_argument("--variant", type=str, default="default", choices=["default", "sparql_cot", "evidence_aug", "evidence_first"], help="Experiment variant")
    parser.add_argument("--ablation", type=str, default="none", choices=_ABLATIONS, help="EvidenceFirst ablation switch")
    parser.add_argument("--jsonl-out", type=str, default=None, help="Per-example prediction JSONL output")
    parser.add_argument("--usage-log", type=str, default=None, help="Per-example usage JSONL output")
    parser.add_argument("--cache-tag", type=str, default=None, help="Cache namespace for this experiment")
    parser.add_argument("--no-global-context", action="store_true", help="For pipeline modes, use only each example's local context")
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
        max_iter=args.max_iter,
        variant=args.variant,
        ablation=args.ablation,
        jsonl_out=args.jsonl_out,
        usage_log=args.usage_log,
        cache_tag=args.cache_tag,
        use_global_context=not args.no_global_context,
    )


if __name__ == "__main__":
    main()
