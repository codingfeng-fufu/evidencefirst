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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki"], default="hotpotqa")
    parser.add_argument("--data", type=str, default=None, help="Input dataset JSON")
    parser.add_argument("--out", type=str, default=None, help="Output KG pickle path")
    parser.add_argument("--skip-sample", action="store_true", help="Skip HotpotQA sampling step if input file is missing")
    args = parser.parse_args()

    hotpot_impl = _load_impl("comagraag_build_kg_impl", PKG / "build_kg.py")
    wiki2_impl = _load_impl("comagraag_build_2wiki_kg_impl", PKG / "data" / "build_2wiki_kg.py")

    if args.dataset == "hotpotqa":
        data_path = args.data or "data/hotpotqa_sample.json"
        out_path = args.out or "data/hotpotqa_kgs.pkl"
        if not args.skip_sample and not Path(data_path).exists():
            hotpot_impl.sample_hotpotqa()
        hotpot_impl.build_all_kgs(data_path=data_path, kg_path=out_path)
    else:
        data_path = args.data or "data/2wiki_sample_fixed.json"
        if not Path(data_path).exists():
            fallback = ROOT / "data" / "2wiki_sample.json"
            data_path = str(fallback if fallback.exists() else data_path)
        out_path = args.out or "data/2wiki_kgs_fixed.pkl"
        wiki2_impl.build_all_2wiki_kgs(data_path=data_path, kg_path=out_path)


if __name__ == "__main__":
    main()
