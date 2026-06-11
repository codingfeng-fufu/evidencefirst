import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "comagraag"
OUT_DIR = ROOT / "results"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if not n:
        return 0.0
    prec = n / len(p_toks)
    rec = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def is_bad_answer(ans: str) -> bool:
    lowered = ans.lower().strip()
    bad_keywords = ("cannot", "no information", "no relevant", "not enough", "cannot determine")
    return (not lowered) or any(keyword in lowered for keyword in bad_keywords)


def export_one(data_path: Path, cache_path: Path, out_path: Path, fallback_cache_path: Path | None = None) -> None:
    data = json.load(open(data_path, encoding="utf-8"))
    cache = json.load(open(cache_path, encoding="utf-8"))
    fallback_cache = (
        json.load(open(fallback_cache_path, encoding="utf-8"))
        if fallback_cache_path and fallback_cache_path.exists()
        else {}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "question", "prediction", "gold", "em", "f1"])
        for i, item in enumerate(data):
            qid = item.get("id", item.get("_id", str(i)))
            pred = cache.get(qid, {}).get("answer", "")
            if fallback_cache and is_bad_answer(str(pred)):
                fallback = fallback_cache.get(qid, {}).get("answer", "")
                if fallback:
                    pred = fallback
            gold = item["answer"]
            writer.writerow([qid, item["question"], pred, gold, em(pred, gold), f"{f1(pred, gold):.4f}"])
    print(f"Wrote {out_path}")


def main() -> None:
    hotpot_data = SRC / "data" / "hotpotqa_sample.json"
    wiki_data = SRC / "data" / "2wiki_sample.json"

    exports = [
        (hotpot_data, SRC / "results" / "cache_full.json", OUT_DIR / "hotpotqa_full.csv", SRC / "results" / "cache_naive_rag.json"),
        (hotpot_data, SRC / "results" / "cache_no_verif.json", OUT_DIR / "hotpotqa_no_verif.csv", SRC / "results" / "cache_naive_rag.json"),
        (hotpot_data, SRC / "results" / "cache_no_decomp.json", OUT_DIR / "hotpotqa_no_decomp.csv", SRC / "results" / "cache_naive_rag.json"),
        (hotpot_data, SRC / "results" / "cache_naive_rag.json", OUT_DIR / "hotpotqa_naive_rag.csv", None),
        (wiki_data, SRC / "results_2wiki_rerun_full" / "cache_2wiki_full.json", OUT_DIR / "2wiki_full.csv", None),
        (wiki_data, SRC / "results_2wiki_rerun" / "cache_2wiki_no_verif.json", OUT_DIR / "2wiki_no_verif.csv", SRC / "results_2wiki_rerun" / "cache_2wiki_naive_rag.json"),
        (wiki_data, SRC / "results_2wiki_rerun_no_decomp" / "cache_2wiki_no_decomp.json", OUT_DIR / "2wiki_no_decomp.csv", None),
        (wiki_data, SRC / "results_2wiki_rerun" / "cache_2wiki_naive_rag.json", OUT_DIR / "2wiki_naive_rag.csv", None),
    ]

    for data_path, cache_path, out_path, fallback_cache_path in exports:
        if not data_path.exists():
            print(f"skip missing data: {data_path}")
            continue
        if not cache_path.exists():
            print(f"skip missing cache: {cache_path}")
            continue
        export_one(data_path, cache_path, out_path, fallback_cache_path)


if __name__ == "__main__":
    main()
