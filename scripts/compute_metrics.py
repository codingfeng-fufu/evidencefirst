import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    text = str(text).lower()
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


def _load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, dict):
            return [
                {"_id": qid, **value}
                for qid, value in data.items()
                if isinstance(value, dict)
            ]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON shape: {path}")
    if suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported file extension: {path}")


def _qid(row: dict, fallback: int) -> str:
    return str(row.get("_id") or row.get("id") or row.get("qid") or fallback)


def _text(value, null_strings_as_empty: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    if null_strings_as_empty and text.strip().lower() in {"null", "none", "nan"}:
        return ""
    return text


def _prediction(
    row: dict,
    prediction_fields: tuple[str, ...] | None = None,
    include_answer_fallback: bool = True,
    null_strings_as_empty: bool = False,
) -> str:
    if prediction_fields:
        for field in prediction_fields:
            if field in row:
                return _text(row.get(field), null_strings_as_empty)
        return ""

    fields = ["prediction", "pred_answer", "predicted_answer", "response"]
    if include_answer_fallback:
        fields.append("answer")
    fields.append("pred")

    for field in fields:
        value = row.get(field)
        if value:
            return _text(value, null_strings_as_empty)
    return ""


def _gold(row: dict) -> str:
    return str(row.get("gold") or row.get("gold_answer") or row.get("answer") or "")


def evaluate(
    predictions_path: Path,
    gold_path: Path,
    prediction_fields: tuple[str, ...] | None = None,
    include_answer_fallback: bool = True,
    null_strings_as_empty: bool = False,
) -> list[dict]:
    preds = {_qid(row, i): row for i, row in enumerate(_load_records(predictions_path))}
    gold_rows = _load_records(gold_path)

    rows = []
    for i, gold_row in enumerate(gold_rows):
        qid = _qid(gold_row, i)
        pred_row = preds.get(qid, {})
        pred = _prediction(
            pred_row,
            prediction_fields=prediction_fields,
            include_answer_fallback=include_answer_fallback,
            null_strings_as_empty=null_strings_as_empty,
        )
        gold = _gold(gold_row)
        rows.append({
            "qid": qid,
            "question": gold_row.get("question", pred_row.get("question", "")),
            "prediction": pred,
            "gold": gold,
            "em": em(pred, gold),
            "f1": f"{f1(pred, gold):.4f}",
            "error": pred_row.get("error", ""),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--prediction-field",
        action="append",
        default=None,
        help="Use this prediction field exactly; repeat to provide ordered alternatives.",
    )
    parser.add_argument(
        "--no-answer-fallback",
        action="store_true",
        help="Do not fall back to an answer field when no prediction field is populated.",
    )
    parser.add_argument(
        "--null-strings-as-empty",
        action="store_true",
        help='Treat string values like "null", "none", and "nan" as empty predictions.',
    )
    args = parser.parse_args()

    rows = evaluate(
        args.predictions,
        args.gold,
        prediction_fields=tuple(args.prediction_field) if args.prediction_field else None,
        include_answer_fallback=not args.no_answer_fallback,
        null_strings_as_empty=args.null_strings_as_empty,
    )
    em_avg = sum(float(r["em"]) for r in rows) / len(rows) if rows else 0.0
    f1_avg = sum(float(r["f1"]) for r in rows) / len(rows) if rows else 0.0
    print(f"EM: {em_avg:.4f} ({sum(int(r['em']) for r in rows)}/{len(rows)})")
    print(f"F1: {f1_avg:.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "question", "prediction", "gold", "em", "f1", "error"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
