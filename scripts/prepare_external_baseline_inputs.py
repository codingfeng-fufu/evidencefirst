import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_items(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset shape: {path}")


def _qid(item: dict, fallback: int) -> str:
    return str(item.get("_id") or item.get("id") or item.get("qid") or fallback)


def _context_pairs(item: dict) -> list[tuple[str, list[str]]]:
    context = item.get("context", [])
    qid = _qid(item, 0)
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [(str(title), list(sents)) for title, sents in zip(titles, sentences)]

    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        pairs = []
        for idx, passage in enumerate(context):
            passage = str(passage)
            if ":" in passage:
                title, text = passage.split(":", 1)
                title = title.strip() or f"{qid}_passage_{idx}"
                text = text.strip()
            else:
                title = f"{qid}_passage_{idx}"
                text = passage
            pairs.append((title, [text] if text else []))
        return pairs

    pairs = []
    for idx, pair in enumerate(context):
        if len(pair) != 2:
            continue
        title, sents = pair
        if isinstance(sents, str):
            sents = [sents]
        pairs.append((str(title), list(sents)))
    return pairs


def _passage_text(title: str, sentences: list[str]) -> str:
    body = " ".join(str(s).strip() for s in sentences if str(s).strip())
    return f"{title}: {body}" if body else title


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(text: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return (name or fallback)[:140]


def _prepare_arag(items: list[dict], out_dir: Path, args: argparse.Namespace) -> list[str]:
    chunks = []
    questions = []
    for i, item in enumerate(items):
        qid = _qid(item, i)
        questions.append({
            "qid": qid,
            "id": qid,
            "question": item["question"],
            "answer": item.get("answer", ""),
            "gold_answer": item.get("answer", ""),
            "type": item.get("type", ""),
        })
        for title, sentences in _context_pairs(item):
            text = _passage_text(title, sentences)
            if not text.strip():
                continue
            chunks.append({
                "id": str(len(chunks)),
                "text": text,
                "qid": qid,
                "title": title,
            })

    chunks_file = out_dir / "arag" / "chunks.json"
    questions_file = out_dir / "arag" / "questions.json"
    index_dir = out_dir / "arag" / "index"
    results_dir = out_dir / "arag" / "results"
    config_file = out_dir / "arag" / "config.yaml"

    _write_json(chunks_file, chunks)
    _write_json(questions_file, questions)
    index_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config_text = f"""llm:
  temperature: 0.0
  max_tokens: {args.arag_max_tokens}

embedding:
  model: "{args.embedding_model}"
  device: "{args.embedding_device}"
  batch_size: {args.embedding_batch_size}

agent:
  max_loops: {args.arag_max_loops}
  max_token_budget: {args.arag_token_budget}
  verbose: false

data:
  chunks_file: "{chunks_file.resolve()}"
  questions_file: "{questions_file.resolve()}"
  index_dir: "{index_dir.resolve()}"
"""
    config_file.write_text(config_text, encoding="utf-8")

    return [
        f"cd {(ROOT / 'external_repos' / 'arag').resolve()}",
        "uv sync --extra full",
        (
            "uv run python scripts/build_index.py "
            f"--chunks {chunks_file.resolve()} "
            f"--output {index_dir.resolve()} "
            f"--model {args.embedding_model} "
            f"--device {args.embedding_device} "
            f"--batch-size {args.embedding_batch_size}"
        ),
        (
            'ARAG_API_KEY="${ARAG_API_KEY:-$OPENAI_API_KEY}" '
            'ARAG_BASE_URL="${ARAG_BASE_URL:-${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}" '
            'ARAG_MODEL="${ARAG_MODEL:-${LLM_MODEL:-qwen-plus}}" '
            "uv run python scripts/batch_runner.py "
            f"--config {config_file.resolve()} "
            f"--questions {questions_file.resolve()} "
            f"--output {results_dir.resolve()} "
            f"--workers {args.workers}"
        ),
        (
            f"python {(ROOT / 'scripts' / 'compute_metrics.py').resolve()} "
            f"--predictions {results_dir.resolve() / 'predictions.jsonl'} "
            f"--gold {args.data.resolve()} "
            f"--out-csv {results_dir.resolve() / 'metrics.csv'}"
        ),
    ]


def _prepare_hoprag(items: list[dict], out_dir: Path) -> list[str]:
    rows = []
    docs_seen: dict[str, str] = {}
    docs_dir = out_dir / "hoprag" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_file = out_dir / "hoprag" / "hotpotqa.jsonl"

    for i, item in enumerate(items):
        qid = _qid(item, i)
        pairs = _context_pairs(item)
        rows.append({
            "_id": qid,
            "id": qid,
            "question": item["question"],
            "answer": item.get("answer", ""),
            "type": item.get("type", ""),
            "context": [[title, sentences] for title, sentences in pairs],
        })
        for idx, (title, sentences) in enumerate(pairs):
            body = "\n\n".join(str(s).strip() for s in sentences if str(s).strip())
            if not body or body in docs_seen:
                continue
            stem = _safe_name(title, f"{qid}_{idx}")
            docs_seen[body] = stem
            (docs_dir / f"{stem}.txt").write_text(body, encoding="utf-8")

    _write_jsonl(data_file, rows)
    cache_dir = out_dir / "hoprag" / "cache_hotpot_online"
    output_dir = out_dir / "hoprag" / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        f"cd {(ROOT / 'external_repos' / 'hoprag').resolve()}",
        "# Edit config.py first: Neo4j credentials, API base/key/model, embedding model path, and node/index names.",
        (
            "python -c \"from HopBuilder import main_nodes; "
            f"main_nodes(cache_dir='{cache_dir.resolve()}', docs_dir='{docs_dir.resolve()}', "
            "label='hotpot_comagraag', start_index=0, span=12000, offline=False)\""
        ),
        (
            "python -c \"from HopBuilder import main_edges_index; "
            f"main_edges_index(cache_dir='{cache_dir.resolve()}', problems_path='{data_file.resolve()}', "
            "label='hotpot_comagraag')\""
        ),
        (
            "python HopGenerator.py "
            "--model_name \"$LLM_MODEL\" "
            f"--data_path {data_file.resolve()} "
            f"--save_dir {output_dir.resolve()} "
            "--retriever_name HopRetriever --max_hop 4 --topk 20 --traversal bfs_node "
            "--mode common --label hotpot_comagraag"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("comagraag/data/hotpotqa_sample.json"))
    parser.add_argument("--out-root", type=Path, default=Path("external_runs"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--baseline", nargs="+", default=["all"], choices=["all", "arag", "hoprag"])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--arag-max-loops", type=int, default=15)
    parser.add_argument("--arag-max-tokens", type=int, default=4096)
    parser.add_argument("--arag-token-budget", type=int, default=128000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    items = _load_items(args.data)
    run_name = args.name or f"{args.data.stem}_{len(items)}"
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {"arag", "hoprag"} if "all" in args.baseline else set(args.baseline)
    commands: list[str] = []
    if "arag" in selected:
        commands.extend(["# A-RAG", *_prepare_arag(items, out_dir, args), ""])
    if "hoprag" in selected:
        commands.extend(["# HopRAG", *_prepare_hoprag(items, out_dir), ""])

    commands_file = out_dir / "commands.txt"
    commands_file.write_text("\n".join(commands), encoding="utf-8")

    print(f"Wrote external-baseline inputs under {out_dir}")
    print(f"Wrote commands to {commands_file}")
    print("\n".join(commands))


if __name__ == "__main__":
    main()
