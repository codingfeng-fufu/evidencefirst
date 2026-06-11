#!/usr/bin/env python3
"""Run an official-style HopRAG HotpotQA slice."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
HOPRAG_DIR = ROOT / "external_repos" / "hoprag"


def _load_items(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset shape: {path}")


def _qid(item: dict, fallback: int) -> str:
    return str(item.get("_id") or item.get("id") or item.get("qid") or fallback)


def _context_pairs(item: dict, fallback_qid: str) -> list[tuple[str, list[str]]]:
    context = item.get("context", [])
    if isinstance(context, dict):
        return [
            (str(title), [str(s) for s in sents])
            for title, sents in zip(context.get("title", []), context.get("sentences", []))
        ]
    if isinstance(context, list) and (not context or isinstance(context[0], str)):
        pairs = []
        for idx, passage in enumerate(context):
            passage = str(passage)
            if ":" in passage:
                title, text = passage.split(":", 1)
                title = title.strip() or f"{fallback_qid}_passage_{idx}"
                text = text.strip()
            else:
                title = f"{fallback_qid}_passage_{idx}"
                text = passage.strip()
            pairs.append((title, [text] if text else []))
        return pairs
    pairs = []
    for idx, pair in enumerate(context):
        if len(pair) != 2:
            continue
        title, sentences = pair
        if isinstance(sentences, str):
            sentences = [sentences]
        pairs.append((str(title), [str(s) for s in sentences]))
    return pairs


def _safe_identifier(value: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe {kind}: {value!r}")
    return value


def _prepare_data(items: list[dict], run_dir: Path) -> tuple[Path, Path, Path]:
    data_file = run_dir / "hotpotqa.jsonl"
    gold_file = run_dir / "gold.jsonl"
    docs_dir = run_dir / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_bodies = set()
    for i, item in enumerate(items):
        qid = _qid(item, i)
        pairs = _context_pairs(item, qid)
        rows.append(
            {
                "_id": qid,
                "id": qid,
                "question": item["question"],
                "answer": item.get("answer", ""),
                "type": item.get("type", ""),
                "context": [[title, sentences] for title, sentences in pairs],
            }
        )
        for title, sentences in pairs:
            body = "\n\n".join(s.strip() for s in sentences if s.strip())
            if not body or body in seen_bodies:
                continue
            seen_bodies.add(body)
            doc_name = title.replace("/", "_") + ".txt"
            (docs_dir / doc_name).write_text(body, encoding="utf-8")

    with data_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gold_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return data_file, docs_dir, gold_file


def _reset_graph(config) -> None:
    from neo4j import GraphDatabase

    label = _safe_identifier(config.node_name, "node label")
    rel_type = _safe_identifier(config.edge_name, "edge type")
    index_names = [
        config.node_dense_index_name,
        config.edge_dense_index_name,
        config.node_sparse_index_name,
        config.edge_sparse_index_name,
    ]
    with GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password),
        database=config.neo4j_dbname,
    ) as driver:
        driver.verify_connectivity()
        with driver.session(database=config.neo4j_dbname) as session:
            for name in index_names:
                session.run(f"DROP INDEX {_safe_identifier(name, 'index name')} IF EXISTS")
            session.run(f"MATCH (n:{label}) DETACH DELETE n")
            session.run(f"MATCH ()-[r:{rel_type}]->() DELETE r")


def _export_predictions(data_file: Path, output_dir: Path, predictions_file: Path) -> Path:
    result_files = sorted(output_dir.glob("*/hotpot_pred_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No HopRAG prediction JSON found under {output_dir}")
    result = json.loads(result_files[-1].read_text(encoding="utf-8"))
    answers = result.get("answer", {})
    questions = [_ for _ in _load_items(data_file)]
    with predictions_file.open("w", encoding="utf-8") as file:
        for i, item in enumerate(questions):
            qid = _qid(item, i)
            row = {
                "_id": qid,
                "id": qid,
                "question": item.get("question", ""),
                "response": answers.get(qid, ""),
                "answer": item.get("answer", ""),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result_files[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "hotpotqa_1000_test.jsonl")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--traversal", default="bfs_node")
    parser.add_argument("--device", default=os.environ.get("HOPRAG_DEVICE", "cuda:0"))
    args = parser.parse_args()
    args.data = args.data if args.data.is_absolute() else ROOT / args.data

    run_name = args.run_name or f"hoprag_s{args.start}_n{args.limit}_bge"
    run_dir = ROOT / "external_runs" / run_name
    cache_dir = run_dir / "cache_hotpot_online"
    output_dir = run_dir / "results"
    predictions_file = run_dir / "predictions.jsonl"
    for path in (cache_dir, output_dir):
        if path.exists():
            shutil.rmtree(path)
    run_dir.mkdir(parents=True, exist_ok=True)

    node_name = f"hotpot_bgeen_qwen_plus_s{args.start}_n{args.limit}"
    os.environ["HOPRAG_NODE_NAME"] = node_name
    os.environ["HOPRAG_EDGE_NAME"] = "pen2ans_" + node_name
    os.environ["HOPRAG_DEVICE"] = args.device

    sys.path.insert(0, str(HOPRAG_DIR))
    os.chdir(HOPRAG_DIR)

    import config
    from HopBuilder import main_edges_index, main_nodes
    from HopGenerator import main_hotpot

    items = _load_items(args.data)[args.start : args.start + args.limit]
    data_file, docs_dir, gold_file = _prepare_data(items, run_dir)

    _reset_graph(config)
    main_nodes(
        cache_dir=str(cache_dir),
        docs_dir=str(docs_dir),
        label=config.node_name,
        start_index=0,
        span=12000,
        offline=False,
    )
    main_edges_index(
        cache_dir=str(cache_dir),
        problems_path=str(data_file),
        label=config.node_name,
    )

    hop_args = SimpleNamespace(
        model_name=config.default_gpt_model,
        traversal_model=config.traversal_model,
        embedding_model=config.embed_model,
        rerank_model=None,
        data_path=str(data_file),
        save_dir=str(output_dir),
        retriever_name="HopRetriever",
        max_hop=args.max_hop,
        start_layer=0,
        max_layer=2,
        entry_type="node",
        trim=False,
        hybrid=False,
        tol=20,
        mock_dense=False,
        mock_sparse=False,
        mode="common",
        label=config.node_name,
        topk=args.topk,
        traversal=args.traversal,
        retrieve_only=False,
    )
    main_hotpot(hop_args)
    result_file = _export_predictions(data_file, output_dir, predictions_file)

    print(f"data_file={data_file}")
    print(f"gold_file={gold_file}")
    print(f"predictions_file={predictions_file}")
    print(f"hoprag_result_file={result_file}")


if __name__ == "__main__":
    main()
