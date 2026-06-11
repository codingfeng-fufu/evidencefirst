#!/usr/bin/env python3
"""Run a resumable, sharded official-style HopRAG HotpotQA job."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
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


def _prepare_data(items: list[dict], run_dir: Path, force: bool = False) -> tuple[Path, Path, Path]:
    data_file = run_dir / "hotpotqa.jsonl"
    gold_file = run_dir / "gold.jsonl"
    docs_dir = run_dir / "docs"
    if force and docs_dir.exists():
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
            path = docs_dir / doc_name
            if not path.exists() or force:
                path.write_text(body, encoding="utf-8")

    with data_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gold_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return data_file, docs_dir, gold_file


def _safe_identifier(value: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe {kind}: {value!r}")
    return value


def _set_hoprag_env(node_name: str, device: str) -> None:
    os.environ["HOPRAG_NODE_NAME"] = node_name
    os.environ["HOPRAG_EDGE_NAME"] = "pen2ans_" + node_name
    os.environ["HOPRAG_DEVICE"] = device


def _import_hoprag(node_name: str, device: str):
    _set_hoprag_env(node_name, device)
    sys.path.insert(0, str(HOPRAG_DIR))
    os.chdir(HOPRAG_DIR)
    import config  # noqa: PLC0415

    return config


def _reset_graph(config) -> None:
    from neo4j import GraphDatabase  # noqa: PLC0415

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


def _worker_offline(args: argparse.Namespace) -> None:
    config = _import_hoprag(args.node_name, args.device)
    from HopBuilder import QABuilder  # noqa: PLC0415

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    docid2nodes_path = args.cache_dir / "docid2nodes.json"
    node2questiondict_path = args.cache_dir / "node2questiondict.pkl"

    if docid2nodes_path.exists():
        docid2nodes = json.loads(docid2nodes_path.read_text(encoding="utf-8"))
    else:
        docid2nodes = {}
    if node2questiondict_path.exists():
        with node2questiondict_path.open("rb") as file:
            node2questiondict = pickle.load(file)
    else:
        node2questiondict = {}

    def save_cache() -> None:
        tmp_pkl = node2questiondict_path.with_suffix(".pkl.tmp")
        tmp_json = docid2nodes_path.with_suffix(".json.tmp")
        with tmp_pkl.open("wb") as file:
            pickle.dump(node2questiondict, file)
        tmp_json.write_text(json.dumps(docid2nodes), encoding="utf-8")
        os.replace(tmp_pkl, node2questiondict_path)
        os.replace(tmp_json, docid2nodes_path)

    doc_files = sorted(p.name for p in args.docs_dir.glob("*.txt") if p.is_file())
    done = set(docid2nodes)
    next_node_id = max((int(node_id) for node_id, _doc_id in node2questiondict), default=0) + 1
    chunk_size = max(1, args.offline_chunk_size)
    print(
        f"offline incremental worker docs={len(doc_files)} done={len(done)} "
        f"chunk_size={chunk_size} cache_dir={args.cache_dir}",
        flush=True,
    )

    builder = QABuilder(done=done, label=config.node_name)
    saved_since = 0
    started = time.time()
    try:
        for doc_id in doc_files:
            if doc_id in docid2nodes:
                continue
            try:
                doc = (args.docs_dir / doc_id).read_text(encoding="utf-8")
                sentence2node = builder.get_single_doc_qa(doc)
                nodes_id = []
                for _text, tup in sentence2node.items():
                    node = {"text": tup[0], "keywords": sorted(list(tup[1])), "embed": tup[2]}
                    node2questiondict[(next_node_id, doc_id)] = (node, tup[3])
                    nodes_id.append(next_node_id)
                    next_node_id += 1
                docid2nodes[doc_id] = nodes_id
                done.add(doc_id)
                builder.done = done
                saved_since += 1
            except Exception as exc:  # Match HopRAG's keep-going behavior for bad docs/API failures.
                print(f"error:{doc_id}--{exc}", flush=True)
                time.sleep(3)
                continue

            if saved_since >= chunk_size:
                save_cache()
                saved_since = 0
                elapsed = time.time() - started
                print(f"checkpoint docs_done={len(docid2nodes)}/{len(doc_files)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        save_cache()
        if builder.driver is not None:
            builder.driver.close()
            builder.driver = None
    print(f"offline worker complete docs_done={len(docid2nodes)}/{len(doc_files)}", flush=True)


def _worker_generate(args: argparse.Namespace) -> None:
    config = _import_hoprag(args.node_name, args.device)
    from HopGenerator import RagPipeline  # noqa: PLC0415

    namespace = argparse.Namespace(
        model_name=config.default_gpt_model,
        traversal_model=config.traversal_model,
        embedding_model=config.embed_model,
        rerank_model=None,
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
    pipeline = RagPipeline(namespace)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    questions = _load_items(args.data_file)
    for idx, item in enumerate(questions, start=1):
        qid = _qid(item, idx)
        cache_file = cache_dir / f"{qid.replace('/', '_')}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("response") != "I don't know because of some errors":
                    continue
            except json.JSONDecodeError:
                pass
            cache_file.unlink()
        query = item["question"]
        try:
            response, context, scores = pipeline.rag(query, retrieve_only=qid)
            context = context or []
            scores = scores or []
        except Exception as exc:  # Keep long jobs moving; error is visible in cache.
            cache_file.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")
            time.sleep(3)
            continue
        cache_file.write_text(
            json.dumps({"response": response, "context": context, "scores": scores}, ensure_ascii=False),
            encoding="utf-8",
        )
    if pipeline.retriever.driver is not None:
        pipeline.retriever.driver.close()
        pipeline.retriever.driver = None


def _partition_files(files: list[Path], count: int) -> list[list[Path]]:
    shards = [[] for _ in range(count)]
    for idx, file in enumerate(files):
        shards[idx % count].append(file)
    return shards


def _partition_items(items: list[dict], count: int) -> list[list[dict]]:
    shards = [[] for _ in range(count)]
    for idx, item in enumerate(items):
        shards[idx % count].append(item)
    return shards


def _prepare_doc_shards(docs_dir: Path, shard_root: Path, workers: int, force: bool) -> list[Path]:
    files = sorted(p for p in docs_dir.glob("*.txt") if p.is_file())
    shards = _partition_files(files, workers)
    shard_dirs = []
    if force and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(shards):
        shard_dir = shard_root / f"docs_{idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for src in shard:
            dst = shard_dir / src.name
            if dst.exists():
                continue
            os.symlink(src, dst)
        shard_dirs.append(shard_dir)
    return shard_dirs


def _prepare_question_shards(data_file: Path, shard_root: Path, workers: int, force: bool) -> list[Path]:
    items = _load_items(data_file)
    shards = _partition_items(items, workers)
    if force and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, shard in enumerate(shards):
        path = shard_root / f"questions_{idx:02d}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for item in shard:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def _run_parallel(commands: list[list[str]], log_paths: list[Path]) -> None:
    procs = []
    for cmd, log_path in zip(commands, log_paths):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((proc, log, log_path))
    failed = []
    for proc, log, log_path in procs:
        code = proc.wait()
        log.close()
        if code:
            failed.append((code, log_path))
    if failed:
        detail = ", ".join(f"{path} exited {code}" for code, path in failed)
        raise RuntimeError(f"Parallel stage failed: {detail}")


def _push_offline_caches(args: argparse.Namespace, config, docs_dir: Path, shard_count: int) -> None:
    from HopBuilder import main_nodes  # noqa: PLC0415

    online_cache = args.run_dir / "cache_hotpot_online"
    for idx in range(shard_count):
        shard_cache = args.run_dir / "offline_caches" / f"cache_{idx:02d}"
        main_nodes(
            cache_dir=str(online_cache),
            docs_dir=str(docs_dir),
            label=config.node_name,
            start_index=0,
            span=1_000_000,
            original_cache_dir=str(shard_cache),
        )


def _build_edges(args: argparse.Namespace, config, data_file: Path) -> None:
    from HopBuilder import main_edges_index  # noqa: PLC0415

    main_edges_index(
        cache_dir=str(args.run_dir / "cache_hotpot_online"),
        problems_path=str(data_file),
        label=config.node_name,
    )


def _merge_predictions(data_file: Path, output_root: Path, predictions_file: Path) -> None:
    cache_dirs = sorted(output_root.glob("gen_*/cache"))
    cache = {}
    for cache_dir in cache_dirs:
        for file in cache_dir.glob("*.json"):
            cache[file.stem] = json.loads(file.read_text(encoding="utf-8"))
    questions = _load_items(data_file)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with predictions_file.open("w", encoding="utf-8") as file:
        for i, item in enumerate(questions):
            qid = _qid(item, i)
            result = cache.get(qid.replace("/", "_"), {})
            file.write(
                json.dumps(
                    {
                        "_id": qid,
                        "id": qid,
                        "question": item.get("question", ""),
                        "response": result.get("response", ""),
                        "answer": item.get("answer", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_metrics(predictions_file: Path, gold_file: Path, metrics_file: Path) -> None:
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("compute_metrics", ROOT / "scripts" / "compute_metrics.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rows = module.evaluate(
            predictions_file,
            gold_file,
            prediction_fields=("response",),
            include_answer_fallback=False,
            null_strings_as_empty=True,
        )
    finally:
        os.chdir(old_cwd)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as file:
        import csv

        writer = csv.DictWriter(file, fieldnames=["qid", "question", "prediction", "gold", "em", "f1", "error"])
        writer.writeheader()
        writer.writerows(rows)
    em_avg = sum(float(row["em"]) for row in rows) / len(rows) if rows else 0
    f1_avg = sum(float(row["f1"]) for row in rows) / len(rows) if rows else 0
    print(f"metrics EM={em_avg:.4f} F1={f1_avg:.4f} n={len(rows)}")


def run(args: argparse.Namespace) -> None:
    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    args.run_dir = ROOT / "external_runs" / args.run_name
    if args.force and args.run_dir.exists():
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(data_path)[args.start : args.start + args.limit]
    data_file, docs_dir, gold_file = _prepare_data(items, args.run_dir, force=args.force)
    print(f"prepared questions={len(items)} docs={len(list(docs_dir.glob('*.txt')))} run_dir={args.run_dir}")

    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    offline_workers = max(1, args.offline_workers)
    generate_workers = max(1, args.generate_workers)

    if args.stage in {"all", "offline"}:
        shard_dirs = _prepare_doc_shards(docs_dir, args.run_dir / "doc_shards", offline_workers, args.force)
        commands = []
        log_paths = []
        for idx, shard_dir in enumerate(shard_dirs):
            commands.append(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--offline-worker",
                    "--docs-dir",
                    str(shard_dir),
                    "--cache-dir",
                    str(args.run_dir / "offline_caches" / f"cache_{idx:02d}"),
                    "--node-name",
                    args.node_name,
                    "--device",
                    devices[idx % len(devices)],
                    "--offline-chunk-size",
                    str(args.offline_chunk_size),
                ]
            )
            log_paths.append(args.run_dir / "logs" / f"offline_{idx:02d}.log")
        print(f"starting offline node generation workers={len(commands)}")
        _run_parallel(commands, log_paths)

    config = _import_hoprag(args.node_name, devices[0])

    if args.stage in {"all", "push"}:
        print("resetting graph and pushing offline caches to Neo4j")
        _reset_graph(config)
        _push_offline_caches(args, config, docs_dir, offline_workers)

    if args.stage in {"all", "edges"}:
        print("building edges and indexes")
        _build_edges(args, config, data_file)

    if args.stage in {"all", "generate"}:
        question_shards = _prepare_question_shards(data_file, args.run_dir / "question_shards", generate_workers, args.force)
        commands = []
        log_paths = []
        for idx, question_shard in enumerate(question_shards):
            commands.append(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--generate-worker",
                    "--data-file",
                    str(question_shard),
                    "--output-dir",
                    str(args.run_dir / "generated" / f"gen_{idx:02d}"),
                    "--node-name",
                    args.node_name,
                    "--device",
                    devices[idx % len(devices)],
                    "--max-hop",
                    str(args.max_hop),
                    "--topk",
                    str(args.topk),
                    "--traversal",
                    args.traversal,
                ]
            )
            log_paths.append(args.run_dir / "logs" / f"generate_{idx:02d}.log")
        print(f"starting generation workers={len(commands)}")
        _run_parallel(commands, log_paths)

    predictions_file = args.run_dir / "predictions.jsonl"
    metrics_file = args.run_dir / "metrics.csv"
    if args.stage in {"all", "generate", "metrics"}:
        _merge_predictions(data_file, args.run_dir / "generated", predictions_file)
        _write_metrics(predictions_file, gold_file, metrics_file)
        print(f"predictions_file={predictions_file}")
        print(f"metrics_file={metrics_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "hotpotqa_combined1000_test.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--run-name", default="hoprag_hotpot1000_bge_fast")
    parser.add_argument("--node-name", default="hotpot_bgeen_qwen_plus_1000")
    parser.add_argument("--offline-workers", type=int, default=2)
    parser.add_argument("--offline-chunk-size", type=int, default=100)
    parser.add_argument("--generate-workers", type=int, default=2)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--traversal", default="bfs_node")
    parser.add_argument("--stage", choices=["all", "offline", "push", "edges", "generate", "metrics"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--generate-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--docs-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.offline_worker:
        _worker_offline(args)
        return
    if args.generate_worker:
        _worker_generate(args)
        return
    run(args)


if __name__ == "__main__":
    main()
