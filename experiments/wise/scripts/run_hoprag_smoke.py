#!/usr/bin/env python3
"""Run a bounded HopRAG smoke test against the local Neo4j instance."""

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
RUN_DIR = ROOT / "external_runs" / "hoprag_smoke"

sys.path.insert(0, str(HOPRAG_DIR))
os.chdir(HOPRAG_DIR)

import config  # noqa: E402
from data_preprocess import process_data  # noqa: E402
from HopBuilder import main_edges_index, main_nodes  # noqa: E402
from HopGenerator import main_hotpot  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


SMOKE_RECORD = {
    "_id": "hoprag_smoke_donnie_mls",
    "answer": "Major League Soccer",
    "question": (
        "Donnie Smith who plays as a left back for New England Revolution "
        "belongs to what league featuring 22 teams?"
    ),
    "supporting_facts": [["Donnie Smith", 0], ["Major League Soccer", 1]],
    "context": [
        [
            "Donnie Smith",
            [
                (
                    'Donald W. "Donnie" Smith is an American soccer player who '
                    "plays as a left back for New England Revolution in Major "
                    "League Soccer."
                )
            ],
        ],
        [
            "Major League Soccer",
            [
                (
                    "Major League Soccer is a men's professional soccer league "
                    "representing the sport's highest level in the United States "
                    "and Canada."
                ),
                "The league comprises 22 teams, 19 in the U.S. and 3 in Canada.",
            ],
        ],
    ],
    "type": "bridge",
    "level": "smoke",
}


def _safe_identifier(value: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe {kind}: {value!r}")
    return value


def _reset_smoke_graph() -> None:
    label = _safe_identifier(config.node_name, "node label")
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


def _prepare_smoke_data() -> tuple[Path, Path]:
    data_path = RUN_DIR / "hotpot_smoke.jsonl"
    docs_dir = RUN_DIR / "hotpot_smoke_docs"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with data_path.open("w") as file:
        file.write(json.dumps(SMOKE_RECORD) + "\n")
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    process_data(str(data_path), str(docs_dir), str(data_path))
    return data_path, docs_dir


def run(retrieve_only: bool) -> None:
    data_path, docs_dir = _prepare_smoke_data()
    cache_dir = RUN_DIR / "cache"
    output_dir = RUN_DIR / "output"
    for path in (cache_dir, output_dir):
        if path.exists():
            shutil.rmtree(path)

    _reset_smoke_graph()
    main_nodes(
        cache_dir=str(cache_dir),
        docs_dir=str(docs_dir),
        label=config.node_name,
        start_index=0,
        span=50,
        offline=False,
    )
    main_edges_index(
        cache_dir=str(cache_dir),
        problems_path=str(data_path),
        label=config.node_name,
    )

    args = SimpleNamespace(
        model_name=config.default_gpt_model,
        traversal_model=config.traversal_model,
        embedding_model=config.embed_model,
        rerank_model=None,
        data_path=str(data_path),
        save_dir=str(output_dir),
        retriever_name="HopRetriever",
        max_hop=1,
        start_layer=0,
        max_layer=2,
        entry_type="node",
        trim=False,
        hybrid=False,
        tol=2,
        mock_dense=False,
        mock_sparse=False,
        mode="common",
        label=config.node_name + "_smoke",
        topk=2,
        traversal="bfs_sim_node",
        retrieve_only=retrieve_only,
    )
    main_hotpot(args)

    cache_files = sorted(output_dir.glob("*/cache/*.json"))
    if cache_files:
        result = json.loads(cache_files[-1].read_text())
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()
    run(retrieve_only=args.retrieve_only)


if __name__ == "__main__":
    main()
