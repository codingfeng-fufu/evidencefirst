"""
Microsoft GraphRAG 基线（CLI 方式）。
对每道题的 supporting passages 建本地索引，用 local search 查询。
LLM: DashScope OpenAI-compatible endpoint
Embedding: 本地 embed_server.py（端口 8765）
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import textwrap
import time

import pandas as pd

import config

EMBED_BASE_URL = "http://127.0.0.1:8765/v1"
EMBED_MODEL    = "all-MiniLM-L6-v2"

SETTINGS_YAML = textwrap.dedent("""\
encoding_model: cl100k_base

llm:
  api_key: {api_key}
  type: openai_chat
  model: {model}
  api_base: {base_url}
  model_supports_json: true
  max_tokens: 500
  temperature: 0

parallelization:
  stagger: 0.1
  num_threads: 2

async_mode: asyncio

embeddings:
  async_mode: asyncio
  llm:
    api_key: dummy
    type: openai_embedding
    model: {embed_model}
    api_base: {embed_base_url}
    max_retries: 3
  vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default
    overwrite: true

input:
  type: file
  file_type: text
  base_dir: input
  file_encoding: utf-8
  file_pattern: ".*[.]txt$"

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

cache:
  type: file
  base_dir: cache

reporting:
  type: file
  base_dir: logs

storage:
  type: file
  base_dir: output

skip_workflows: ["create_final_community_reports"]

entity_extraction:
  max_gleanings: 0

community_reports:
  max_length: 500
  max_input_length: 4000

claim_extraction:
  enabled: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  max_tokens: 500
""")

FAILURE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results", "ms_graphrag_failures")
)

INDEX_RUNNER = r"""
import asyncio
import json
import sys
from pathlib import Path

from graphrag.api.index import _get_workflows_list
from graphrag.config.load_config import load_config
from graphrag.config.resolve_path import resolve_paths
from graphrag.index.run.run_workflows import run_workflows


async def main(root_dir: str):
    config = load_config(Path(root_dir), None)
    resolve_paths(config)
    workflows = [w for w in _get_workflows_list(config) if w != "create_final_community_reports"]
    outputs = []
    async for output in run_workflows(workflows, config):
        outputs.append({
            "workflow": output.workflow,
            "errors": [str(e) for e in (output.errors or [])],
        })
    errors = [o for o in outputs if o["errors"]]
    print(json.dumps({"outputs": outputs, "errors": errors}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


asyncio.run(main(sys.argv[1]))
"""


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _empty_community_reports_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": pd.Series(dtype="string"),
        "community": pd.Series(dtype="int64"),
        "level": pd.Series(dtype="int64"),
        "title": pd.Series(dtype="string"),
        "summary": pd.Series(dtype="string"),
        "full_content": pd.Series(dtype="string"),
        "rank": pd.Series(dtype="float64"),
    })


def _prepare_empty_community_reports(tmp_dir: str) -> None:
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "create_final_community_reports.parquet")
    _empty_community_reports_df().to_parquet(out_path, index=False)


def _save_failure_artifacts(
    case_id: str,
    stage: str,
    question: str,
    tmp_dir: str,
    stdout="",
    stderr="",
) -> None:
    os.makedirs(FAILURE_DIR, exist_ok=True)
    fail_dir = os.path.join(FAILURE_DIR, case_id)
    if os.path.exists(fail_dir):
        shutil.rmtree(fail_dir, ignore_errors=True)
    os.makedirs(fail_dir, exist_ok=True)

    meta = {
        "case_id": case_id,
        "stage": stage,
        "question": question,
        "tmp_dir": tmp_dir,
    }
    with open(os.path.join(fail_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(fail_dir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(_safe_text(stdout))
    with open(os.path.join(fail_dir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(_safe_text(stderr))

    for name in ("settings.yaml", ".env"):
        src = os.path.join(tmp_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fail_dir, name))

    for name in ("logs", "output", "input"):
        src = os.path.join(tmp_dir, name)
        dst = os.path.join(fail_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _context_to_text(context) -> list:
    """返回 [(title, text), ...] 列表"""
    if isinstance(context, dict):
        titles    = context["title"]
        sentences = context["sentences"]
    else:
        titles    = [t for t, _ in context]
        sentences = [s for _, s in context]
    return [(t, " ".join(s)) for t, s in zip(titles, sentences)]


def _run_index_workflows(tmp_dir: str) -> tuple[bool, str, str]:
    r = subprocess.run(
        ["python3", "-c", INDEX_RUNNER, tmp_dir],
        capture_output=True,
        text=True,
        timeout=config.MS_GRAPHRAG_INDEX_TIMEOUT,
        env={**os.environ, "GRAPHRAG_API_KEY": config.OPENAI_API_KEY},
    )
    if r.returncode == 0:
        return True, "", r.stdout
    detail = (r.stdout or "").strip() or (r.stderr or "").strip()
    return False, detail, r.stderr


def ms_graphrag(
    question: str,
    context,
    retries: int | None = None,
    case_id: str | None = None,
) -> str:
    retries = config.MS_GRAPHRAG_RETRIES if retries is None else retries
    case_id = case_id or f"q_{abs(hash(question))}"
    for attempt in range(retries):
        tmp_dir = None
        try:
            signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("per-question timeout")))
            signal.alarm(config.MS_GRAPHRAG_TOTAL_TIMEOUT)
            tmp_dir = tempfile.mkdtemp(prefix="graphrag_")
            answer  = _run_cli(question, context, tmp_dir, case_id=case_id)
            signal.alarm(0)
            return answer
        except Exception as e:
            signal.alarm(0)
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  [ERR] ms_graphrag: {e}")
                return ""
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_cli(question: str, context, tmp_dir: str, case_id: str) -> str:
    # 写 settings.yaml
    with open(os.path.join(tmp_dir, "settings.yaml"), "w") as f:
        f.write(SETTINGS_YAML.format(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            embed_model=EMBED_MODEL,
            embed_base_url=EMBED_BASE_URL,
        ))

    # 写 .env（graphrag 需要）
    with open(os.path.join(tmp_dir, ".env"), "w") as f:
        f.write(f"GRAPHRAG_API_KEY={config.OPENAI_API_KEY}\n")

    # GraphRAG local search can work with empty reports; pre-create the parquet so
    # generate_text_embeddings can still run after we skip the flaky report workflow.
    _prepare_empty_community_reports(tmp_dir)

    # 写 input 文本
    input_dir = os.path.join(tmp_dir, "input")
    os.makedirs(input_dir)
    passages = _context_to_text(context)
    for title, text in passages:
        fname = os.path.join(input_dir, f"{title[:50].replace('/', '_')}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{text}")

    # graphrag index
    try:
        ok, detail, raw_stderr = _run_index_workflows(tmp_dir)
    except subprocess.TimeoutExpired as e:
        _save_failure_artifacts(
            case_id=case_id,
            stage="index_timeout",
            question=question,
            tmp_dir=tmp_dir,
            stdout=e.stdout or "",
            stderr=e.stderr or f"index timeout ({config.MS_GRAPHRAG_INDEX_TIMEOUT}s)",
        )
        raise RuntimeError(f"index timeout ({config.MS_GRAPHRAG_INDEX_TIMEOUT}s)")
    except Exception as e:
        _save_failure_artifacts(
            case_id=case_id,
            stage="index_failed",
            question=question,
            tmp_dir=tmp_dir,
            stdout="",
            stderr=str(e),
        )
        raise RuntimeError(f"index failed: {e}")
    if not ok:
        _save_failure_artifacts(
            case_id=case_id,
            stage="index_failed",
            question=question,
            tmp_dir=tmp_dir,
            stdout=detail,
            stderr=raw_stderr,
        )
        raise RuntimeError(f"index failed: {detail[-500:]}")

    # graphrag query (local search)
    try:
        r = subprocess.run(
            ["python3", "-m", "graphrag", "query",
             "--root", tmp_dir,
             "--method", "local",
             "--query", question],
            capture_output=True, text=True, timeout=config.MS_GRAPHRAG_QUERY_TIMEOUT,
            env={**os.environ, "GRAPHRAG_API_KEY": config.OPENAI_API_KEY}
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-f", f"graphrag query.*{tmp_dir}"], capture_output=True)
        _save_failure_artifacts(
            case_id=case_id,
            stage="query_timeout",
            question=question,
            tmp_dir=tmp_dir,
            stdout="",
            stderr=f"query timeout ({config.MS_GRAPHRAG_QUERY_TIMEOUT}s)",
        )
        raise RuntimeError(f"query timeout ({config.MS_GRAPHRAG_QUERY_TIMEOUT}s)")
    if r.returncode != 0:
        _save_failure_artifacts(
            case_id=case_id,
            stage="query_failed",
            question=question,
            tmp_dir=tmp_dir,
            stdout=r.stdout,
            stderr=r.stderr,
        )
        raise RuntimeError(f"query failed: {r.stderr[-300:]}")

    # 提取答案：取 "SUCCESS: Local Search Response:" 之后的内容
    stdout = r.stdout
    marker = "SUCCESS: Local Search Response:"
    idx = stdout.find(marker)
    if idx != -1:
        answer_text = stdout[idx + len(marker):].strip()
        # 取第一段（遇到空行截止）
        first_para = answer_text.split("\n\n")[0].strip()
        if not first_para:
            _save_failure_artifacts(
                case_id=case_id,
                stage="empty_query_response",
                question=question,
                tmp_dir=tmp_dir,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        return first_para if first_para else ""
    # fallback：去掉 INFO/SUCCESS/WARNING 及 creating 开头的行
    lines = [l.strip() for l in stdout.split("\n")
             if l.strip()
             and not l.startswith("INFO")
             and not l.startswith("SUCCESS")
             and not l.startswith("WARNING")
             and not l.startswith("creating")]
    answer = lines[0] if lines else ""
    if not answer:
        _save_failure_artifacts(
            case_id=case_id,
            stage="empty_query_parse",
            question=question,
            tmp_dir=tmp_dir,
            stdout=r.stdout,
            stderr=r.stderr,
        )
    return answer
