import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
_load_env_file(_repo_root / ".env")
_load_env_file(_pkg_dir / ".env")

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
LLM_MODEL        = os.environ.get("LLM_MODEL", "qwen-plus")
LLM_BASE_URL     = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL      = "all-MiniLM-L6-v2"

MS_GRAPHRAG_RETRIES       = int(os.environ.get("MS_GRAPHRAG_RETRIES", "2"))
MS_GRAPHRAG_TOTAL_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_TOTAL_TIMEOUT", "240"))
MS_GRAPHRAG_INDEX_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_INDEX_TIMEOUT", "180"))
MS_GRAPHRAG_QUERY_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_QUERY_TIMEOUT", "90"))

MAX_ITER         = 3      # 最多迭代轮次
THRESHOLD        = 0.5    # VA 分数阈值
MAX_TRIPLES      = 50     # 传给 AGA 的最大 triple 数
KG_HOPS          = 2      # BFS 跳数
ENTITY_SIM_THRESH = 0.75  # embedding 相似度阈值
TOP_K_FALLBACK   = 3      # fallback seed 节点数
ENTITY_LINK_THRESH = ENTITY_SIM_THRESH
ENTITY_LINK_TOP_K = TOP_K_FALLBACK
PASSAGE_FALLBACK_THRESH = 15  # KG triples < this → also use BM25 passage retrieval
PASSAGE_TOP_K    = 5      # number of passages to retrieve as fallback
CONTEXT_FALLBACK_TOP_K = 10
CONTEXT_FALLBACK_SCORE_THRESHOLD = 0.80
GLOBAL_CONTEXT_TOP_K = 8
EVIDENCE_AUG_TOP_K = 8
EVIDENCE_AUG_MAX_PASSAGE_CHARS = 7000
EVIDENCE_AUG_MAX_CANDIDATES = 24
EVIDENCE_AUG_CHOICE_CONFIDENCE_THRESHOLD = 0.72
EVIDENCE_FIRST_POSTPROCESS = os.environ.get("EVIDENCE_FIRST_POSTPROCESS", "1") != "0"
EVIDENCE_FIRST_POSTPROCESS_TOP_K = int(os.environ.get("EVIDENCE_FIRST_POSTPROCESS_TOP_K", "8"))

W_FACTUAL    = 0.4
W_COMPLETE   = 0.3
W_CONSISTENT = 0.3

# 实验规模（先跑小的验证，再跑全量）
QUICK_N   = 50     # 快速验证用
FULL_N    = 500    # 正式实验用（bridge 250 + comparison 250）
