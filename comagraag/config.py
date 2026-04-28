import os

# 从 .env 文件加载配置（如果存在）
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
LLM_MODEL        = os.environ.get("LLM_MODEL", "qwen-plus")
LLM_BASE_URL     = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL      = "all-MiniLM-L6-v2"

MS_GRAPHRAG_RETRIES       = int(os.environ.get("MS_GRAPHRAG_RETRIES", "2"))
MS_GRAPHRAG_TOTAL_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_TOTAL_TIMEOUT", "240"))
MS_GRAPHRAG_INDEX_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_INDEX_TIMEOUT", "180"))
MS_GRAPHRAG_QUERY_TIMEOUT = int(os.environ.get("MS_GRAPHRAG_QUERY_TIMEOUT", "90"))

MAX_ITER         = 2      # 最多迭代轮次
THRESHOLD        = 0.5    # VA 分数阈值
MAX_TRIPLES      = 50     # 传给 AGA 的最大 triple 数
KG_HOPS          = 2      # BFS 跳数
ENTITY_SIM_THRESH = 0.75  # embedding 相似度阈值
TOP_K_FALLBACK   = 3      # fallback seed 节点数
PASSAGE_FALLBACK_THRESH = 15  # KG triples < this → also use BM25 passage retrieval
PASSAGE_TOP_K    = 5      # number of passages to retrieve as fallback

W_FACTUAL    = 0.4
W_COMPLETE   = 0.3
W_CONSISTENT = 0.3

# 实验规模（先跑小的验证，再跑全量）
QUICK_N   = 50     # 快速验证用
FULL_N    = 500    # 正式实验用（bridge 250 + comparison 250）
