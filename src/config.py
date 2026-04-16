# 配置：路径与 API 等
import os
from pathlib import Path

# 加载 .env（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# 国内网络：默认使用 HuggingFace 镜像，避免连接 huggingface.co 超时/被拒
# 若需直连官方，可在 .env 中设置 HF_ENDPOINT=https://huggingface.co 或 HF_ENDPOINT=
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 嵌入模型 (sentence-transformers，可换为 OpenAI 等)
# 国内网络可先在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 若模型已下载到本地，可设置 EMBEDDING_MODEL=本地路径 并设置 EMBEDDING_LOCAL_ONLY=true 避免联网
EMBEDDING_LOCAL_ONLY = os.getenv("EMBEDDING_LOCAL_ONLY", "").lower() in ("1", "true", "yes")

# Web Search (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Web fetch (URL ingest)
WEB_FETCH_VERIFY_SSL = os.getenv("WEB_FETCH_VERIFY_SSL", "true").lower() in ("1", "true", "yes")

# Postgres / pgvector
# 例: postgresql://postgres:password@localhost:5432/rag
POSTGRES_DSN = (os.getenv("POSTGRES_DSN", "") or "").strip()


def _as_psycopg_conninfo(dsn: str) -> str:
    """
    psycopg(3) 需要 libpq conninfo 或标准 postgresql:// URL，
    不能包含 SQLAlchemy 的 driver 标记（例如 postgresql+psycopg://）。
    """
    dsn = (dsn or "").strip()
    if dsn.startswith("postgres://"):
        return "postgresql://" + dsn[len("postgres://") :]
    if dsn.startswith("postgresql+"):
        return "postgresql://" + dsn[len("postgresql+") :]
    return dsn


def _as_sqlalchemy_url(dsn: str) -> str:
    """
    SQLAlchemy 使用 `postgresql://...` 会默认走 psycopg2。
    本项目优先 psycopg3，因此在未指定 driver 时改写为 `postgresql+psycopg://...`。
    """
    dsn = (dsn or "").strip()
    if not dsn:
        return dsn
    if dsn.startswith("postgresql+"):
        return dsn
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://") :]
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    return dsn


POSTGRES_CONNINFO = _as_psycopg_conninfo(POSTGRES_DSN)
POSTGRES_SQLALCHEMY_URL = _as_sqlalchemy_url(POSTGRES_DSN)
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "rag_docs")

# 知识库“命名空间”（用于隔离关键词表、图谱文件、docs_index）。
# 默认与 PGVECTOR_COLLECTION 一致：你只需要改 PGVECTOR_COLLECTION，就能得到一个完全隔离的新演示库。
KB_NAMESPACE = (os.getenv("KB_NAMESPACE", "") or PGVECTOR_COLLECTION or "default").strip()

# 图谱持久化文件：按命名空间隔离
KG_PERSIST_PATH = DATA_DIR / f"knowledge_graph_{KB_NAMESPACE}.pkl"

# 知识图谱三元组抽取（入库/上传）：KG_LLM_INGEST_RATE 控制「每段文本」是否调用 LLM。
#   1.0 = 每段都先 LLM 再必要时回退规则（最慢、图最丰富）
#   0.0 = 入库不调 LLM，仅用规则（最快，适合大批量语料）
#   0~1 = 每段以该概率尝试 LLM，否则直接用规则（折中）
def _float_01(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, v))


KG_LLM_INGEST_RATE = _float_01("KG_LLM_INGEST_RATE", 1.0)

# LLM API（预留，可从环境变量读取）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")  # 如 DeepSeek: https://api.deepseek.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")  # 或 qwen、kimi 等


def _positive_int(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(1, min(v, 32))


def _positive_int_loose(name: str, default: int, *, cap: int = 64) -> int:
    """与 _positive_int 类似，上限可放宽（图谱扩展条数可能 >32）。"""
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(1, min(v, cap))


# 混合检索 top-k / 图谱扩展（网页端与 run_eval 等默认；可用 .env 覆盖）
RETRIEVAL_VECTOR_TOP_K = _positive_int("RETRIEVAL_VECTOR_TOP_K", 5)
RETRIEVAL_KEYWORD_TOP_K = _positive_int("RETRIEVAL_KEYWORD_TOP_K", 5)
RETRIEVAL_GRAPH_MAX = _positive_int_loose("RETRIEVAL_GRAPH_MAX", 10, cap=64)

# 含 PascalCase 等强区分词时：过滤后若无任何命中，是否禁止回退到「全量误召回」（建议演示开 true）
RETRIEVAL_DISTINCTIVE_STRICT = os.getenv("RETRIEVAL_DISTINCTIVE_STRICT", "1").lower() in (
    "1",
    "true",
    "yes",
)

# 检索重排序：交叉编码器（sentence-transformers CrossEncoder）
# 开启后向量/关键词会先扩大召回，再重排并截断到原 top_k 预算
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

RERANK_RECALL_MULT = _positive_int("RERANK_RECALL_MULT", 3)
RERANK_MAX_PASSAGE_CHARS = _positive_int("RERANK_MAX_PASSAGE_CHARS", 1800)

# 重排序后最终喂给 LLM 的“文档 chunk”数量上限（不含图谱 chunk）。
# 不设置或空：沿用 hybrid 逻辑 cap_default = vector_top_k + keyword_top_k + BM25_TOP_K（若开 BM25）。
RERANK_DOC_CAP = _positive_int("RERANK_DOC_CAP", 0) if (os.getenv("RERANK_DOC_CAP", "") or "").strip() else 0

# 关键词检索增强：在 pg_trgm 之外启用 PostgreSQL 全文检索（BM25-like）。
BM25_ENABLED = os.getenv("BM25_ENABLED", "1").lower() in ("1", "true", "yes")
BM25_TOP_K = _positive_int("BM25_TOP_K", 5)

# 是否严格仅基于知识库回答（禁止模型用常识补全）
KB_STRICT_ONLY = os.getenv("KB_STRICT_ONLY", "1").lower() in ("1", "true", "yes")
# 仅文档模式：关闭图谱检索与图谱推理（演示可更稳定）
DOC_ONLY_MODE = os.getenv("DOC_ONLY_MODE", "0").lower() in ("1", "true", "yes")
# 仅内部文档作答：图谱可用于推理提示，但最终事实只允许来自内部文档
INTERNAL_DOC_ONLY_ANSWER = os.getenv("INTERNAL_DOC_ONLY_ANSWER", "0").lower() in ("1", "true", "yes")

# 三路融合（vector / keyword-trgm / bm25）参数
FUSION_METHOD = (os.getenv("FUSION_METHOD", "rrf") or "rrf").strip().lower()
RRF_K = _positive_int("RRF_K", 60)


def _bounded_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(v, hi))


# 合成提示里每条「内部文档」截断长度（字符）；Top-K 增大时可略降本条以控制总上下文
INTERNAL_DOC_PROMPT_CHARS = _bounded_int("INTERNAL_DOC_PROMPT_CHARS", 1100, lo=400, hi=8000)

# 入库切块默认值（网页上传、process_uploaded_files；ingest 脚本未显式传参时亦应对齐）
# 过小（如 128）+ 无 overlap 易把数字与上下文切开，加重检索混篇与假拒答
INGEST_CHUNK_SIZE = _bounded_int("INGEST_CHUNK_SIZE", 600, lo=128, hi=4000)
INGEST_CHUNK_OVERLAP = _bounded_int("INGEST_CHUNK_OVERLAP", 120, lo=0, hi=800)
