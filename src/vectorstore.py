from __future__ import annotations

from typing import Optional

from typing import Any

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore

from .config import PGVECTOR_COLLECTION, POSTGRES_DSN, POSTGRES_SQLALCHEMY_URL
from .data_loader import get_embeddings
from .pg_db import ensure_db_objects


def load_vectorstore(collection_name: str | None = None) -> Optional[PGVector]:
    """
    返回 PGVector 向量库句柄（PostgreSQL + pgvector）。
    说明：PGVector 本身不需要“加载持久化目录”，数据都在 PostgreSQL。
    """
    if not (POSTGRES_DSN or "").strip():
        return None
    ensure_db_objects()
    return PGVector(
        connection=POSTGRES_SQLALCHEMY_URL,
        embeddings=get_embeddings(),
        collection_name=collection_name or PGVECTOR_COLLECTION,
        use_jsonb=True,
    )

