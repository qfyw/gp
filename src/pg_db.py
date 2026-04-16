from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg

from .config import KB_NAMESPACE, POSTGRES_CONNINFO


@dataclass(frozen=True)
class KeywordRow:
    content: str
    source: str
    score: float


def _require_dsn() -> str:
    dsn = (POSTGRES_CONNINFO or "").strip()
    if not dsn:
        raise RuntimeError("未配置 POSTGRES_DSN（.env），无法使用 PostgreSQL/pgvector。")
    return dsn


def get_conn():
    return psycopg.connect(_require_dsn(), autocommit=True)


def ensure_db_objects() -> None:
    """
    创建 pgvector 扩展（如有权限）和关键词检索表。
    - pgvector 扩展不一定有权限创建；但 langchain_postgres 在写入向量表时会报更明确错误。
    - 关键词检索使用 pg_trgm 优先，否则回退 ILIKE（索引不可用但能跑）。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # pgvector extension (may require superuser; ignore if fails)
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception:
                pass

            # keyword search extensions / table
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            except Exception:
                pass

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_keyword_chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    filename TEXT,
                    url TEXT
                );
                """
            )

            # migrate old table (best-effort): add namespace if it doesn't exist
            try:
                cur.execute(
                    "ALTER TABLE rag_keyword_chunks ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default';"
                )
            except Exception:
                # if alter fails due to permissions/locks, we still keep running;
                # keyword isolation may not work until schema is updated.
                pass

            # trigram index (best-effort)
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS rag_keyword_chunks_content_trgm
                    ON rag_keyword_chunks
                    USING GIN (content gin_trgm_ops);
                    """
                )
            except Exception:
                pass

            # full-text index for BM25-like retrieval (best-effort)
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS rag_keyword_chunks_content_tsv
                    ON rag_keyword_chunks
                    USING GIN (to_tsvector('simple', content));
                    """
                )
            except Exception:
                pass


def stable_chunk_id(content: str, source: str = "", filename: str = "", url: str = "") -> str:
    h = hashlib.sha256()
    h.update((content or "").encode("utf-8"))
    h.update(b"\n")
    h.update((source or "").encode("utf-8"))
    h.update(b"\n")
    h.update((filename or "").encode("utf-8"))
    h.update(b"\n")
    h.update((url or "").encode("utf-8"))
    return h.hexdigest()


def upsert_keyword_chunks(rows: Iterable[Dict[str, Any]]) -> None:
    ensure_db_objects()
    vals: List[Tuple[str, str, str, Optional[str], Optional[str]]] = []
    for r in rows:
        content = (r.get("content") or "").strip()
        source = (r.get("source") or "").strip() or "未知来源"
        filename = (r.get("filename") or None) if r.get("filename") else None
        url = (r.get("url") or None) if r.get("url") else None
        if not content:
            continue
        _id = stable_chunk_id(
            content, source=source, filename=filename or "", url=url or ""
        )
        # include namespace in primary key to avoid collisions across isolated KBs
        _id = f"{KB_NAMESPACE}:{_id}"
        vals.append((_id, content, source, filename, url))
    if not vals:
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_keyword_chunks (id, content, source, namespace, filename, url)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
                """,
                [(i, c, s, KB_NAMESPACE, f, u) for (i, c, s, f, u) in vals],
            )


def keyword_search(query: str, top_k: int = 5) -> List[KeywordRow]:
    ensure_db_objects()
    q = (query or "").strip()
    if not q:
        return []

    # Prefer trigram similarity if available; otherwise fallback to ILIKE.
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT content, source, similarity(content, %s) AS score
                    FROM rag_keyword_chunks
                    WHERE namespace = %s
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (q, KB_NAMESPACE, top_k),
                )
                rows = cur.fetchall()
                return [KeywordRow(content=r[0], source=r[1], score=float(r[2] or 0.0)) for r in rows]
            except Exception:
                like = f"%{q}%"
                cur.execute(
                    """
                    SELECT content, source, 0.0 AS score
                    FROM rag_keyword_chunks
                    WHERE namespace = %s AND content ILIKE %s
                    LIMIT %s;
                    """,
                    (KB_NAMESPACE, like, top_k),
                )
                rows = cur.fetchall()
                return [KeywordRow(content=r[0], source=r[1], score=0.0) for r in rows]


def bm25_search(query: str, top_k: int = 5) -> List[KeywordRow]:
    """
    PostgreSQL 全文检索（tsvector + ts_rank_cd）。
    说明：这不是 Lucene/ES 的原生 BM25，但在工程上可作为 BM25-like 稀疏检索通道。
    """
    ensure_db_objects()
    q = (query or "").strip()
    if not q:
        return []

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    WITH q AS (
                        SELECT websearch_to_tsquery('simple', %s) AS tsq
                    )
                    SELECT
                        c.content,
                        c.source,
                        ts_rank_cd(to_tsvector('simple', c.content), q.tsq) AS score
                    FROM rag_keyword_chunks c, q
                    WHERE c.namespace = %s
                      AND q.tsq IS NOT NULL
                      AND to_tsvector('simple', c.content) @@ q.tsq
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (q, KB_NAMESPACE, top_k),
                )
                rows = cur.fetchall()
                return [KeywordRow(content=r[0], source=r[1], score=float(r[2] or 0.0)) for r in rows]
            except Exception:
                # Fallback when tsquery parsing/config fails on some inputs.
                like = f"%{q}%"
                cur.execute(
                    """
                    SELECT content, source, 0.0 AS score
                    FROM rag_keyword_chunks
                    WHERE namespace = %s AND content ILIKE %s
                    LIMIT %s;
                    """,
                    (KB_NAMESPACE, like, top_k),
                )
                rows = cur.fetchall()
                return [KeywordRow(content=r[0], source=r[1], score=0.0) for r in rows]

