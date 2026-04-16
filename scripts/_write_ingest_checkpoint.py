"""写入入库进度检查点（供强制停止后人工续跑参考）。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def main() -> int:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if not dsn:
        print("未配置 POSTGRES_DSN", file=sys.stderr)
        return 1
    ns = (os.getenv("KB_NAMESPACE") or os.getenv("PGVECTOR_COLLECTION") or "crud_eval").strip()
    col = (os.getenv("PGVECTOR_COLLECTION") or "crud_eval").strip()
    conn = psycopg.connect(dsn, autocommit=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM rag_keyword_chunks WHERE namespace = %s;", (ns,)
    )
    kw = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*)
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s;
        """,
        (col,),
    )
    vec = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(DISTINCT (e.cmetadata->>'source'))
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s;
        """,
        (col,),
    )
    distinct = cur.fetchone()[0]
    conn.close()

    out = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "强制停止前快照；续跑请以 ingest_crud_news 的 --skip-docs 为准，勿直接用 distinct 当 skip（除非确认单进程顺序入库）。",
        "kb_namespace": ns,
        "pgvector_collection": col,
        "rag_keyword_chunks_rows": kw,
        "langchain_pg_embedding_rows": vec,
        "distinct_source_articles_approx": distinct,
        "resume_hint": (
            "若本趟从 --skip-docs 0 开始且未重复跑：续跑可试 --skip-docs = 本趟脚本打印的 total；"
            "若已乱序/双进程，请用 scripts/_ingest_progress.py 对照语料后再定 skip。"
        ),
    }
    path = ROOT / "data" / "ingest_checkpoint_forced_stop.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
