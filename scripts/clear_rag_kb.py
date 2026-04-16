#!/usr/bin/env python3
"""
清空与本项目 RAG 相关的本地状态（慎用）。

会删除：
- PostgreSQL 中指定 PGVector collection 的全部向量行及 collection 记录
- 表 rag_keyword_chunks 的全部行（关键词通道与 collection 未绑定，一并清空）
- data/docs_index.json 重置为 []
- data/knowledge_graph.pkl（若存在）

用法（在项目根目录）:
  python scripts/clear_rag_kb.py --yes

更稳妥做法：不改旧库，在 .env 里另设 PGVECTOR_COLLECTION=crud_rag_eval 做对照实验。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, KB_NAMESPACE, KG_PERSIST_PATH, PGVECTOR_COLLECTION
from src.pg_db import get_conn


def main() -> int:
    parser = argparse.ArgumentParser(description="清空 RAG 向量 collection、关键词表与本地索引")
    parser.add_argument(
        "--collection",
        default=PGVECTOR_COLLECTION,
        help="要删除的 langchain_pg_collection 名称（默认与 .env 中 PGVECTOR_COLLECTION 一致）",
    )
    parser.add_argument("--yes", action="store_true", help="确认执行破坏性操作")
    args = parser.parse_args()

    if not args.yes:
        print("这是破坏性操作。若确认清空，请追加参数 --yes", file=sys.stderr)
        print("提示：也可在 .env 设置 PGVECTOR_COLLECTION=新名字，避免删旧库。", file=sys.stderr)
        return 1

    col = (args.collection or "").strip()
    with get_conn() as conn:
        with conn.cursor() as cur:
            if col:
                cur.execute(
                    "SELECT uuid FROM langchain_pg_collection WHERE name = %s;",
                    (col,),
                )
                row = cur.fetchone()
                if row:
                    cid = row[0]
                    cur.execute(
                        "DELETE FROM langchain_pg_embedding WHERE collection_id = %s;",
                        (cid,),
                    )
                    cur.execute(
                        "DELETE FROM langchain_pg_collection WHERE name = %s;",
                        (col,),
                    )
                    print(f"已删除向量 collection: {col}")
                else:
                    print(f"未找到 collection「{col}」，跳过向量表删除。")

            # 关键词表已支持 namespace 隔离：只清当前 namespace（更适合“隔离库演示”）
            try:
                cur.execute("DELETE FROM rag_keyword_chunks WHERE namespace = %s;", (KB_NAMESPACE,))
                print(f"已清空 rag_keyword_chunks（namespace={KB_NAMESPACE}）")
            except Exception:
                # 兼容旧 schema（无 namespace 列）
                cur.execute("TRUNCATE TABLE rag_keyword_chunks;")
                print("已 TRUNCATE rag_keyword_chunks（legacy schema，无 namespace）")

    docs_index = DATA_DIR / f"docs_index_{KB_NAMESPACE}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with docs_index.open("w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)
    print(f"已清空 {docs_index}")

    if KG_PERSIST_PATH.is_file():
        KG_PERSIST_PATH.unlink()
        print(f"已删除 {KG_PERSIST_PATH}")
    else:
        print("无 knowledge_graph.pkl（当前 namespace），跳过")

    print("完成。请重启 Streamlit / 重新 load_vectorstore + build_or_load_graph。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
