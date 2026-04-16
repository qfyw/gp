"""一次性：从 PG 推断 ingest_crud_news 进度（按分片顺序）。"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
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
    conn = psycopg.connect(dsn, autocommit=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(DISTINCT (e.cmetadata->>'source'))
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        """,
        ("crud_eval",),
    )
    distinct = cur.fetchone()[0]
    print("distinct_sources(唯一篇/文件名):", distinct)

    cur.execute(
        """
        SELECT e.cmetadata->>'source' AS s
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        GROUP BY 1
        """,
        ("crud_eval",),
    )
    pat = re.compile(r"^(.+)#L(\d+)\.txt$")
    by_shard: dict[str, set[int]] = defaultdict(set)
    bad = 0
    for (s,) in cur:
        if not s:
            bad += 1
            continue
        m = pat.match(s)
        if m:
            by_shard[m.group(1)].add(int(m.group(2)))
        else:
            bad += 1
    print("shards_with_data:", len(by_shard), "unparsed_sources:", bad)
    print("unique_articles(parsed):", sum(len(v) for v in by_shard.values()))

    docs_dir = ROOT / "CRUD_RAG" / "data" / "80000_docs"
    shards = sorted([p for p in docs_dir.glob("documents*") if p.is_file()])
    expect: dict[str, int] = {}
    for p in shards:
        txt = p.read_text(encoding="utf-8", errors="replace")
        expect[p.name] = sum(1 for line in txt.splitlines() if len(line.strip()) >= 40)

    for i, p in enumerate(shards, 1):
        name = p.name
        exp = expect[name]
        got = len(by_shard.get(name, set()))
        if got < exp:
            mx = max(by_shard[name]) if by_shard[name] else 0
            print(
                f"按脚本顺序第一个未满: 第 {i}/{len(shards)} 个分片 | {name} | "
                f"已 {got}/{exp} 篇 | 本文件内已见最大物理行号 L{mx}"
            )
            break
    else:
        print("按顺序所有分片均已达到预期篇数（或库中无缺失）")

    print("预估可入库总行(≥40字):", sum(expect.values()))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
