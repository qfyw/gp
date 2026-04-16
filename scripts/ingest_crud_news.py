#!/usr/bin/env python3
"""
将 CRUD-RAG 的 data/80000_docs 分片按「每行一篇新闻」入库（与官方 txt 语料结构一致）。

分片文件名通常无 .txt 后缀（如 documents_dup_part_1_part_1）。

用法:
  # 全量（约 8 万+ 篇，耗时长；嵌入模型只加载一次）
  python scripts/ingest_crud_news.py --docs-dir "d:\\graduation project\\CRUD_RAG\\data\\80000_docs" --max-docs 0

  # 仅主 dup 分片、前 2000 篇
  python scripts/ingest_crud_news.py --docs-dir "..." --max-docs 2000 --glob-pattern "documents_dup*"

  # 续跑：已入库前 25000 篇，接着入库剩余（须与上次相同的 --glob-pattern / 切块参数）
  python scripts/ingest_crud_news.py --docs-dir "..." --skip-docs 25000 --max-docs 0

切块默认与 .env 中 INGEST_CHUNK_SIZE / INGEST_CHUNK_OVERLAP 一致（当前推荐约 600/120）。
若曾用旧默认 128/0 入库，建议清空向量库后按新参数全量重建，否则检索质量难提升。

提示：若要与之前「部分入库」彻底不重复，请先:
  python scripts/clear_rag_kb.py --yes

可选:
  --glob-pattern "documents*"       默认：dup + hallu 全部分片
  --upload-batch 100              每批篇数
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 若存在此文件，则在「下一次 flush() 成功完成后」干净退出（适合全量入库中途停）。
INGEST_STOP_FLAG = ROOT / "data" / ".ingest_stop_request"

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.config import INGEST_CHUNK_OVERLAP, INGEST_CHUNK_SIZE
from src.data_loader import build_or_load_graph, process_uploaded_files
from src.vectorstore import load_vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="CRUD-RAG 80000_docs 按行入库")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="指向 CRUD_RAG/data/80000_docs",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="最多入库篇数（行），0 表示不限制（整目录全量）",
    )
    parser.add_argument(
        "--skip-docs",
        type=int,
        default=0,
        help="跳过前 N 条「有效行」（长度≥40 字符），用于续跑；须与上次相同的 glob/chunk 规则",
    )
    parser.add_argument(
        "--glob-pattern",
        default="documents*",
        help="分片匹配，默认 documents*（含 documents_dup* 与 documents_hallu*）",
    )
    parser.add_argument("--upload-batch", type=int, default=100)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=INGEST_CHUNK_SIZE,
        help=f"单块最大字符数（默认 {INGEST_CHUNK_SIZE}，可由 .env INGEST_CHUNK_SIZE 覆盖）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=INGEST_CHUNK_OVERLAP,
        help=f"块间重叠字符（默认 {INGEST_CHUNK_OVERLAP}，可由 .env INGEST_CHUNK_OVERLAP 覆盖）",
    )
    args = parser.parse_args()

    d = args.docs_dir
    if not d.is_dir():
        print(f"目录不存在: {d}", file=sys.stderr)
        return 1

    shards = sorted(d.glob(args.glob_pattern))
    shards = [p for p in shards if p.is_file()]
    if not shards:
        print(f"未匹配到文件: {d}/{args.glob_pattern}", file=sys.stderr)
        return 1

    limit = args.max_docs if args.max_docs > 0 else None
    skip_left = max(0, args.skip_docs)
    skip_msg = f"跳过有效行 {skip_left} 条（续跑）；" if skip_left else ""
    print(
        f"分片数: {len(shards)}，glob={args.glob_pattern!r}，{skip_msg}"
        f"计划本趟再入库: {'剩余全部' if limit is None else limit} 篇。"
    )

    vs = load_vectorstore()
    g = build_or_load_graph()

    batch: list[tuple[bytes, str]] = []
    total = 0

    def flush() -> bool:
        """返回 True 表示应在 flush 后停止（已检测到停止标记）。"""
        nonlocal vs, g, batch
        if not batch:
            return False
        vs, g, _ = process_uploaded_files(
            batch,
            existing_vectorstore=vs,
            existing_graph=g,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        batch = []
        if INGEST_STOP_FLAG.is_file():
            print(
                f"检测到停止标记 {INGEST_STOP_FLAG}，本批已写入，正在退出。\n"
                f"本趟已累计处理有效篇数 total={total}。"
                f"若本趟从 skip-docs=0 开始，续跑请先删除该文件，再使用 --skip-docs {total}。",
                flush=True,
            )
            return True
        return False

    for shard in shards:
        if limit is not None and total >= limit:
            break
        try:
            text = shard.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"跳过读取失败 {shard}: {e}", file=sys.stderr)
            continue
        line_no = 0
        for line in text.splitlines():
            if limit is not None and total >= limit:
                break
            line_no += 1
            piece = line.strip()
            if len(piece) < 40:
                continue
            if skip_left > 0:
                skip_left -= 1
                continue
            # 唯一文件名：供 metadata / 溯源
            fname = f"{shard.name}#L{line_no}.txt"
            batch.append((piece.encode("utf-8"), fname))
            total += 1
            if len(batch) >= args.upload_batch:
                if flush():
                    return 0
                print(f"  已入库 {total} 篇…")

    if flush():
        return 0
    print(f"完成，共入库 {total} 篇。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
