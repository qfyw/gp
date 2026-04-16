#!/usr/bin/env python3
"""
将目录下 .txt 分批写入当前项目的向量库 + 关键词表 + 图谱（与 Streamlit 上传逻辑一致）。

用于 CRUD-RAG 的 data/80000_docs 等纯文本语料。大批量时请用小 batch、多次运行。

用法（在项目根目录）:
  python scripts/ingest_txt_dir.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000

说明：
- 每批调用 process_uploaded_files，避免一次性加载数万文件进内存。
- 若需与旧库隔离，先在 .env 设 PGVECTOR_COLLECTION=crud_rag_eval 并清空/新建库后再跑。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not __import__("os").environ.get("HF_ENDPOINT"):
    __import__("os").environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.config import INGEST_CHUNK_OVERLAP, INGEST_CHUNK_SIZE
from src.data_loader import build_or_load_graph, process_uploaded_files
from src.vectorstore import load_vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="批量入库 .txt 目录")
    parser.add_argument("--dir", type=Path, required=True, help="含 .txt 的目录（不递归子目录则加 --no-recursive）")
    parser.add_argument("--batch-size", type=int, default=100, help="每批文件数")
    parser.add_argument("--skip", type=int, default=0, help="跳过排序后的前 N 个文件（用于分阶段增量入库）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理文件数，0 表示不限制")
    parser.add_argument("--recursive", action="store_true", help="递归子目录")
    parser.add_argument("--num-shards", type=int, default=1, help="把文件列表分成 N 份（默认 1，不分片）")
    parser.add_argument("--shard-idx", type=int, default=0, help="当前处理第几份（从 0 开始）")
    parser.add_argument(
        "--shard-mode",
        choices=("contiguous", "roundrobin"),
        default="contiguous",
        help="分片方式：contiguous=按排序后连续切片；roundrobin=按 i%%N 轮询分配",
    )
    parser.add_argument("--chunk-size", type=int, default=INGEST_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=INGEST_CHUNK_OVERLAP)
    args = parser.parse_args()

    root: Path = args.dir
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 1

    pattern = "**/*.txt" if args.recursive else "*.txt"
    files = sorted(root.glob(pattern))

    if args.skip and args.skip > 0:
        files = files[args.skip :]

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if args.num_shards < 1:
        print("--num-shards 必须 >= 1", file=sys.stderr)
        return 1
    if not (0 <= args.shard_idx < args.num_shards):
        print("--shard-idx 必须满足 0 <= shard-idx < num-shards", file=sys.stderr)
        return 1

    if args.num_shards > 1:
        if args.shard_mode == "roundrobin":
            files = [p for i, p in enumerate(files) if i % args.num_shards == args.shard_idx]
        else:
            # contiguous: 稳定按排序后切成 N 份（前面的份可能多 1 个）
            n = len(files)
            base = n // args.num_shards
            extra = n % args.num_shards
            start = args.shard_idx * base + min(args.shard_idx, extra)
            end = start + base + (1 if args.shard_idx < extra else 0)
            files = files[start:end]

    if not files:
        print("未找到 .txt 文件", file=sys.stderr)
        return 1

    shard_info = (
        f" | shard {args.shard_idx+1}/{args.num_shards} ({args.shard_mode})" if args.num_shards > 1 else ""
    )
    print(f"共 {len(files)} 个文件，batch_size={args.batch_size}{shard_info}")
    vs = load_vectorstore()
    g = build_or_load_graph()

    batch: list[tuple[bytes, str]] = []
    n_done = 0
    n_chunks_total = 0

    def flush() -> None:
        nonlocal vs, g, batch, n_done, n_chunks_total
        if not batch:
            return
        batch_files = len(batch)
        vs, g, ingest = process_uploaded_files(
            batch,
            existing_vectorstore=vs,
            existing_graph=g,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        n_done += batch_files
        n_chunks_total += int(getattr(ingest, "chunk_count", 0) or 0)
        print(
            f"  本批: 文件 {batch_files} 个, 新增 chunk {getattr(ingest, 'chunk_count', 0)} 条 | "
            f"累计: 文件 {n_done}/{len(files)} 个, chunk {n_chunks_total} 条"
        )
        batch = []

    for p in files:
        try:
            data = p.read_bytes()
        except OSError as e:
            print(f"跳过读取失败: {p} ({e})", file=sys.stderr)
            continue
        # 与 CRUD 官方一致：用文件名作为文档标识（metadata filename / source）
        batch.append((data, p.name))
        if len(batch) >= args.batch_size:
            flush()

    flush()
    print(f"全部完成。累计入库文件 {n_done} 个，累计新增 chunk {n_chunks_total} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
