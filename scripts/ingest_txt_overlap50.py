#!/usr/bin/env python3
"""
使用 chunk_overlap=50 重新入库，保持上下文连续性。

根据优化建议，使用 chunk_overlap=50 可以：
1. 保持上下文连续性
2. 减少信息碎片化
3. 提升检索质量

用法（在项目根目录）:
  python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000

说明：
- 使用 chunk_size=128, chunk_overlap=50（CRUD-RAG 最佳实践）
- 若需与旧库隔离，先在 .env 设 PGVECTOR_COLLECTION=crud_eval_overlap50 并清空/新建库后再跑
- 第一次运行前，请先清空向量库：
  python scripts/clear_rag_kb.py
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

from src.data_loader import build_or_load_graph, process_uploaded_files
from src.vectorstore import load_vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 chunk_overlap=50 重新入库")
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
    parser.add_argument("--chunk-size", type=int, default=128, help="chunk 大小（CRUD-RAG 最佳实践：128）")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="chunk 重叠大小（优化建议：50）")
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
    print(f"优化配置：")
    print(f"  - chunk_size: {args.chunk_size}（CRUD-RAG 最佳实践）")
    print(f"  - chunk_overlap: {args.chunk_overlap}（保持上下文连续性）")
    print(f"\n提示：")
    print(f"  1. 若需清空旧库，请先运行：python scripts/clear_rag_kb.py")
    print(f"  2. 或在 .env 中设置新的 PGVECTOR_COLLECTION，如：crud_eval_overlap50")

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