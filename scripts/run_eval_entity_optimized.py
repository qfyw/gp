#!/usr/bin/env python3
"""
使用优化版实体感知检索进行评测。
主要改进：
1. 优化实体匹配（模糊匹配、降低过滤强度）
2. 增大实体权重
3. 使用CRUD-RAG最佳实践参数

用法：
  python scripts/run_eval_entity_optimized.py --dataset datasets/crud_read_eval.csv --max-rows 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# HF 镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 优化版配置
os.environ["EVAL_PROMPT_STYLE"] = "crud_optimized_v3"
os.environ["KB_STRICT_ONLY"] = "false"
os.environ["INTERNAL_DOC_ONLY_ANSWER"] = "false"
os.environ["RETRIEVAL_VECTOR_TOP_K"] = "8"
os.environ["RETRIEVAL_KEYWORD_TOP_K"] = "8"
os.environ["RETRIEVAL_GRAPH_MAX"] = "8"
os.environ["RERANK_RECALL_MULT"] = "3"
os.environ["RRF_K"] = "60"
os.environ["OPENAI_TEMPERATURE"] = "0.1"
os.environ["USE_ENTITY_AWARE_RETRIEVAL"] = "true"  # 启用实体感知检索
os.environ["MIN_ENTITY_MATCHES"] = "0"  # 不强制过滤

from src.config import (
    RETRIEVAL_GRAPH_MAX,
    RETRIEVAL_KEYWORD_TOP_K,
    RETRIEVAL_VECTOR_TOP_K,
)
from src.data_loader import build_or_load_graph
from src.eval_runner import (
    build_full_summary,
    load_rows,
    write_summary_json,
    write_summary_markdown,
)
from src.vectorstore import load_vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="使用优化版实体感知检索进行评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "crud_read_eval.csv",
        help="CSV：需含 question、answer；可选 id、evidence、type",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="只评测前 N 条（0 表示全量）",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=int(os.getenv("RETRIEVAL_VECTOR_TOP_K", "8")),
        help="向量检索 top-k",
    )
    parser.add_argument(
        "--keyword-top-k",
        type=int,
        default=int(os.getenv("RETRIEVAL_KEYWORD_TOP_K", "8")),
        help="关键词检索 top-k",
    )
    parser.add_argument(
        "--graph-max",
        type=int,
        default=int(os.getenv("RETRIEVAL_GRAPH_MAX", "8")),
        help="图谱检索最大数量",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM 温度参数",
    )
    parser.add_argument(
        "--min-entity-matches",
        type=int,
        default=int(os.getenv("MIN_ENTITY_MATCHES", "0")),
        help="最小实体匹配数（0=不强制过滤）",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"找不到数据集: {args.dataset}", file=sys.stderr)
        return 1

    rows = load_rows(args.dataset)
    if not rows:
        print("数据集为空或无法解析", file=sys.stderr)
        return 1

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    print(f"加载向量库与知识图谱…")
    print(f"优化版实体感知检索配置：")
    print(f"  - chunk_size: 128（CRUD-RAG 最佳实践）")
    print(f"  - chunk_overlap: 50（保持上下文连续性）")
    print(f"  - vector_top_k: {args.vector_top_k}")
    print(f"  - keyword_top_k: {args.keyword_top_k}")
    print(f"  - graph_max: {args.graph_max}")
    print(f"  - RERANK_RECALL_MULT: 3")
    print(f"  - RRF_K: 60（CRUD-RAG 默认）")
    print(f"  - temperature: {args.temperature}（CRUD-RAG 默认）")
    print(f"  - KB_STRICT_ONLY: false（宽松模式）")
    print(f"  - 提示词: crud_optimized_v3（增强完整回答要求）")
    print(f"  - 实体感知检索: true")
    print(f"  - min_entity_matches: {args.min_entity_matches}（不强制过滤）")

    vs = load_vectorstore()
    g = build_or_load_graph()
    if vs is None and (g is None or g.number_of_nodes() == 0):
        print("警告：向量库未加载且图谱为空，多数问题将无检索结果。", file=sys.stderr)

    # 输出路径
    timestamp = int(os.path.getmtime(__file__))
    output = ROOT / "datasets" / f"eval_entity_optimized_n{len(rows)}_{timestamp}"
    output_csv = output.with_suffix(".csv")
    output_summary_json = output.with_name(output.name + "_summary.json")
    output_summary_md = output.with_name(output.name + "_summary.md")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 批量评测
    print(f"\n开始评测 {len(rows)} 条问题（优化版实体感知检索）...")
    from src.eval_runner import run_eval_batch

    batch = run_eval_batch(
        rows,
        vs,
        g,
        ["full"],
        args.vector_top_k,
        args.keyword_top_k,
        args.graph_max,
        output_csv,
        label="entity_optimized",
        pred_style="short",
    )

    print(f"已写入: {output_csv}")

    # 生成汇总
    full = build_full_summary(args.dataset, output_csv, batch)
    write_summary_json(output_summary_json, full)
    write_summary_markdown(output_summary_md, full)

    print(f"已写入整体汇总: {output_summary_json}")
    print(f"已写入整体汇总(Markdown): {output_summary_md}")

    # 打印关键指标
    for mode in ["full"]:
        m = batch.get("per_mode", {}).get(mode)
        if not m:
            continue
        n = m["n"]
        print(f"\n=== mode={mode}  (n={n}) ===")
        print(f"  EM 均值:     {m['mean_em']:.4f}")
        print(f"  字符 F1 均值: {m['mean_char_f1']:.4f}")
        print(f"  延迟 P50:    {m['latency_p50_sec']:.4f}s   P95: {m['latency_p95_sec']:.4f}s")

    # 提示下一步
    print(f"\n下一步：运行 LLM 评测以获取语义准确率")
    llm_output = output_csv.with_name(output_csv.stem + "_llm_judged.csv")
    print(f"  python scripts/llm_eval_judge.py --input {output_csv} --output {llm_output} --sleep 0.2")

    # 提示分析错误
    print(f"\n错误分析：")
    print(f"  python scripts/analyze_eval_errors.py {output_csv} --out {output_csv.stem}_error_report.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())