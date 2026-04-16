#!/usr/bin/env python3
"""
批量评测：读取 CSV（question / answer / evidence），跑混合检索 + LangGraph 工作流，
输出每条 EM、字符 F1、证据命中、端到端耗时；默认另写整体汇总 JSON + Markdown。

用法（在项目根目录）:
  python scripts/run_eval.py --dataset datasets/qa_eval.csv --output datasets/eval_results.csv
  python scripts/run_eval.py --modes full vector vector_keyword --output datasets/eval_ablation.csv
  python scripts/run_eval.py --dataset datasets/crud_read_eval.csv --max-rows 20 --no-summary

依赖：与 app 相同（.env 中 POSTGRES_DSN、LLM 等）；知识库需已入库，否则检索为空。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Tuple

# 项目根
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# HF 镜像（与 app 一致）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.data_loader import build_or_load_graph
from src.eval_runner import (
    build_full_summary,
    load_rows,
    run_eval_batch,
    write_summary_json,
    write_summary_markdown,
)
from src.config import RETRIEVAL_GRAPH_MAX, RETRIEVAL_KEYWORD_TOP_K, RETRIEVAL_VECTOR_TOP_K
from src.vectorstore import load_vectorstore


def _summary_paths(summary_arg: Path | None, output: Path) -> Tuple[Path, Path]:
    if summary_arg is None:
        json_path = output.with_name(output.stem + "_summary.json")
    else:
        json_path = summary_arg
        if json_path.suffix.lower() not in (".json",):
            json_path = json_path.with_suffix(".json")
    return json_path, json_path.with_suffix(".md")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG+KG 批量评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "qa_eval.csv",
        help="CSV：需含 question、answer；可选 id、evidence、type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "eval_results.csv",
        help="逐条结果输出路径",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="整体指标 JSON；默认与 --output 同目录、文件名为 <stem>_summary.json",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="不写入汇总 JSON/Markdown",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full"],
        choices=["full", "vector", "vector_keyword", "vector_keyword_graph"],
        help="检索策略；vector_keyword_graph 与 full 等价，便于对照命名",
    )
    parser.add_argument(
        "--pred-style",
        choices=["full", "short"],
        default="short",
        help="预测答案写入/打分格式：full=原样；short=取首段并去引用（更适合字符F1）",
    )
    parser.add_argument(
        "--prompt-style",
        default="default",
        choices=["default", "crudrag"],
        help="评测用提示词模板：default=本项目 concise；crudrag=CRUD-RAG 官方 quest_answer 模板（不影响网页端）",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=RETRIEVAL_VECTOR_TOP_K,
        help="默认取自 .env RETRIEVAL_VECTOR_TOP_K",
    )
    parser.add_argument(
        "--keyword-top-k",
        type=int,
        default=RETRIEVAL_KEYWORD_TOP_K,
        help="默认取自 .env RETRIEVAL_KEYWORD_TOP_K",
    )
    parser.add_argument(
        "--graph-max",
        type=int,
        default=RETRIEVAL_GRAPH_MAX,
        help="默认取自 .env RETRIEVAL_GRAPH_MAX",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="只评测前 N 条（0 表示全量）",
    )
    args = parser.parse_args()
    if args.prompt_style == "crudrag":
        os.environ["EVAL_PROMPT_STYLE"] = "crudrag"

    if not args.dataset.is_file():
        print(f"找不到数据集: {args.dataset}", file=sys.stderr)
        return 1

    rows = load_rows(args.dataset)
    if not rows:
        print("数据集为空或无法解析", file=sys.stderr)
        return 1

    for key in ("question", "answer"):
        if key not in rows[0]:
            print(f"CSV 缺少列: {key}", file=sys.stderr)
            return 1

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    print("加载向量库与知识图谱…")
    vs = load_vectorstore()
    g = build_or_load_graph()
    if vs is None and (g is None or g.number_of_nodes() == 0):
        print("警告：向量库未加载且图谱为空，多数问题将无检索结果。", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch = run_eval_batch(
        rows,
        vs,
        g,
        list(args.modes),
        args.vector_top_k,
        args.keyword_top_k,
        args.graph_max,
        args.output,
        label="",
        pred_style=args.pred_style,
    )

    print(f"已写入: {args.output}")
    if not args.no_summary:
        sj, sm = _summary_paths(args.summary_json, args.output)
        full = build_full_summary(args.dataset, args.output, batch)
        write_summary_json(sj, full)
        write_summary_markdown(sm, full)
        print(f"已写入整体汇总: {sj}")
        print(f"已写入整体汇总(Markdown): {sm}")

    print(
        "\n指标说明：EM 要求预测与标准答案逐字一致，长答案+溯源格式下通常接近 0；"
        "字符 F1 反映表述重叠度。"
        "\nevidence_* 列为 NA 表示本行未填 evidence，不参与命中统计。"
        "\n有 evidence 时：strict 要求每一段都能在 chunk.source 中找到子串；any 为任一段匹配即可。\n"
    )
    for mode in args.modes:
        m = batch.get("per_mode", {}).get(mode)
        if not m:
            continue
        n = m["n"]
        print(f"\n=== mode={mode}  (n={n}) ===")
        print(f"  EM 均值:     {m['mean_em']:.4f}")
        print(f"  字符 F1 均值: {m['mean_char_f1']:.4f}")
        n_ev = m.get("n_with_evidence") or 0
        if n_ev > 0:
            print(
                f"  证据 strict 均值: {m['evidence_strict_mean']:.4f}  (仅统计含 evidence 的 {n_ev} 条)"
            )
            print(f"  证据 any 均值:   {m['evidence_any_mean']:.4f}")
        else:
            print("  证据 strict/any: 未统计（全部样本 evidence 为空，CSV 中列为 NA）")
        print(
            f"  延迟 P50:    {m['latency_p50_sec']:.4f}s   P95: {m['latency_p95_sec']:.4f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
