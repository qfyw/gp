#!/usr/bin/env python3
"""
多阶段检索评测：实体感知 + 多阶段检索优化
预期准确率：50% → 60-65%

优化策略：
1. 实体感知：提取问题中的关键实体，只检索包含这些实体的文档
2. 多阶段检索：
   - 阶段1：扩大召回（3倍）
   - 阶段2：实体过滤（只保留包含关键实体的文档）
   - 阶段3：精确重排序
3. CRUD-RAG 最佳实践：chunk_size=128, top_k=8, 温度=0.1

用法：
  python scripts/run_eval_multi_stage.py --dataset datasets/crud_read_eval.csv --max-rows 30
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

# 最佳实践配置
os.environ["EVAL_PROMPT_STYLE"] = "crud_optimized"
os.environ["KB_STRICT_ONLY"] = "false"
os.environ["INTERNAL_DOC_ONLY_ANSWER"] = "false"
os.environ["RETRIEVAL_VECTOR_TOP_K"] = "8"
os.environ["RETRIEVAL_KEYWORD_TOP_K"] = "8"
os.environ["RETRIEVAL_GRAPH_MAX"] = "8"
os.environ["OPENAI_TEMPERATURE"] = "0.1"

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
    parser = argparse.ArgumentParser(description="多阶段检索评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "crud_read_eval.csv",
        help="CSV：需含 question、answer；可选 id、evidence、type",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
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
        "--stage1-recall-mult",
        type=int,
        default=3,
        help="阶段1 召回倍数（默认3）",
    )
    parser.add_argument(
        "--min-entity-matches",
        type=int,
        default=1,
        help="最小实体匹配数（默认1）",
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
    print(f"多阶段检索配置：")
    print(f"  - chunk_size: 128（CRUD_RAG 最佳实践）")
    print(f"  - vector_top_k: {args.vector_top_k}（最终返回）")
    print(f"  - keyword_top_k: {args.keyword_top_k}（最终返回）")
    print(f"  - graph_max: {args.graph_max}")
    print(f"  - 阶段1召回倍数: {args.stage1_recall_mult}（扩大召回）")
    print(f"  - 最小实体匹配数: {args.min_entity_matches}")
    print(f"  - 温度: 0.1（CRUD_RAG 最佳实践）")
    print(f"  - 提示词: crud_optimized（角色设定 + 结构化输出）")

    vs = load_vectorstore()
    g = build_or_load_graph()
    if vs is None and (g is None or g.number_of_nodes() == 0):
        print("警告：向量库未加载且图谱为空，多数问题将无检索结果。", file=sys.stderr)

    # 输出路径
    timestamp = int(os.path.getmtime(__file__))
    output = ROOT / "datasets" / f"eval_multi_stage_n{len(rows)}_{timestamp}"
    output_csv = output.with_suffix(".csv")
    output_summary_json = output.with_name(output.name + "_summary.json")
    output_summary_md = output.with_name(output.name + "_summary.md")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 批量评测（使用多阶段检索）
    print(f"\n开始评测 {len(rows)} 条问题（多阶段检索）...")

    import csv
    import time
    from datetime import datetime, timezone

    from src.agents.workflow import run_advanced_workflow
    from src.eval_metrics import char_level_f1, exact_match
    from src.multi_stage_retriever import multi_stage_hybrid_retrieve
    from src.retriever import RetrievedChunk

    fieldnames = [
        "run_label",
        "mode",
        "id",
        "type",
        "question",
        "gold_answer",
        "pred_answer",
        "pred_answer_raw",
        "em",
        "char_f1",
        "evidence_strict_hit",
        "evidence_any_hit",
        "retrieve_sec",
        "llm_sec",
        "latency_sec",
        "error",
        "stage1_count",
        "stage2_count",
        "stage3_count",
        "entity_counts",
    ]

    all_latencies: List[float] = []
    all_f1: List[float] = []
    all_em: List[float] = []

    with output_csv.open("w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            q = row.get("question", "")
            gold = row.get("answer", "")
            ev = row.get("evidence", "")
            rid = row.get("id", "")
            typ = row.get("type", "")

            err = ""
            pred = ""
            lat = 0.0
            retrieve_sec = 0.0
            llm_sec = 0.0
            stage1_count = 0
            stage2_count = 0
            stage3_count = 0
            entity_counts = {}

            t0 = time.perf_counter()
            try:
                # 多阶段检索
                t_retrieve0 = time.perf_counter()
                chunks = multi_stage_hybrid_retrieve(
                    q,
                    vs,
                    g,
                    args.vector_top_k,
                    args.keyword_top_k,
                    args.graph_max,
                    stage1_recall_mult=args.stage1_recall_mult,
                    min_entity_matches=args.min_entity_matches,
                )
                retrieve_sec = time.perf_counter() - t_retrieve0
                stage1_count = len(chunks)

                if chunks:
                    t_llm0 = time.perf_counter()
                    state = run_advanced_workflow(q, chunks, graph=g, answer_style="crud_optimized")
                    pred = state.get("final_answer") or ""
                    llm_sec = time.perf_counter() - t_llm0
                    stage2_count = len(state.get("internal_docs", []))
                    stage3_count = len(state.get("kg_triples", []))

                lat = time.perf_counter() - t0
            except Exception as e:
                err = str(e)
                lat = time.perf_counter() - t0

            # 格式化预测答案
            pred_raw = pred
            pred = pred_raw.strip()

            # 计算指标
            em = 1.0 if exact_match(pred, gold) else 0.0
            f1 = char_level_f1(pred, gold)

            # 证据命中（如果有）
            has_evidence = bool((ev or "").strip())
            if has_evidence:
                srcs = [c.source for c in chunks]
                from src.eval_metrics import evidence_hit_sources, evidence_hit_sources_any

                ev_ok, _ = evidence_hit_sources(ev, srcs)
                ev_hit = 1.0 if ev_ok else 0.0
                ev_any_ok, _ = evidence_hit_sources_any(ev, srcs)
                ev_any = 1.0 if ev_any_ok else 0.0
                ev_s_str = f"{ev_hit:.4f}"
                ev_a_str = f"{ev_any:.4f}"
            else:
                ev_s_str = "NA"
                ev_a_str = "NA"

            # 写入 CSV
            writer.writerow(
                {
                    "run_label": "multi_stage",
                    "mode": "full",
                    "id": rid,
                    "type": typ,
                    "question": q,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "pred_answer_raw": pred_raw,
                    "em": f"{em:.4f}",
                    "char_f1": f"{f1:.4f}",
                    "evidence_strict_hit": ev_s_str,
                    "evidence_any_hit": ev_a_str,
                    "retrieve_sec": f"{retrieve_sec:.4f}",
                    "llm_sec": f"{llm_sec:.4f}",
                    "latency_sec": f"{lat:.4f}",
                    "error": err,
                    "stage1_count": stage1_count,
                    "stage2_count": stage2_count,
                    "stage3_count": stage3_count,
                    "entity_counts": str(entity_counts),
                }
            )

            all_latencies.append(lat)
            all_f1.append(f1)
            all_em.append(em)

    print(f"已写入: {output_csv}")

    # 生成汇总
    n = len(all_f1)
    import numpy as np

    summary = {
        "input": str(args.dataset.resolve()),
        "output": str(output_csv.resolve()),
        "n": n,
        "mean_em": np.mean(all_em) if all_em else 0.0,
        "mean_char_f1": np.mean(all_f1) if all_f1 else 0.0,
        "latency_p50_sec": float(np.percentile(all_latencies, 50)) if all_latencies else 0.0,
        "latency_p95_sec": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
        "config": {
            "vector_top_k": args.vector_top_k,
            "keyword_top_k": args.keyword_top_k,
            "graph_max": args.graph_max,
            "stage1_recall_mult": args.stage1_recall_mult,
            "min_entity_matches": args.min_entity_matches,
            "temperature": 0.1,
            "prompt_style": "crud_optimized",
        },
    }

    import json

    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入整体汇总: {output_summary_json}")

    # Markdown 报告
    markdown = f"""# 多阶段检索评测汇总

## 配置
- vector_top_k: {args.vector_top_k}
- keyword_top_k: {args.keyword_top_k}
- graph_max: {args.graph_max}
- 阶段1召回倍数: {args.stage1_recall_mult}
- 最小实体匹配数: {args.min_entity_matches}
- 温度: 0.1
- 提示词: crud_optimized

## 指标
- 样本数: {n}
- EM 均值: {summary['mean_em']:.4f}
- 字符 F1 均值: {summary['mean_char_f1']:.4f}
- 延迟 P50: {summary['latency_p50_sec']:.4f}s
- 延迟 P95: {summary['latency_p95_sec']:.4f}s

## 下一步
运行 LLM 语义评测：
```bash
python scripts/llm_eval_judge.py --input {output_csv} --output {output_csv.with_name(output_csv.stem + '_llm_judged.csv')} --sleep 0.2
```
"""
    output_summary_md.write_text(markdown, encoding="utf-8")
    print(f"已写入整体汇总(Markdown): {output_summary_md}")

    # 打印关键指标
    print(f"\n=== 多阶段检索评测结果 (n={n}) ===")
    print(f"  EM 均值:     {summary['mean_em']:.4f}")
    print(f"  字符 F1 均值: {summary['mean_char_f1']:.4f}")
    print(f"  延迟 P50:    {summary['latency_p50_sec']:.4f}s")
    print(f"  延迟 P95:    {summary['latency_p95_sec']:.4f}s")

    # 提示下一步
    print(f"\n下一步：运行 LLM 评测以获取语义准确率")
    llm_output = output_csv.with_name(output_csv.stem + "_llm_judged.csv")
    print(f"  python scripts/llm_eval_judge.py --input {output_csv} --output {llm_output} --sleep 0.2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())