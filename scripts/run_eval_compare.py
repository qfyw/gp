#!/usr/bin/env python3
"""
多组对比评测：同一数据集上依次跑若干预设，只加载一次向量库/图谱；
输出 compare_summary.json / .md，及每组独立 CSV。

--profile ablation（默认）: 混合/仅向量/向量+关键词 + 不同 k 的消融。
--profile topk: 专调 vector_top_k / keyword_top_k / graph_max 的 6 组对照
  （看 mean_char_f1 / mean_em；若要 BLEU/ROUGE 可对最优 slug 再跑
   scripts/run_crud_original_metrics.py --input <slug>.csv）。

用法:
  python scripts/run_eval_compare.py --dataset datasets/crud_read_eval.csv --max-rows 50 \\
      --profile topk --out-dir datasets/compare_topk_n50
  python scripts/run_eval_compare.py --dataset datasets/crud_read_eval.csv --out-dir datasets/compare_ablation
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.data_loader import build_or_load_graph
from src.eval_runner import (
    load_rows,
    run_eval_batch,
    write_compare_json,
    write_compare_markdown,
)
from src.vectorstore import load_vectorstore

# 每组: label 展示用, slug 文件名, mode, vector_top_k, keyword_top_k, graph_max
ABLATION_PRESETS: List[Dict[str, Any]] = [
    {
        "label": "G1_混合检索_k5",
        "slug": "G1_full_k5",
        "mode": "full",
        "vector_top_k": 5,
        "keyword_top_k": 5,
        "graph_max": 5,
    },
    {
        "label": "G2_仅向量_k5",
        "slug": "G2_vector_k5",
        "mode": "vector",
        "vector_top_k": 5,
        "keyword_top_k": 5,
        "graph_max": 5,
    },
    {
        "label": "G3_向量+关键词_k5",
        "slug": "G3_vector_keyword_k5",
        "mode": "vector_keyword",
        "vector_top_k": 5,
        "keyword_top_k": 5,
        "graph_max": 5,
    },
    {
        "label": "G4_混合检索_k8",
        "slug": "G4_full_k8",
        "mode": "full",
        "vector_top_k": 8,
        "keyword_top_k": 8,
        "graph_max": 8,
    },
    {
        "label": "G5_混合检索_k3",
        "slug": "G5_full_k3",
        "mode": "full",
        "vector_top_k": 3,
        "keyword_top_k": 3,
        "graph_max": 3,
    },
]

# top-k 调参对照：统一 full，对比「全小 / 均衡 / 全大 / 向量多 / 关键词多 / 图谱多」
TOPK_TUNING_PRESETS: List[Dict[str, Any]] = [
    {
        "label": "T1_全紧凑_v3k3g3",
        "slug": "topk_v3_k3_g3",
        "mode": "full",
        "vector_top_k": 3,
        "keyword_top_k": 3,
        "graph_max": 3,
    },
    {
        "label": "T2_均衡基线_v5k5g5",
        "slug": "topk_v5_k5_g5",
        "mode": "full",
        "vector_top_k": 5,
        "keyword_top_k": 5,
        "graph_max": 5,
    },
    {
        "label": "T3_全放宽_v8k8g8",
        "slug": "topk_v8_k8_g8",
        "mode": "full",
        "vector_top_k": 8,
        "keyword_top_k": 8,
        "graph_max": 8,
    },
    {
        "label": "T4_向量偏多_v12k5g3",
        "slug": "topk_v12_k5_g3",
        "mode": "full",
        "vector_top_k": 12,
        "keyword_top_k": 5,
        "graph_max": 3,
    },
    {
        "label": "T5_关键词偏多_v6k10g4",
        "slug": "topk_v6_k10_g4",
        "mode": "full",
        "vector_top_k": 6,
        "keyword_top_k": 10,
        "graph_max": 4,
    },
    {
        "label": "T6_图谱偏多_v5k5g10",
        "slug": "topk_v5_k5_g10",
        "mode": "full",
        "vector_top_k": 5,
        "keyword_top_k": 5,
        "graph_max": 10,
    },
]


def main() -> int:
    p = argparse.ArgumentParser(description="多组检索配置对比评测")
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "crud_read_eval.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "datasets" / "eval_compare",
        help="汇总与逐组 CSV 输出目录",
    )
    p.add_argument("--max-rows", type=int, default=0, help="只评前 N 条，0=全量")
    p.add_argument(
        "--pred-style",
        choices=["full", "short"],
        default="short",
        help="预测答案写入/打分格式：full=原样；short=取首段并去引用（更适合字符F1）",
    )
    p.add_argument(
        "--merge-csv",
        action="store_true",
        help="额外写入 all_runs_merged.csv（含 run_label 列）",
    )
    p.add_argument(
        "--profile",
        choices=("ablation", "topk"),
        default="ablation",
        help="ablation=模式消融(G1-G5); topk=6组 top-k 对照(仅调 full 三路 k)",
    )
    args = p.parse_args()
    presets: List[Dict[str, Any]] = (
        TOPK_TUNING_PRESETS if args.profile == "topk" else ABLATION_PRESETS
    )

    if not args.dataset.is_file():
        print(f"找不到数据集: {args.dataset}", file=sys.stderr)
        return 1

    rows = load_rows(args.dataset)
    if not rows:
        print("数据集为空", file=sys.stderr)
        return 1
    for key in ("question", "answer"):
        if key not in rows[0]:
            print(f"CSV 缺少列: {key}", file=sys.stderr)
            return 1

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    print("加载向量库与知识图谱（仅一次）…")
    vs = load_vectorstore()
    g = build_or_load_graph()

    import csv as csv_mod

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / "all_runs_merged.csv"
    merged_f = (
        merged_path.open("w", encoding="utf-8-sig", newline="") if args.merge_csv else None
    )
    merged_writer: Optional[csv_mod.DictWriter] = None

    results: List[Dict[str, Any]] = []

    try:
        for cfg in presets:
            label = cfg["label"]
            slug = cfg.get("slug") or label
            mode = cfg["mode"]
            vk, kk, gm = cfg["vector_top_k"], cfg["keyword_top_k"], cfg["graph_max"]
            per_csv = args.out_dir / f"{slug}.csv"
            print(f"\n>>> {label}  ({mode}, vk={vk}, kw={kk}, g={gm})")
            batch = run_eval_batch(
                rows,
                vs,
                g,
                [mode],
                vk,
                kk,
                gm,
                per_csv,
                label=label,
                pred_style=args.pred_style,
            )
            results.append(batch)
            m = batch["per_mode"].get(mode, {})
            print(
                f"    mean_char_f1={m.get('mean_char_f1')}  mean_em={m.get('mean_em')}  "
                f"P50={m.get('latency_p50_sec')}s"
            )

            if merged_f is not None:
                with per_csv.open(encoding="utf-8-sig", newline="") as fin:
                    reader = csv_mod.DictReader(fin)
                    if merged_writer is None:
                        merged_writer = csv_mod.DictWriter(merged_f, fieldnames=reader.fieldnames)
                        merged_writer.writeheader()
                    for row in reader:
                        merged_writer.writerow(row)
    finally:
        if merged_f is not None:
            merged_f.close()

    payload = {
        "dataset": str(args.dataset.resolve()),
        "n_rows": len(rows),
        "profile": args.profile,
        "presets": presets,
        "runs": results,
    }
    write_compare_json(args.out_dir / "compare_summary.json", payload)
    write_compare_markdown(args.out_dir / "compare_summary.md", results)
    print(f"\n已写入: {args.out_dir / 'compare_summary.json'}")
    print(f"已写入: {args.out_dir / 'compare_summary.md'}")
    if args.merge_csv:
        print(f"已写入: {merged_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
