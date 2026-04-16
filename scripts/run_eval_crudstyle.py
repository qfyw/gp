#!/usr/bin/env python3
"""
一键：生成本项目预测结果 + 计算 CRUD_RAG 原版指标（BLEU / ROUGE-L / BERTScore(text2vec)）。

输出文件：
  1) <output>.csv                       逐条预测（来自 run_eval_batch）
  2) <output>_summary.json / .md        本项目汇总（EM / char_F1 / 延迟）
  3) <output>_crudstyle.json            CRUD 风格指标汇总（avg_bleu / avg_rougeL / avg_bertScore_text2vec）
  4) <output>_combined.json             合并汇总（方便论文表格/画图）

示例：
  python scripts/run_eval_crudstyle.py --dataset datasets/crud_read_eval.csv --max-rows 30 --output datasets/n30_v8k5g2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# HF 镜像（尽量避免连 huggingface.co）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 有些环境 SSL 会拦截，评测阶段可临时关闭（仅本地开发/毕业设计环境）
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

from src.config import RETRIEVAL_GRAPH_MAX, RETRIEVAL_KEYWORD_TOP_K, RETRIEVAL_VECTOR_TOP_K
from src.data_loader import build_or_load_graph
from src.eval_runner import (
    build_full_summary,
    load_rows,
    run_eval_batch,
    write_summary_json,
    write_summary_markdown,
)
from src.vectorstore import load_vectorstore


def _jieba_cut(text: str) -> List[str]:
    import jieba

    return list(jieba.cut(text or ""))


def _load_bleu_rouge() -> tuple[Any, Any]:
    import evaluate

    # 优先使用 CRUD_RAG 自带离线脚本
    local_bleu = ROOT / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "bleu"
    local_rouge = ROOT / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "rouge"
    bleu = evaluate.load(str(local_bleu)) if local_bleu.is_dir() else evaluate.load("bleu")
    rouge = (
        evaluate.load(str(local_rouge)) if local_rouge.is_dir() else evaluate.load("rouge")
    )
    return bleu, rouge


def compute_crudstyle_metrics_from_eval_csv(path: Path, *, with_bert: bool) -> Dict[str, Any]:
    import csv

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"input": str(path.resolve()), "n": 0}

    bleu, rouge = _load_bleu_rouge()

    bleu_list: List[float] = []
    rouge_list: List[float] = []
    bert_list: List[float] = []

    sim = None
    if with_bert:
        try:
            from text2vec import Similarity

            sim = Similarity()
        except Exception:
            sim = None

    for r in rows:
        ref = (r.get("gold_answer") or "").strip()
        pred = (r.get("pred_answer") or "").strip()
        if not ref:
            continue

        b = bleu.compute(predictions=[pred], references=[[ref]], tokenizer=_jieba_cut)
        bleu_list.append(float(b.get("bleu") or 0.0))

        rg = rouge.compute(
            predictions=[pred],
            references=[[ref]],
            tokenizer=_jieba_cut,
            rouge_types=["rougeL"],
        )
        rouge_list.append(float(rg.get("rougeL") or 0.0))

        if sim is not None:
            try:
                bert_list.append(float(sim.get_score(pred, ref) or 0.0))
            except Exception:
                bert_list.append(0.0)

    n = len(bleu_list)
    out: Dict[str, Any] = {
        "input": str(path.resolve()),
        "n": n,
        "avg_bleu": (sum(bleu_list) / n) if n else 0.0,
        "avg_rougeL": (sum(rouge_list) / n) if n else 0.0,
    }
    if with_bert:
        out["avg_bertScore_text2vec"] = (sum(bert_list) / n) if (n and bert_list) else 0.0
    else:
        out["avg_bertScore_text2vec"] = None
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="一键评测 + CRUD 原版指标")
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "crud_read_eval.csv")
    p.add_argument("--max-rows", type=int, default=30, help="只跑前 N 条，0=全量")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出前缀（不带扩展名），例如 datasets/n30_v8k5g2",
    )
    p.add_argument(
        "--mode",
        default="full",
        choices=["full", "vector", "vector_keyword", "vector_keyword_graph"],
        help="检索模式（单个）",
    )
    p.add_argument("--vector-top-k", type=int, default=RETRIEVAL_VECTOR_TOP_K)
    p.add_argument("--keyword-top-k", type=int, default=RETRIEVAL_KEYWORD_TOP_K)
    p.add_argument("--graph-max", type=int, default=RETRIEVAL_GRAPH_MAX)
    p.add_argument("--pred-style", choices=["full", "short"], default="short")
    p.add_argument(
        "--prompt-style",
        default="default",
        choices=["default", "crudrag"],
        help="评测用提示词模板：default=本项目 concise；crudrag=CRUD-RAG 官方 quest_answer 模板（不影响网页端）",
    )
    p.add_argument(
        "--no-bert",
        action="store_true",
        help="不算 text2vec-BERTScore（避免下载模型/联网问题）",
    )
    args = p.parse_args()
    if args.prompt_style == "crudrag":
        os.environ["EVAL_PROMPT_STYLE"] = "crudrag"

    if not args.dataset.is_file():
        print(f"找不到数据集: {args.dataset}", file=sys.stderr)
        return 1

    rows = load_rows(args.dataset)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        print("数据集为空", file=sys.stderr)
        return 1

    # 输出路径
    out_csv = args.output.with_suffix(".csv")
    out_summary_json = args.output.with_name(args.output.name + "_summary.json")
    out_summary_md = args.output.with_name(args.output.name + "_summary.md")
    out_crud_json = args.output.with_name(args.output.name + "_crudstyle.json")
    out_combined = args.output.with_name(args.output.name + "_combined.json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("加载向量库与知识图谱…")
    vs = load_vectorstore()
    g = build_or_load_graph()

    batch = run_eval_batch(
        rows,
        vs,
        g,
        [args.mode],
        args.vector_top_k,
        args.keyword_top_k,
        args.graph_max,
        out_csv,
        label="",
        pred_style=args.pred_style,
    )

    full = build_full_summary(args.dataset, out_csv, batch)
    write_summary_json(out_summary_json, full)
    write_summary_markdown(out_summary_md, full)

    crud = compute_crudstyle_metrics_from_eval_csv(out_csv, with_bert=not args.no_bert)
    with out_crud_json.open("w", encoding="utf-8") as f:
        json.dump(crud, f, ensure_ascii=False, indent=2)

    combined = {"eval_summary": full, "crudstyle": crud}
    with out_combined.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"已写入: {out_csv}")
    print(f"已写入: {out_summary_md}")
    print(f"已写入: {out_crud_json}")
    print(f"已写入: {out_combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

