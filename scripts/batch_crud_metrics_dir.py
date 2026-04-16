#!/usr/bin/env python3
"""
对目录下多份 run_eval 输出 CSV 批量计算 CRUD 论文常用三项均值：
  avg_bleu, avg_rougeL, avg_bertScore_text2vec
（jieba 分词 + huggingface evaluate；text2vec 只加载一次）

示例:
  python scripts/batch_crud_metrics_dir.py --dir datasets/compare_topk_n50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parent.parent


def _jieba_cut(text: str) -> List[str]:
    import jieba

    return list(jieba.cut(text or ""))


def _load_bleu_rouge() -> tuple[Any, Any]:
    import evaluate

    local_bleu = ROOT / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "bleu"
    local_rouge = ROOT / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "rouge"
    bleu = evaluate.load(str(local_bleu)) if local_bleu.is_dir() else evaluate.load("bleu")
    rouge = (
        evaluate.load(str(local_rouge)) if local_rouge.is_dir() else evaluate.load("rouge")
    )
    return bleu, rouge


def _one_bleu(bleu: Any, pred: str, ref: str) -> float:
    res = bleu.compute(predictions=[pred], references=[[ref]], tokenizer=_jieba_cut)
    return float(res.get("bleu") or 0.0)


def _one_rouge(rouge: Any, pred: str, ref: str) -> float:
    res = rouge.compute(
        predictions=[pred],
        references=[[ref]],
        tokenizer=_jieba_cut,
        rouge_types=["rougeL"],
    )
    return float(res.get("rougeL") or 0.0)


def compute_file(
    path: Path,
    *,
    bleu: Any,
    rouge: Any,
    sim: Any | None,
) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for k in ("gold_answer", "pred_answer"):
            if k not in fieldnames:
                raise ValueError(f"{path.name} 缺少列: {k}")
        rows = list(reader)

    bleu_list: list[float] = []
    rouge_list: list[float] = []
    bert_list: list[float] = []

    for r in rows:
        ref = (r.get("gold_answer") or "").strip()
        pred = (r.get("pred_answer") or "").strip()
        if not ref:
            continue
        bleu_list.append(_one_bleu(bleu, pred, ref))
        rouge_list.append(_one_rouge(rouge, pred, ref))
        if sim is not None:
            try:
                bert_list.append(float(sim.get_score(pred, ref) or 0.0))
            except Exception:
                bert_list.append(0.0)

    n = len(bleu_list)
    out: dict[str, Any] = {
        "slug": path.stem,
        "input": str(path.resolve()),
        "n": n,
        "avg_bleu": (sum(bleu_list) / n) if n else 0.0,
        "avg_rougeL": (sum(rouge_list) / n) if n else 0.0,
    }
    if sim is not None:
        out["avg_bertScore_text2vec"] = (sum(bert_list) / n) if (n and bert_list) else 0.0
    else:
        out["avg_bertScore_text2vec"] = None
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="目录内多 CSV 批量 CRUD 三项指标")
    p.add_argument("--dir", type=Path, required=True, help="含 topk_*.csv 等的目录")
    p.add_argument(
        "--glob",
        default="topk_*.csv",
        help="匹配文件名，默认只评各 topk 单跑 CSV（不含 all_runs_merged）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="汇总 JSON；默认 <dir>/crud_paper_metrics_batch.json",
    )
    p.add_argument(
        "--no-bert",
        action="store_true",
        help="不算 text2vec（更快）",
    )
    args = p.parse_args()

    if not args.dir.is_dir():
        print(f"目录不存在: {args.dir}", file=sys.stderr)
        return 1

    files = sorted(args.dir.glob(args.glob))
    if not files:
        print(f"未匹配到文件: {args.dir}/{args.glob}", file=sys.stderr)
        return 1

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

    print("加载 BLEU / ROUGE …", flush=True)
    bleu, rouge = _load_bleu_rouge()
    sim = None
    if not args.no_bert:
        print("加载 text2vec Similarity（仅一次）…", flush=True)
        from text2vec import Similarity

        sim = Similarity()

    runs: list[dict[str, Any]] = []
    for fp in files:
        print(f"  {fp.name} …", flush=True)
        runs.append(compute_file(fp, bleu=bleu, rouge=rouge, sim=sim))

    out_path = args.output or (args.dir / "crud_paper_metrics_batch.json")
    payload = {"dir": str(args.dir.resolve()), "glob": args.glob, "runs": runs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已写入: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
