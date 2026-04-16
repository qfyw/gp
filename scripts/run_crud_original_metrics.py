#!/usr/bin/env python3
"""
用 CRUD_RAG 论文/官方常用的指标（BLEU / ROUGE-L / BERTScore(text2vec)）对本项目评测输出再打分。

输入：本项目 `scripts/run_eval.py` 生成的 CSV（需含 gold_answer、pred_answer）。
输出：打印整体均值，并可写入 summary JSON。

示例：
  python scripts/run_eval.py --dataset datasets/crud_read_eval.csv --max-rows 15 --output datasets/tmp_eval.csv --pred-style short
  python scripts/run_crud_original_metrics.py --input datasets/tmp_eval.csv --output datasets/tmp_eval_crud_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _jieba_cut(text: str) -> List[str]:
    import jieba

    return list(jieba.cut(text or ""))


def _bleu(pred: str, ref: str) -> Dict[str, float]:
    import evaluate

    # 优先使用 CRUD_RAG 自带的离线 metric 脚本，避免联网下载
    root = Path(__file__).resolve().parent.parent
    local_bleu = root / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "bleu"
    bleu = evaluate.load(str(local_bleu)) if local_bleu.is_dir() else evaluate.load("bleu")
    res = bleu.compute(predictions=[pred], references=[[ref]], tokenizer=_jieba_cut)
    return {
        "bleu": float(res.get("bleu") or 0.0),
        "bleu1": float((res.get("precisions") or [0, 0, 0, 0])[0]),
        "bleu2": float((res.get("precisions") or [0, 0, 0, 0])[1]),
        "bleu3": float((res.get("precisions") or [0, 0, 0, 0])[2]),
        "bleu4": float((res.get("precisions") or [0, 0, 0, 0])[3]),
        "brevity_penalty": float(res.get("brevity_penalty") or 0.0),
    }


def _rouge_l(pred: str, ref: str) -> float:
    import evaluate

    root = Path(__file__).resolve().parent.parent
    local_rouge = root / "CRUD_RAG" / "src" / ".cache" / "huggingface" / "rouge"
    rouge = evaluate.load(str(local_rouge)) if local_rouge.is_dir() else evaluate.load("rouge")
    res = rouge.compute(
        predictions=[pred],
        references=[[ref]],
        tokenizer=_jieba_cut,
        rouge_types=["rougeL"],
    )
    return float(res.get("rougeL") or 0.0)


def _text2vec_bert_score(pred: str, ref: str) -> float:
    # text2vec 的 Similarity（名字叫 bert_score，但实现是句向量相似度）
    import os

    # 国内/证书问题：尽量走镜像，并在必要时关闭 SSL 校验（仅用于离线评测环境）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    from text2vec import Similarity

    try:
        sim = Similarity()
        return float(sim.get_score(pred or "", ref or "") or 0.0)
    except Exception:
        return 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="对本项目预测结果计算 CRUD 原版指标")
    p.add_argument("--input", type=Path, required=True, help="run_eval.py 输出 CSV")
    p.add_argument("--output", type=Path, default=None, help="写入 summary JSON（可选）")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"找不到输入文件: {args.input}")

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit("输入 CSV 为空")

    for k in ("gold_answer", "pred_answer"):
        if k not in (reader.fieldnames or []):
            raise SystemExit(f"输入 CSV 缺少列: {k}")

    bleu_list: List[float] = []
    rouge_list: List[float] = []
    bert_list: List[float] = []

    for r in rows:
        ref = (r.get("gold_answer") or "").strip()
        pred = (r.get("pred_answer") or "").strip()
        if not ref:
            continue
        b = _bleu(pred, ref)["bleu"]
        bleu_list.append(b)
        rouge_list.append(_rouge_l(pred, ref))
        bert_list.append(_text2vec_bert_score(pred, ref))

    n = len(bleu_list)
    summary = {
        "input": str(args.input.resolve()),
        "n": n,
        "avg_bleu": (sum(bleu_list) / n) if n else 0.0,
        "avg_rougeL": (sum(rouge_list) / n) if n else 0.0,
        "avg_bertScore_text2vec": (sum(bert_list) / n) if n else 0.0,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

