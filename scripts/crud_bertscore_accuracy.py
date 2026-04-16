#!/usr/bin/env python3
"""
按条计算：若 text2vec Similarity(pred, gold) >= 阈值则判对，准确率 = 判对数 / 有效条数。

用于多份 run_eval 输出 CSV（需 gold_answer / pred_answer）。

示例:
  python scripts/crud_bertscore_accuracy.py --dir datasets/compare_topk_n50 --threshold 0.8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def accuracy_for_csv(path: Path, sim: object, threshold: float) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        for k in ("gold_answer", "pred_answer"):
            if k not in fn:
                raise ValueError(f"{path.name} 缺少列: {k}")
        rows = list(reader)

    ok = 0
    n = 0
    scores: list[float] = []
    for r in rows:
        ref = (r.get("gold_answer") or "").strip()
        pred = (r.get("pred_answer") or "").strip()
        if not ref:
            continue
        n += 1
        try:
            s = float(sim.get_score(pred, ref) or 0.0)
        except Exception:
            s = 0.0
        scores.append(s)
        if s >= threshold:
            ok += 1

    return {
        "slug": path.stem,
        "input": str(path.resolve()),
        "n": n,
        "threshold": threshold,
        "correct": ok,
        "accuracy": (ok / n) if n else 0.0,
        "mean_bertScore_text2vec": (sum(scores) / len(scores)) if scores else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="text2vec 相似度阈值准确率")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--glob", default="topk_*.csv")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="相似度 >= 该值判为正确（默认 0.8）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认 <dir>/bertscore_accuracy_t{threshold}.json",
    )
    args = p.parse_args()

    if not args.dir.is_dir():
        print(f"目录不存在: {args.dir}", file=sys.stderr)
        return 1

    files = sorted(args.dir.glob(args.glob))
    if not files:
        print(f"未匹配: {args.dir}/{args.glob}", file=sys.stderr)
        return 1

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    from text2vec import Similarity

    print("加载 text2vec Similarity …", flush=True)
    sim = Similarity()

    runs = [accuracy_for_csv(fp, sim, args.threshold) for fp in files]
    out = {
        "dir": str(args.dir.resolve()),
        "glob": args.glob,
        "runs": runs,
    }
    if args.output is not None:
        out_path = args.output
    else:
        tstr = str(args.threshold).replace(".", "p")
        out_path = args.dir / f"bertscore_accuracy_t{tstr}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"已写入: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
