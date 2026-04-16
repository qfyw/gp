#!/usr/bin/env python3
"""
从 run_eval 输出的 CSV 生成「人工判分」用表格（Excel 可打开 utf-8-sig）。

标注员在 human_correct 列填写：
  1 / Y / 是 / true  → 判对
  0 / N / 否 / false → 判错
留空 → 不计入分母（可后补）

用法:
  python scripts/prepare_human_eval_sheet.py datasets/eval_read_full.csv -o datasets/human_eval_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="生成人工判分表")
    p.add_argument("eval_csv", type=Path, help="run_eval 输出的 CSV")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    if not args.eval_csv.is_file():
        print(f"找不到: {args.eval_csv}", file=sys.stderr)
        return 1

    out_fields = [
        "id",
        "question",
        "gold_answer",
        "pred_answer",
        "human_correct",
        "notes",
    ]

    with args.eval_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": (r.get("id") or "").strip(),
                    "question": (r.get("question") or "").strip(),
                    "gold_answer": (r.get("gold_answer") or "").strip(),
                    "pred_answer": (r.get("pred_answer") or "").strip(),
                    "human_correct": "",
                    "notes": "",
                }
            )

    print(f"已写入 {len(rows)} 行: {args.output}")
    print("判分标准建议写在 README 或论文：核心事实与 gold 一致、无严重幻觉即判对；表述不同可判对。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
