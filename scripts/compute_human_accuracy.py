#!/usr/bin/env python3
"""
对已填写 human_correct 的表格计算人工准确率。

用法:
  python scripts/compute_human_accuracy.py datasets/human_eval_sheet_filled.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _parse_correct(raw: str) -> bool | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s in ("1", "y", "yes", "是", "对", "true", "正确", "√", "✓"):
        return True
    if s in ("0", "n", "no", "否", "错", "false", "错误", "×"):
        return False
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="人工准确率统计")
    p.add_argument("sheet_csv", type=Path)
    args = p.parse_args()

    if not args.sheet_csv.is_file():
        print(f"找不到: {args.sheet_csv}", file=sys.stderr)
        return 1

    with args.sheet_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ok = 0
    bad = 0
    skip = 0
    unknown = 0
    for r in rows:
        v = _parse_correct(r.get("human_correct", ""))
        if v is True:
            ok += 1
        elif v is False:
            bad += 1
        elif v is None and not (r.get("human_correct") or "").strip():
            skip += 1
        else:
            unknown += 1

    labeled = ok + bad
    if labeled == 0:
        print("没有已标注行（human_correct 为空或无法解析）", file=sys.stderr)
        return 1

    acc = ok / labeled
    print(f"已标注: {labeled} 条（跳过未填: {skip}）")
    if unknown:
        print(f"警告: 无法解析 human_correct 的行数: {unknown}", file=sys.stderr)
    print(f"判对: {ok}  判错: {bad}")
    print(f"人工准确率: {acc:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
