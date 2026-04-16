#!/usr/bin/env python3
"""
将 CRUD-RAG 官方 split_merged.json 中的问答子集导出为本项目 run_eval.py 可用的 CSV。

官方仓库: https://github.com/IAAR-Shanghai/CRUD_RAG
数据文件通常为: data/crud_split/split_merged.json

用法:
  python scripts/crud_rag_export_eval_csv.py ^
    --json "D:\\CRUD_RAG\\data\\crud_split\\split_merged.json" ^
    --output datasets/crud_read_eval.csv ^
    --subset questanswer_1doc ^
    --limit 200

subset 可选: questanswer_1doc | questanswer_2docs | questanswer_3docs | all_read
  all_read 表示合并上述三个列表。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _as_answer(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return "\n".join(str(x) for x in val if x is not None)
    return str(val).strip()


def _pick_question(obj: Dict[str, Any]) -> str:
    for k in ("questions", "question", "query"):
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _pick_answer(obj: Dict[str, Any]) -> str:
    for k in ("answers", "answer", "reference", "references"):
        v = obj.get(k)
        if v is not None:
            t = _as_answer(v)
            if t:
                return t
    return ""


def _pick_id(obj: Dict[str, Any], fallback: int) -> str:
    for k in ("ID", "id", "idx"):
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return str(fallback)


def _pick_evidence(obj: Dict[str, Any]) -> str:
    for k in ("docs", "doc_names", "sources", "filenames", "file", "doc_path"):
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            s = ";".join(str(x) for x in v if x)
            if s:
                return s
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="CRUD-RAG JSON -> qa_eval 格式 CSV")
    parser.add_argument("--json", type=Path, required=True, help="split_merged.json 路径")
    parser.add_argument("--output", type=Path, default=Path("datasets/crud_read_eval.csv"))
    parser.add_argument(
        "--subset",
        default="questanswer_1doc",
        choices=[
            "questanswer_1doc",
            "questanswer_2docs",
            "questanswer_3docs",
            "all_read",
        ],
    )
    parser.add_argument("--limit", type=int, default=0, help="最多导出条数，0 为全部")
    args = parser.parse_args()

    if not args.json.is_file():
        print(f"找不到文件: {args.json}", file=sys.stderr)
        return 1

    with args.json.open(encoding="utf-8") as f:
        data = json.load(f)

    keys: List[str]
    if args.subset == "all_read":
        keys = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
    else:
        keys = [args.subset]

    rows_out: List[Dict[str, str]] = []
    for key in keys:
        part = data.get(key)
        if not isinstance(part, list):
            print(f"警告: JSON 中无列表字段「{key}」，跳过", file=sys.stderr)
            continue
        for i, obj in enumerate(part):
            if not isinstance(obj, dict):
                continue
            q = _pick_question(obj)
            a = _pick_answer(obj)
            if not q or not a:
                continue
            rows_out.append(
                {
                    "id": _pick_id(obj, len(rows_out) + 1),
                    "question": q,
                    "answer": a,
                    "evidence": _pick_evidence(obj),
                    "type": key,
                }
            )

    if args.limit and args.limit > 0:
        rows_out = rows_out[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer", "evidence", "type"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"已写入 {len(rows_out)} 条 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
