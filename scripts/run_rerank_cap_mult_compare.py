#!/usr/bin/env python3
"""
固定 vector / keyword / graph 下，对比重排参数 RERANK_RECALL_MULT 与 RERANK_DOC_CAP。

说明（与 src/retriever.hybrid_retrieve 一致）：
- 须 RERANK_ENABLED=1；本脚本每次子进程都会设 RERANK_ENABLED=1。
- RERANK_DOC_CAP=0 或不设置：重排后保留条数 = vector_top_k + keyword_top_k + BM25_TOP_K（BM25 开启时，
  对 5/5/5 即 cap_default=15）。
- RERANK_DOC_CAP>0：实际 cap = min(cap_default, RERANK_DOC_CAP)。

因 config 在 import 时读环境变量，每组用独立子进程跑 run_eval.py。

用法:
  python scripts/run_rerank_cap_mult_compare.py --max-rows 30 --out-dir datasets/compare_rerank_n30
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _default_sweeps() -> list[dict[str, Any]]:
    """mult × cap 小网格；cap None 表示不设置环境变量（走默认 0）。"""
    return [
        {"slug": "rr_m2_cap_auto", "label": "mult=2,cap默认", "mult": "2", "cap": None},
        {"slug": "rr_m3_cap_auto", "label": "mult=3,cap默认", "mult": "3", "cap": None},
        {"slug": "rr_m4_cap_auto", "label": "mult=4,cap默认", "mult": "4", "cap": None},
        {"slug": "rr_m3_cap8", "label": "mult=3,cap=8", "mult": "3", "cap": "8"},
        {"slug": "rr_m3_cap10", "label": "mult=3,cap=10", "mult": "3", "cap": "10"},
        {"slug": "rr_m3_cap12", "label": "mult=3,cap=12", "mult": "3", "cap": "12"},
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="RERANK mult/cap 对照（子进程+run_eval）")
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "crud_read_eval.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "compare_rerank")
    p.add_argument("--max-rows", type=int, default=30)
    p.add_argument("--vector-top-k", type=int, default=5)
    p.add_argument("--keyword-top-k", type=int, default=5)
    p.add_argument("--graph-max", type=int, default=10)
    args = p.parse_args()

    if not args.dataset.is_file():
        print(f"找不到数据集: {args.dataset}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    run_eval = ROOT / "scripts" / "run_eval.py"

    sweeps = _default_sweeps()
    merged_rows: list[dict[str, Any]] = []

    for cfg in sweeps:
        slug = cfg["slug"]
        out_csv = args.out_dir / f"{slug}.csv"
        sum_json = args.out_dir / f"{slug}_summary.json"
        env = os.environ.copy()
        env["RERANK_ENABLED"] = "1"
        env["RERANK_RECALL_MULT"] = str(cfg["mult"])
        if cfg["cap"] is None:
            env.pop("RERANK_DOC_CAP", None)
        else:
            env["RERANK_DOC_CAP"] = str(cfg["cap"])

        cmd = [
            py,
            str(run_eval),
            "--dataset",
            str(args.dataset),
            "--output",
            str(out_csv),
            "--summary-json",
            str(sum_json),
            "--modes",
            "full",
            "--vector-top-k",
            str(args.vector_top_k),
            "--keyword-top-k",
            str(args.keyword_top_k),
            "--graph-max",
            str(args.graph_max),
            "--pred-style",
            "short",
        ]
        if args.max_rows > 0:
            cmd.extend(["--max-rows", str(args.max_rows)])

        print(f"\n>>> {cfg['label']}  ({slug})", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f"失败 exit={r.returncode}: {slug}", file=sys.stderr)
            return r.returncode or 1

        row: dict[str, Any] = {
            "slug": slug,
            "label": cfg["label"],
            "RERANK_RECALL_MULT": cfg["mult"],
            "RERANK_DOC_CAP": cfg["cap"] if cfg["cap"] is not None else "0(默认)",
        }
        if sum_json.is_file():
            with sum_json.open(encoding="utf-8") as f:
                sj = json.load(f)
            pm = (sj.get("per_mode") or {}).get("full") or {}
            row["n"] = pm.get("n")
            row["mean_char_f1"] = pm.get("mean_char_f1")
            row["mean_em"] = pm.get("mean_em")
            row["latency_p50_sec"] = pm.get("latency_p50_sec")
            row["latency_p95_sec"] = pm.get("latency_p95_sec")
        merged_rows.append(row)

    out_path = args.out_dir / "rerank_cap_mult_summary.json"
    payload = {
        "dataset": str(args.dataset.resolve()),
        "max_rows": args.max_rows,
        "vector_top_k": args.vector_top_k,
        "keyword_top_k": args.keyword_top_k,
        "graph_max": args.graph_max,
        "RERANK_ENABLED": "1",
        "runs": merged_rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n汇总（mean_char_f1 降序）:", flush=True)
    for r in sorted(merged_rows, key=lambda x: float(x.get("mean_char_f1") or 0), reverse=True):
        print(
            f"  {r['slug']}: char_f1={r.get('mean_char_f1')}  em={r.get('mean_em')}  "
            f"P50={r.get('latency_p50_sec')}s  mult={r['RERANK_RECALL_MULT']} cap={r['RERANK_DOC_CAP']}",
            flush=True,
        )
    print(f"\n已写入: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
