#!/usr/bin/env python3
"""
对 run_eval 输出的 CSV 做「错误类型」粗分类（不重新跑检索）。

说明：
- CSV 里没有写入检索片段，无法严格判定「只怪检索」或「只怪生成」。
- 用可观测信号做代理：
  - abstain：预测里出现「未提及/无法确定」等模板句 → 多为证据未进上下文或模型保守拒答
  - gold_span_recall：把参考答案按标点切成短片段，看有多少子串出现在预测中 → 低则整体偏离 gold（常伴随检索未命中或严重幻觉）
  - digit_recall：gold 中的阿拉伯数字是否在 pred 中出现 → 低则事实/数字层面错

用法:
  python scripts/analyze_eval_errors.py datasets/compare_topk_n50/topk_v5_k5_g10.csv
  python scripts/analyze_eval_errors.py datasets/eval_n50.csv --out datasets/error_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 拒答/保守表述（命中则标为 abstain_proxy）
ABSTAIN_PATTERNS = (
    "未明确提及",
    "未提及",
    "未能检索",
    "无法确定",
    "无法回答",
    "找不到",
    "没有明确",
    "资料中未",
    "未在检索",
    "检索结果中未",
    "根据检索",
    "未找到",
    "暂无",
    "不清楚",
    "未能从",
    "没有提供",
)

# 按标点切 gold，过短片段丢弃
_SPLIT_RE = re.compile(r"[，。；、\n]+")


def _segments(text: str, *, min_len: int = 6, max_segs: int = 12) -> list[str]:
    parts = []
    for s in _SPLIT_RE.split((text or "").strip()):
        s = s.strip()
        if len(s) >= min_len:
            parts.append(s)
    return parts[:max_segs]


def _gold_span_recall(gold: str, pred: str) -> float:
    segs = _segments(gold)
    if not segs:
        return 1.0 if (gold or "").strip() in (pred or "") else 0.0
    hit = sum(1 for s in segs if s in (pred or ""))
    return hit / len(segs)


def _digit_recall(gold: str, pred: str) -> tuple[float, int]:
    """gold 中出现的数字串，有多少在 pred 里出现。"""
    g = re.findall(r"\d+(?:\.\d+)?", gold or "")
    if not g:
        return 1.0, 0
    p = pred or ""
    hit = sum(1 for d in g if d in p)
    return hit / len(g), len(g)


def classify_row(
    gold: str,
    pred: str,
    char_f1: float,
    *,
    f1_high: float,
    f1_low: float,
    span_low: float,
    digit_low: float,
) -> str:
    if char_f1 >= f1_high:
        return "ok_high_f1"
    pred = pred or ""
    if any(p in pred for p in ABSTAIN_PATTERNS):
        return "abstain_proxy"
    span_r = _gold_span_recall(gold, pred)
    dig_r, n_d = _digit_recall(gold, pred)

    if n_d >= 2 and dig_r < digit_low and char_f1 < f1_high:
        return "numeric_fact_mismatch"
    if span_r < span_low and char_f1 < f1_low:
        return "low_gold_coverage"
    if span_r >= span_low and char_f1 < f1_high:
        return "partial_span_but_low_f1"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description="评测 CSV 错误类型粗分")
    p.add_argument("csv_path", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Markdown 报告路径")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--f1-high", type=float, default=0.72, help="高于此视为整体可接受")
    p.add_argument("--f1-low", type=float, default=0.55)
    p.add_argument("--span-low", type=float, default=0.25, help="gold 片段命中率低于此且 f1 低 → 强偏离")
    p.add_argument("--digit-low", type=float, default=0.5)
    args = p.parse_args()

    if not args.csv_path.is_file():
        print(f"找不到: {args.csv_path}", file=sys.stderr)
        return 1

    with args.csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    counts: Counter[str] = Counter()
    by_type: dict[str, list[dict]] = defaultdict(list)

    for r in rows:
        try:
            f1 = float((r.get("char_f1") or "0").strip())
        except ValueError:
            f1 = 0.0
        gold = (r.get("gold_answer") or "").strip()
        pred = (r.get("pred_answer") or "").strip()
        rid = (r.get("id") or "").strip()
        typ = classify_row(
            gold,
            pred,
            f1,
            f1_high=args.f1_high,
            f1_low=args.f1_low,
            span_low=args.span_low,
            digit_low=args.digit_low,
        )
        counts[typ] += 1
        span_r = _gold_span_recall(gold, pred)
        dig_r, n_dig = _digit_recall(gold, pred)
        by_type[typ].append(
            {
                "id": rid,
                "char_f1": round(f1, 4),
                "gold_span_recall": round(span_r, 4),
                "digit_recall": round(dig_r, 4),
                "n_digits_in_gold": n_dig,
                "question_preview": (r.get("question") or "")[:80],
            }
        )

    n = len(rows)
    lines = [
        f"# 错误类型粗分（代理指标）",
        f"",
        f"- 输入: `{args.csv_path}`",
        f"- 样本数: {n}",
        f"",
        f"## 类型说明",
        f"",
        f"| 标签 | 含义（代理） |",
        f"|------|----------------|",
        f"| `ok_high_f1` | 字符 F1≥{args.f1_high}，整体可接受（仍可能 EM=0） |",
        f"| `abstain_proxy` | 预测含「未提及/无法确定」等 → **证据未用上或模型拒答** |",
        f"| `numeric_fact_mismatch` | gold 多数字未进预测 → **事实/数字错误**（检索或生成皆可能） |",
        f"| `low_gold_coverage` | gold 片段命中率低且 F1 低 → **答案整体跑偏**（常像检索未命中） |",
        f"| `partial_span_but_low_f1` | 命中部分原文片段但 F1 仍低 → **表述/漏要点/多扯无关**（偏生成侧） |",
        f"| `other` | 未归入上类 |",
        f"",
        f"> 若要**严格区分检索 vs 生成**，需在评测时把检索片段写入 CSV 或对本脚本加 `--re-retrieve`（未实现）。",
        f"",
        f"## 占比",
        f"",
    ]
    for k, v in counts.most_common():
        lines.append(f"- **{k}**: {v} ({100.0 * v / n:.1f}%)")
    lines.append("")
    lines.append("## 各类型样例 id（最多 5 条）")
    lines.append("")
    for typ in sorted(by_type.keys()):
        lines.append(f"### {typ}")
        for item in by_type[typ][:5]:
            lines.append(f"- `{item['id']}` F1={item['char_f1']} span_rec={item['gold_span_recall']} digit_rec={item['digit_recall']} — {item['question_preview']}…")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\n已写入: {args.out}", file=sys.stderr)

    if args.json_out:
        payload = {
            "input": str(args.csv_path.resolve()),
            "n": n,
            "thresholds": {
                "f1_high": args.f1_high,
                "f1_low": args.f1_low,
                "span_low": args.span_low,
                "digit_low": args.digit_low,
            },
            "counts": dict(counts),
            "by_type": {k: v[:20] for k, v in by_type.items()},
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入: {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
