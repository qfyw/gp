#!/usr/bin/env python3
"""
用 LLM 对 run_eval 的 CSV 做「是否答对 + 错误原因」自动标注（辅助人工，不可替代严谨人工抽检）。

读取列：question, gold_answer, pred_answer（及 id 可选）。
输出：在原表基础上增加 llm_correct, llm_error_category, llm_reason（或单独 CSV）。

错误类别（固定枚举，便于统计）：
  ok                  — 核心事实与参考答案一致（表述可不同）
  factual_error       — 关键事实/数字/主体错误
  incomplete          — 明显漏答子问题或缺关键要点
  false_abstain       — 无理由拒答、说资料未提及但 gold 应为可答事实题
  over_claim          — 编造或过度推断（疑似幻觉）
  format_noise        — 事实基本对但大量套话/重复/结构差导致难判（仍可按题设判错）
  other               — 其他

用法:
  python scripts/llm_eval_judge.py --input datasets/eval_read_full.csv --output datasets/eval_read_full_llm_judged.csv
  python scripts/llm_eval_judge.py --input datasets/eval_read_full.csv --max-rows 20 --sleep 0.3

依赖：.env 中 OPENAI_API_KEY / BASE / MODEL（与项目一致）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORIES = (
    "ok",
    "factual_error",
    "incomplete",
    "false_abstain",
    "over_claim",
    "format_noise",
    "other",
)

JUDGE_PROMPT = """你是严谨的中文问答评测员。只根据「问题、参考答案、模型预测」三者判断，不要编造参考答案里没有的信息。

## 判对标准（宽松）
- 若预测中的**核心事实**（主体、数字、时间、结论）与参考答案**一致或等价表述**，判为正确。
- 表述不同、多一句无关套话、略少修饰语，**仍可判对**。
- 若预测**关键数字/主体/结论错误**，或**漏掉多子问中的主要一问**，判为错。

## 错误类别（必须选且仅选一个）
- ok：判对
- factual_error：关键事实/数字/对象错误
- incomplete：明显漏答、要点不全
- false_abstain：预测称无法回答/资料未提及等，但参考答案显示应为明确事实答案
- over_claim：明显编造、过度推断、添加参考答案中无依据的断言
- format_noise：事实基本成立但冗长重复/结构混乱到影响理解（若核心仍错则用 factual_error）
- other：无法归入以上

## 输出格式（仅一行 JSON，无 markdown）
{{"correct": true 或 false, "category": "上面枚举之一", "reason": "20字以内中文简述"}}

## 问题
{question}

## 参考答案
{gold}

## 模型预测
{pred}
"""


def _parse_json_obj(text: str) -> dict | None:
    text = (text or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def judge_one(llm, question: str, gold: str, pred: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question or "",
        gold=gold or "",
        pred=pred or "",
    )
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    obj = _parse_json_obj(raw)
    if not obj:
        return {
            "llm_correct": "",
            "llm_error_category": "parse_error",
            "llm_reason": (raw or "")[:200],
            "llm_raw": (raw or "")[:500],
        }
    cat = (obj.get("category") or "other").strip().lower()
    if cat not in CATEGORIES:
        cat = "other"
    correct = obj.get("correct")
    if isinstance(correct, str):
        correct = correct.strip().lower() in ("true", "1", "yes", "是")
    else:
        correct = bool(correct)
    # 对错与类别对齐：判对则统一标 ok；判错则不允许 category=ok
    if correct:
        cat = "ok"
    elif cat == "ok":
        cat = "other"
    reason = str(obj.get("reason") or "")[:500]
    return {
        "llm_correct": "1" if correct else "0",
        "llm_error_category": cat,
        "llm_reason": reason,
        "llm_raw": "",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="LLM 自动评测 + 错误原因")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0, help="0=全量")
    p.add_argument("--sleep", type=float, default=0.2, help="每条间隔秒数，降限流风险")
    p.add_argument("--pred-column", default="pred_answer", help="用哪一列作为预测（默认 pred_answer）")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"找不到: {args.input}", file=sys.stderr)
        return 1

    from src.generator import get_llm

    llm = get_llm()
    if llm is None:
        print("无法创建 LLM：请配置 OPENAI_API_KEY 等", file=sys.stderr)
        return 1
    if hasattr(llm, "temperature"):
        try:
            llm.temperature = 0
        except Exception:
            pass

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    pred_col = args.pred_column
    for k in ("question", "gold_answer", pred_col):
        if rows and k not in rows[0]:
            print(f"CSV 缺少列: {k}", file=sys.stderr)
            return 1

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_fields = list(rows[0].keys()) if rows else []
    for extra in ("llm_correct", "llm_error_category", "llm_reason", "llm_raw"):
        if extra not in out_fields:
            out_fields.append(extra)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok_n = 0
    labeled = 0

    with args.output.open("w", encoding="utf-8-sig", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            q = (r.get("question") or "").strip()
            gold = (r.get("gold_answer") or "").strip()
            pred = (r.get(pred_col) or "").strip()
            try:
                j = judge_one(llm, q, gold, pred)
            except Exception as e:
                j = {
                    "llm_correct": "",
                    "llm_error_category": "api_error",
                    "llm_reason": str(e)[:300],
                    "llm_raw": "",
                }
            row = {**r, **j}
            w.writerow(row)
            if j.get("llm_correct") in ("0", "1"):
                labeled += 1
                if j["llm_correct"] == "1":
                    ok_n += 1
            print(f"[{i+1}/{len(rows)}] id={r.get('id','')[:8]}… {j.get('llm_error_category')} correct={j.get('llm_correct')}", flush=True)
            if args.sleep > 0 and i + 1 < len(rows):
                time.sleep(args.sleep)

    acc = (ok_n / labeled) if labeled else 0.0
    summary_path = args.output.with_name(args.output.stem + "_llm_summary.json")
    # 统计 category 频数
    from collections import Counter

    cat_counts: Counter[str] = Counter()
    with args.output.open(encoding="utf-8-sig", newline="") as f:
        for r2 in csv.DictReader(f):
            cat_counts[r2.get("llm_error_category") or ""] += 1

    payload = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "n_rows": len(rows),
        "llm_labeled": labeled,
        "llm_accuracy": round(acc, 4),
        "category_counts": dict(cat_counts),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已写入: {args.output}", flush=True)
    print(f"已写入: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
