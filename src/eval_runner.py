# 批量评测核心逻辑（供 run_eval.py / run_eval_compare.py 复用）
from __future__ import annotations

import csv
import json
import re
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.workflow import run_advanced_workflow
from src.eval_metrics import (
    char_level_f1,
    evidence_hit_sources,
    evidence_hit_sources_any,
    exact_match,
)
from src.config import RERANK_DOC_CAP, RERANK_ENABLED, RERANK_RECALL_MULT
from src.reranker import rerank_doc_chunks
from src.retriever import (
    RetrievedChunk,
    hybrid_retrieve,
    keyword_search,
    merge_vector_keyword_chunks,
    vector_search,
)


_CITATION_BRACKETS_RE = re.compile(r"\[[^\]]*\]")
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")


def format_pred_answer(pred: str, *, style: str) -> str:
    """
    评测用的预测答案格式化（让 F1 更贴近“结论是否答对”）。

    style:
      - full: 原样
      - short: 去引用/Markdown 加粗，取首段（遇到空行截断），并做轻量清洗
    """
    s = (pred or "").strip()
    if not s or style == "full":
        return s

    # 先截断到首段（大多数长答案第一段就包含结论）
    if "\n\n" in s:
        s = s.split("\n\n", 1)[0].strip()
    else:
        # 有些答案用换行分点；取前 3 行足够覆盖结论
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = " ".join(lines[:3]).strip()

    # 去掉引用/标注与常见格式
    s = _CITATION_BRACKETS_RE.sub("", s)
    s = _MD_BOLD_RE.sub(r"\1", s)

    # 去掉常见“注/说明”尾巴（尽量不伤及答案主体）
    for marker in ("注：", "注:", "备注：", "备注:", "说明：", "说明:"):
        if marker in s:
            s = s.split(marker, 1)[0].strip()

    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_chunks(results: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen: set[str] = set()
    out: List[RetrievedChunk] = []
    for c in results:
        if c.content not in seen:
            seen.add(c.content)
            out.append(c)
    return out


def retrieve_by_mode(
    query: str,
    vectorstore: Any,
    graph: Any,
    mode: str,
    vector_top_k: int,
    keyword_top_k: int,
    graph_max: int,
) -> List[RetrievedChunk]:
    if mode in ("full", "vector_keyword_graph"):
        return hybrid_retrieve(
            query,
            vectorstore=vectorstore,
            graph=graph,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            graph_max=graph_max,
        )
    mult = RERANK_RECALL_MULT if RERANK_ENABLED else 1
    if mode == "vector":
        chunks: List[RetrievedChunk] = []
        if vectorstore is not None:
            vk = max(1, vector_top_k * mult)
            chunks.extend(vector_search(vectorstore, query, top_k=vk))
        chunks = _dedupe_chunks(chunks)
        if RERANK_ENABLED and chunks:
            chunks = rerank_doc_chunks(query, chunks, top_n=vector_top_k)
        return chunks
    if mode == "vector_keyword":
        vec: List[RetrievedChunk] = []
        if vectorstore is not None:
            vec = vector_search(vectorstore, query, top_k=max(1, vector_top_k * mult))
        kw = keyword_search(query, top_k=max(1, keyword_top_k * mult))
        merged_kw = merge_vector_keyword_chunks(vec, kw)
        if RERANK_ENABLED and merged_kw:
            cap_default = max(1, vector_top_k + keyword_top_k)
            cap = min(cap_default, RERANK_DOC_CAP) if (RERANK_DOC_CAP and RERANK_DOC_CAP > 0) else cap_default
            merged_kw = rerank_doc_chunks(
                query, merged_kw, top_n=cap
            )
        return merged_kw
    raise ValueError(f"未知 mode: {mode}")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        rows = []
        for row in reader:
            rows.append({k: (v or "").strip() if v else "" for k, v in row.items()})
        return rows


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = k - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def run_eval_batch(
    rows: List[Dict[str, str]],
    vs: Any,
    g: Any,
    modes: List[str],
    vector_top_k: int,
    keyword_top_k: int,
    graph_max: int,
    output_csv: Path | None,
    *,
    label: str = "",
    pred_style: str = "short",
) -> Dict[str, Any]:
    """
    跑一批样本，可选写入逐条 CSV。
    返回 summary：含 per_mode 聚合指标、可选 label（对比实验名）。
    """
    fieldnames = [
        "run_label",
        "mode",
        "id",
        "type",
        "question",
        "gold_answer",
        "pred_answer",
        "pred_answer_raw",
        "em",
        "char_f1",
        "evidence_strict_hit",
        "evidence_any_hit",
        "retrieve_sec",
        "llm_sec",
        "latency_sec",
        "error",
    ]

    all_latencies: Dict[str, List[float]] = {m: [] for m in modes}
    all_f1: Dict[str, List[float]] = {m: [] for m in modes}
    all_em: Dict[str, List[float]] = {m: [] for m in modes}
    all_evhit: Dict[str, List[float]] = {m: [] for m in modes}
    all_evany: Dict[str, List[float]] = {m: [] for m in modes}

    fout_ctx = (
        output_csv.open("w", encoding="utf-8-sig", newline="")
        if output_csv is not None
        else None
    )
    try:
        writer = None
        if fout_ctx is not None:
            writer = csv.DictWriter(fout_ctx, fieldnames=fieldnames)
            writer.writeheader()

        for mode in modes:
            for row in rows:
                q = row.get("question", "")
                gold = row.get("answer", "")
                ev = row.get("evidence", "")
                rid = row.get("id", "")
                typ = row.get("type", "")
                err = ""
                pred = ""
                lat = 0.0
                retrieve_sec = 0.0
                llm_sec = 0.0
                chunks: List[RetrievedChunk] = []
                t0 = time.perf_counter()
                try:
                    t_retrieve0 = time.perf_counter()
                    chunks = retrieve_by_mode(
                        q,
                        vs,
                        g,
                        mode,
                        vector_top_k,
                        keyword_top_k,
                        graph_max,
                    )
                    retrieve_sec = time.perf_counter() - t_retrieve0
                    if chunks:
                        t_llm0 = time.perf_counter()
                        # 评测默认用 concise，避免"强制溯源格式"拉低字符 F1
                        # 支持 eval_optimized 和 crud_optimized 模式
                        env_style = (os.getenv("EVAL_PROMPT_STYLE", "") or "").strip().lower()
                        if pred_style == "short" and env_style == "crudrag":
                            style = "crudrag"
                        elif pred_style == "short" and env_style == "crud_optimized":
                            style = "crud_optimized"
                        elif pred_style == "short" and env_style == "eval_optimized":
                            style = "eval_optimized"
                        else:
                            style = "concise" if pred_style == "short" else "sourced"
                        state = run_advanced_workflow(q, chunks, graph=g, answer_style=style)
                        pred = state.get("final_answer") or ""
                        llm_sec = time.perf_counter() - t_llm0
                    lat = time.perf_counter() - t0
                except Exception as e:
                    err = str(e)
                    lat = time.perf_counter() - t0

                pred_raw = pred
                pred = format_pred_answer(pred_raw, style=pred_style)
                # 避免 CSV 单条记录里含真实换行，导致编辑器/Excel 显示成“多行”而卡顿
                pred_raw_csv = (
                    (pred_raw or "")
                    .replace("\r\n", "\\n")
                    .replace("\n", "\\n")
                    .replace("\r", "\\n")
                )
                em = 1.0 if exact_match(pred, gold) else 0.0
                f1 = char_level_f1(pred, gold)
                srcs = [c.source for c in chunks]

                has_evidence = bool((ev or "").strip())
                if has_evidence:
                    ev_ok, _ = evidence_hit_sources(ev, srcs)
                    ev_hit = 1.0 if ev_ok else 0.0
                    ev_any_ok, _ = evidence_hit_sources_any(ev, srcs)
                    ev_any = 1.0 if ev_any_ok else 0.0
                    ev_s_str = f"{ev_hit:.4f}"
                    ev_a_str = f"{ev_any:.4f}"
                else:
                    ev_s_str = "NA"
                    ev_a_str = "NA"

                if writer is not None:
                    writer.writerow(
                        {
                            "run_label": label,
                            "mode": mode,
                            "id": rid,
                            "type": typ,
                            "question": q,
                            "gold_answer": gold,
                            "pred_answer": pred,
                            "pred_answer_raw": pred_raw_csv,
                            "em": f"{em:.4f}",
                            "char_f1": f"{f1:.4f}",
                            "evidence_strict_hit": ev_s_str,
                            "evidence_any_hit": ev_a_str,
                            "retrieve_sec": f"{retrieve_sec:.4f}",
                            "llm_sec": f"{llm_sec:.4f}",
                            "latency_sec": f"{lat:.4f}",
                            "error": err,
                        }
                    )

                all_latencies[mode].append(lat)
                all_em[mode].append(em)
                all_f1[mode].append(f1)
                if has_evidence:
                    all_evhit[mode].append(ev_hit)
                    all_evany[mode].append(ev_any)
    finally:
        if fout_ctx is not None:
            fout_ctx.close()

    per_mode: Dict[str, Any] = {}
    for mode in modes:
        n = len(all_f1[mode])
        if n == 0:
            continue
        lat_sorted = sorted(all_latencies[mode])
        n_ev = len(all_evhit[mode])
        per_mode[mode] = {
            "n": n,
            "mean_em": round(sum(all_em[mode]) / n, 6),
            "mean_char_f1": round(sum(all_f1[mode]) / n, 6),
            "evidence_strict_mean": (
                round(sum(all_evhit[mode]) / n_ev, 6) if n_ev > 0 else None
            ),
            "evidence_any_mean": (
                round(sum(all_evany[mode]) / n_ev, 6) if n_ev > 0 else None
            ),
            "n_with_evidence": n_ev,
            "latency_p50_sec": round(percentile(lat_sorted, 50), 4),
            "latency_p95_sec": round(percentile(lat_sorted, 95), 4),
        }

    return {
        "label": label,
        "vector_top_k": vector_top_k,
        "keyword_top_k": keyword_top_k,
        "graph_max": graph_max,
        "modes": modes,
        "per_mode": per_mode,
    }


def build_full_summary(
    dataset_path: Path,
    output_csv: Path | None,
    batch_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.resolve()),
        "output_csv": str(output_csv.resolve()) if output_csv else None,
        **batch_meta,
    }


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# 评测汇总",
        "",
        f"- 时间（UTC）: `{summary.get('generated_at', '')}`",
        f"- 数据集: `{summary.get('dataset', '')}`",
        f"- 逐条 CSV: `{summary.get('output_csv', '')}`",
        "",
        "## 指标说明",
        "",
        "- **mean_em**：预测与标准答案逐字完全一致的比例；长答案下通常很低。",
        "- **mean_char_f1**：字符级 F1，主看检索+生成质量。",
        "- **evidence_***：仅统计 CSV 里 `evidence` 非空的样本。",
        "",
        "## 按检索模式",
        "",
    ]
    per_mode = summary.get("per_mode") or {}
    if isinstance(per_mode, dict) and per_mode:
        lines.append(
            "| mode | n | mean_em | mean_char_f1 | ev_strict | ev_any | n_ev | P50(s) | P95(s) |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for mode, m in per_mode.items():
            es = m.get("evidence_strict_mean")
            ea = m.get("evidence_any_mean")
            lines.append(
                f"| {mode} | {m.get('n')} | {m.get('mean_em')} | {m.get('mean_char_f1')} | "
                f"{es if es is not None else '—'} | {ea if ea is not None else '—'} | "
                f"{m.get('n_with_evidence')} | {m.get('latency_p50_sec')} | {m.get('latency_p95_sec')} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_compare_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    """多组实验对比表（每组通常单 mode）。"""
    lines = [
        "# 多组对比汇总",
        "",
        "| 实验组 | mode | vec_k | kw_k | g_max | n | mean_em | mean_char_f1 | ev_strict | P50(s) | P95(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        pm = r.get("per_mode") or {}
        mode = (r.get("modes") or ["?"])[0]
        stats = pm.get(mode, {})
        es = stats.get("evidence_strict_mean")
        lines.append(
            f"| {r.get('label', '')} | {mode} | {r.get('vector_top_k')} | {r.get('keyword_top_k')} | "
            f"{r.get('graph_max')} | {stats.get('n', '')} | {stats.get('mean_em', '')} | "
            f"{stats.get('mean_char_f1', '')} | {es if es is not None else '—'} | "
            f"{stats.get('latency_p50_sec', '')} | {stats.get('latency_p95_sec', '')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_compare_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
