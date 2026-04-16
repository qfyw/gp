# 评测指标：规范化、字符级 F1、完全匹配、证据命中（宽松）
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Sequence, Tuple


def normalize_text(s: str) -> str:
    """用于 EM / F1：NFKC、去首尾空白、压缩连续空白。"""
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", s)
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def exact_match(pred: str, ref: str) -> bool:
    return normalize_text(pred) == normalize_text(ref)


def char_level_f1(pred: str, ref: str) -> float:
    """
    字符级 F1（多集合重叠，与 SQuAD token-F1 同构）。
    适合中文短答案；pred/ref 会先 normalize_text。
    """
    p = normalize_text(pred)
    r = normalize_text(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def split_evidence_field(evidence: str) -> List[str]:
    """按分号/换行拆分期望证据描述，去空。"""
    if not (evidence or "").strip():
        return []
    parts = re.split(r"[;\n]+", evidence)
    return [p.strip() for p in parts if p.strip()]


def evidence_hit_sources(
    evidence: str,
    chunk_sources: Sequence[str],
) -> Tuple[bool, List[str]]:
    """
    宽松命中：任一证据片段是某个 chunk.source 的子串，或反过来 source 是片段子串。
    用于评测「溯源是否指向标注文档」。
    """
    pieces = split_evidence_field(evidence)
    if not pieces:
        return True, []
    srcs = [normalize_text(s) for s in chunk_sources if s]
    matched: List[str] = []
    for piece in pieces:
        pn = normalize_text(piece)
        ok = any(
            pn in src or src in pn
            for src in srcs
        )
        if ok:
            matched.append(piece)
    return len(matched) == len(pieces), matched


def evidence_hit_sources_any(
    evidence: str,
    chunk_sources: Sequence[str],
) -> Tuple[bool, List[str]]:
    """
    弱约束：至少一条证据片段与某个 chunk.source 子串互含即记为命中。
    适合「标注里只列了部分出处」或严格全匹配过严的场景。
    """
    pieces = split_evidence_field(evidence)
    if not pieces:
        return True, []
    srcs = [normalize_text(s) for s in chunk_sources if s]
    matched: List[str] = []
    for piece in pieces:
        pn = normalize_text(piece)
        if any(pn in src or src in pn for src in srcs):
            matched.append(piece)
    return len(matched) > 0, matched


def evidence_hit_answer(evidence: str, answer: str) -> Tuple[bool, List[str]]:
    """备选：答案文本中是否出现证据关键词（弱指标，易受幻觉影响）。"""
    pieces = split_evidence_field(evidence)
    if not pieces:
        return True, []
    an = normalize_text(answer)
    matched = [p for p in pieces if normalize_text(p) in an]
    return len(matched) == len(pieces), matched
