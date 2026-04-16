#!/usr/bin/env python3
"""
简化的实体感知检索模块（避免编码问题）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .config import (
    BM25_TOP_K,
    BM25_ENABLED,
    RERANK_DOC_CAP,
    RERANK_ENABLED,
    RERANK_RECALL_MULT,
    RRF_K,
)
from .pg_db import bm25_search as pg_bm25_search
from .pg_db import keyword_search as pg_keyword_search
from .reranker import rerank_doc_chunks
from .retriever import (
    RetrievedChunk,
    bm25_search,
    boost_chunks_by_query_anchors,
    expand_query_for_hybrid,
    fuse_doc_chunks_rrf,
    graph_search,
    merge_vector_keyword_chunks,
    normalize_query_for_search,
    query_anchor_phrases,
    vector_search,
)

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore


@dataclass
class EntityMatch:
    """实体匹配信息"""
    entity: str
    match_type: str  # "exact" | "fuzzy" | "alias"
    positions: List[int]  # 在问题中的位置


def extract_entities_simple(question: str) -> Dict[str, List[str]]:
    """
    简化的实体提取（避免大量中文字符串）。
    
    实体类型：
    - date: 日期（YYYY年MM月DD日等格式）
    - number: 数字（金额、数量、百分比）
    - quoted: 引号内的内容（专有名词）
    - camel_case: 英文驼峰命名
    """
    q = question or ""
    entities: Dict[str, List[str]] = {
        "date": [],
        "number": [],
        "quoted": [],
        "camel_case": [],
    }

    # 1. 提取日期
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}[日号]",
        r"\d{4}年\d{1,2}月",
        r"\d{1,2}月\d{1,2}[日号]",
    ]
    for pattern in date_patterns:
        for m in re.finditer(pattern, q):
            entities["date"].append(m.group(0))

    # 2. 提取数字（金额、数量、百分比）
    number_patterns = [
        r"\d+万元?",
        r"\d+亿元?",
        r"\d+百万元?",
        r"\d+个?",
        r"\d+家?",
        r"\d+名?",
        r"\d+项?",
        r"\d+[多条|个|种|类]",
        r"\d+\.\d+%",
        r"\d+%",
    ]
    for pattern in number_patterns:
        for m in re.finditer(pattern, q):
            entities["number"].append(m.group(0))

    # 3. 提取引号内容（专有名词）
    quote_pattern = r'["\uff0c\uff1b\u201c\u201d\u300c\u300d](.+?)["\uff0c\uff1b\u201c\u201d\u300c\u300d]'
    for m in re.finditer(quote_pattern, q):
        entity = m.group(1)
        if entity:
            entities["quoted"].append(entity)

    # 4. 提取英文驼峰命名
    camel_case_pattern = r"(?<![A-Za-z0-9])([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"
    for m in re.finditer(camel_case_pattern, q):
        entities["camel_case"].append(m.group(1))

    return entities


def filter_chunks_by_entities_simple(
    chunks: List[RetrievedChunk],
    entities: Dict[str, List[str]],
    min_entity_matches: int = 0,  # 改为0，降低过滤强度
) -> List[RetrievedChunk]:
    """
    根据实体过滤文档（优化版本）。

    优化：
    - min_entity_matches 默认改为 0（不强制过滤）
    - 使用模糊匹配而非精确匹配
    - 保留更多相关文档
    """
    # 合并所有实体
    all_entities = set()
    for entity_list in entities.values():
        all_entities.update(entity_list)

    # 如果没有实体或 min_entity_matches=0，不过滤
    if not all_entities or min_entity_matches == 0:
        return chunks

    filtered_chunks: List[RetrievedChunk] = []
    for chunk in chunks:
        content = chunk.content.lower()
        source = chunk.source.lower()
        combined = content + "\n" + source

        # 计算匹配的实体数量（使用模糊匹配）
        entity_matches = 0
        for entity in all_entities:
            entity_lower = entity.lower()
            # 精确匹配
            if entity_lower in combined:
                entity_matches += 1
            # 模糊匹配（实体长度>=3，且出现2/3以上）
            elif len(entity_lower) >= 3:
                if entity_lower[:int(len(entity_lower)*0.7)] in combined:
                    entity_matches += 1

        # 至少匹配 min_entity_matches 个实体
        if entity_matches >= min_entity_matches:
            filtered_chunks.append(chunk)

    # 如果过滤后为空，保留原始文档（避免无结果）
    return filtered_chunks or chunks


def score_chunks_by_entities_simple(
    chunks: List[RetrievedChunk],
    entities: Dict[str, List[str]],
) -> List[RetrievedChunk]:
    """
    根据实体匹配数量对文档进行加权评分（优化版本）。

    优化：
    - 使用模糊匹配
    - 增加实体类型权重
    - 提高实体匹配的分数
    """
    # 合并所有实体，并按类型分组
    all_entities = set()
    entity_types: Dict[str, List[str]] = {
        "date": [],
        "number": [],
        "quoted": [],
        "camel_case": [],
    }
    for entity_type, entity_list in entities.items():
        entity_types[entity_type] = entity_list
        all_entities.update(entity_list)

    if not all_entities:
        return chunks

    # 实体类型权重（数字和引号内容权重更高）
    entity_type_weights = {
        "date": 1.0,
        "number": 1.5,
        "quoted": 2.0,
        "camel_case": 1.2,
    }

    # 为每个文档计算实体匹配分数
    scored_chunks: List[tuple[float, RetrievedChunk]] = []
    for i, chunk in enumerate(chunks):
        content = chunk.content.lower()
        source = chunk.source.lower()
        combined = content + "\n" + source

        # 计算加权实体匹配分数
        weighted_score = 0.0
        for entity_type, entity_list in entity_types.items():
            weight = entity_type_weights.get(entity_type, 1.0)
            for entity in entity_list:
                entity_lower = entity.lower()
                # 精确匹配
                if entity_lower in combined:
                    weighted_score += weight * 2.0
                # 模糊匹配（实体长度>=3，且出现2/3以上）
                elif len(entity_lower) >= 3:
                    if entity_lower[:int(len(entity_lower)*0.7)] in combined:
                        weighted_score += weight * 1.0

        # 基础分数 + 加权分数
        base_score = 1.0  # 保证至少有基础分数
        score = base_score + weighted_score

        scored_chunks.append((score, chunk))

    # 按分数降序排序，保持同分数的相对顺序
    scored_chunks.sort(key=lambda x: (-x[0], x[1]))

    return [chunk for _, chunk in scored_chunks]


def entity_aware_hybrid_retrieve_simple(
    query: str,
    vectorstore: Optional[PGVector] = None,
    graph: Optional[nx.DiGraph] = None,
    vector_top_k: int = 8,
    keyword_top_k: int = 8,
    graph_max: int = 8,
    min_entity_matches: int = 1,
) -> List[RetrievedChunk]:
    """
    简化的实体感知混合检索。
    """
    import networkx as nx

    # 1. 提取实体
    entities = extract_entities_simple(query)

    # 2. 执行混合检索（扩大召回）
    mult = 4  # 扩大4倍召回
    vk = max(1, vector_top_k * mult)
    kk = max(1, keyword_top_k * mult)

    q_ret = expand_query_for_hybrid(query)

    vector_chunks: List[RetrievedChunk] = []
    if vectorstore is not None:
        vector_chunks = vector_search(vectorstore, q_ret, top_k=vk)

    keyword_chunks = keyword_search(q_ret, top_k=kk)
    bm25_chunks: List[RetrievedChunk] = []
    if BM25_ENABLED:
        bm_k = max(1, BM25_TOP_K * mult)
        bm25_chunks = bm25_search(q_ret, top_k=bm_k)

    # RRF 融合
    doc_chunks = fuse_doc_chunks_rrf(vector_chunks, keyword_chunks, bm25_chunks, k=RRF_K)

    # 3. 实体过滤
    doc_chunks = filter_chunks_by_entities_simple(doc_chunks, entities, min_entity_matches)

    # 4. 实体加权
    doc_chunks = score_chunks_by_entities_simple(doc_chunks, entities)

    # 5. 锚点加权
    doc_chunks = boost_chunks_by_query_anchors(doc_chunks, query)

    # 6. 重排序（如果启用）
    if RERANK_ENABLED and doc_chunks:
        cap_default = max(1, vector_top_k + keyword_top_k + (BM25_TOP_K if BM25_ENABLED else 0))
        cap = min(cap_default, RERANK_DOC_CAP) if (RERANK_DOC_CAP and RERANK_DOC_CAP > 0) else cap_default
        doc_chunks = rerank_doc_chunks(query, doc_chunks, top_n=cap)

    # 7. 截断到目标数量
    doc_chunks = doc_chunks[:vector_top_k + keyword_top_k]

    # 8. 图谱检索
    graph_chunks: List[RetrievedChunk] = []
    if graph is not None:
        graph_chunks = graph_search(graph, query, max_neighbors=graph_max)

    # 9. 合并去重
    seen_content: set[str] = {c.content for c in doc_chunks}
    merged: List[RetrievedChunk] = list(doc_chunks)
    for c in graph_chunks:
        if c.content not in seen_content:
            seen_content.add(c.content)
            merged.append(c)

    return merged


# 类型注解
from typing import Any