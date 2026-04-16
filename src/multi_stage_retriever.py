#!/usr/bin/env python3
"""
多阶段检索模块：
阶段1：扩大召回（top_k=20-30）
阶段2：实体匹配过滤
阶段3：重排序（top_k=8-10）

优化策略：
1. 阶段1：扩大召回，确保不漏掉相关文档
2. 阶段2：实体过滤，只保留包含关键实体的文档
3. 阶段3：精确重排序，确保 Top-K 质量

用法：
  from src.multi_stage_retriever import multi_stage_hybrid_retrieve
  chunks = multi_stage_hybrid_retrieve(query, vectorstore, graph, top_k=10)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .config import (
    BM25_TOP_K,
    BM25_ENABLED,
    RERANK_DOC_CAP,
    RERANK_ENABLED,
    RERANK_RECALL_MULT,
    RRF_K,
)
from .entity_aware_retriever_simple import (
    extract_entities_simple,
    filter_chunks_by_entities_simple,
    score_chunks_by_entities_simple,
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
    vector_search,
)

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore


def multi_stage_hybrid_retrieve(
    query: str,
    vectorstore: Optional[PGVector] = None,
    graph: Optional[Any] = None,
    vector_top_k: int = 8,
    keyword_top_k: int = 8,
    graph_max: int = 8,
    stage1_recall_mult: int = 3,  # 阶段1 召回倍数
    min_entity_matches: int = 1,  # 最小实体匹配数
) -> List[RetrievedChunk]:
    """
    多阶段混合检索：

    阶段1：扩大召回（确保不漏掉）
    - 向量检索：vector_top_k * stage1_recall_mult
    - 关键词检索：keyword_top_k * stage1_recall_mult
    - BM25检索：BM25_TOP_K * stage1_recall_mult
    - RRF融合

    阶段2：实体过滤（只保留相关文档）
    - 提取问题中的关键实体
    - 扩展实体（同义词、别名）
    - 过滤：只保留包含关键实体的文档

    阶段3：精确重排序（确保 Top-K 质量）
    - 实体加权（按实体匹配数排序）
    - 锚点加权（按问题锚点排序）
    - 重排序（BGE-reranker）
    - 截断到目标数量

    阶段4：图谱检索
    - 知识图谱检索（2-hop）

    最终：合并去重
    """
    import networkx as nx

    # ===== 阶段1：扩大召回 =====
    print(f"[阶段1] 扩大召回：top_k={vector_top_k}*{stage1_recall_mult}={vector_top_k * stage1_recall_mult}")
    vk = max(1, vector_top_k * stage1_recall_mult)
    kk = max(1, keyword_top_k * stage1_recall_mult)

    q_ret = expand_query_for_hybrid(query)

    vector_chunks: List[RetrievedChunk] = []
    if vectorstore is not None:
        vector_chunks = vector_search(vectorstore, q_ret, top_k=vk)

    keyword_chunks = pg_keyword_search(q_ret, top_k=kk)
    bm25_chunks: List[RetrievedChunk] = []
    if BM25_ENABLED:
        bm_k = max(1, BM25_TOP_K * stage1_recall_mult)
        bm25_chunks = bm25_search(q_ret, top_k=bm_k)

    # RRF 融合
    doc_chunks = fuse_doc_chunks_rrf(vector_chunks, keyword_chunks, bm25_chunks, k=RRF_K)
    print(f"[阶段1] 检索完成，共 {len(doc_chunks)} 条文档")

    # ===== 阶段2：实体过滤 =====
    print(f"[阶段2] 实体过滤：最小匹配数={min_entity_matches}")

    # 提取实体
    entities = extract_entities_simple(query)
    entity_counts = {k: len(v) for k, v in entities.items()}
    print(f"[阶段2] 提取实体：{entity_counts}")

    # 实体过滤（使用简化版本）
    doc_chunks = filter_chunks_by_entities_simple(doc_chunks, entities, min_entity_matches)
    print(f"[阶段2] 过滤后：{len(doc_chunks)} 条文档")

    # 如果过滤后太少，降低过滤标准
    if len(doc_chunks) < vector_top_k:
        print(f"[阶段2] 过滤后文档不足，降低过滤标准")
        doc_chunks = filter_chunks_by_entities_simple(
            fuse_doc_chunks_rrf(vector_chunks, keyword_chunks, bm25_chunks, k=RRF_K),
            entities,
            min_entity_matches=0  # 不限制
        )
        print(f"[阶段2] 降低过滤标准后：{len(doc_chunks)} 条文档")

    # ===== 阶段3：精确重排序 =====
    print(f"[阶段3] 精确重排序：目标 top_k={vector_top_k + keyword_top_k}")

    # 3.1 实体加权
    doc_chunks = score_chunks_by_entities_simple(doc_chunks, entities)

    # 3.2 锚点加权
    doc_chunks = boost_chunks_by_query_anchors(doc_chunks, query)

    # 3.3 重排序
    if RERANK_ENABLED and doc_chunks:
        cap_default = max(1, vector_top_k + keyword_top_k + (BM25_TOP_K if BM25_ENABLED else 0))
        cap = min(cap_default, RERANK_DOC_CAP) if (RERANK_DOC_CAP and RERANK_DOC_CAP > 0) else cap_default
        doc_chunks = rerank_doc_chunks(query, doc_chunks, top_n=cap)
        print(f"[阶段3] 重排序后：{len(doc_chunks)} 条文档")

    # 3.4 截断
    target_k = vector_top_k + keyword_top_k
    doc_chunks = doc_chunks[:target_k]
    print(f"[阶段3] 最终截断：{len(doc_chunks)} 条文档")

    # ===== 阶段4：图谱检索 =====
    print(f"[阶段4] 图谱检索：max_neighbors={graph_max}")
    graph_chunks: List[RetrievedChunk] = []
    if graph is not None:
        graph_chunks = graph_search(graph, query, max_neighbors=graph_max)

    # ===== 合并去重 =====
    seen_content: set[str] = {c.content for c in doc_chunks}
    merged: List[RetrievedChunk] = list(doc_chunks)
    for c in graph_chunks:
        if c.content not in seen_content:
            seen_content.add(c.content)
            merged.append(c)

    print(f"[最终] 返回 {len(merged)} 条文档（文档+图谱）")

    return merged


# 类型注解
from typing import Any