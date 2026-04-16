# 混合检索模块：向量检索 + 图谱检索，结果融合去重
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx

from typing import TYPE_CHECKING, Any

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore

from .config import (
    BM25_ENABLED,
    BM25_TOP_K,
    FUSION_METHOD,
    RERANK_DOC_CAP,
    RERANK_ENABLED,
    RERANK_RECALL_MULT,
    RETRIEVAL_DISTINCTIVE_STRICT,
    RRF_K,
)
from .pg_db import bm25_search as pg_bm25_search
from .pg_db import keyword_search as pg_keyword_search


@dataclass
class RetrievedChunk:
    """单条检索结果，统一表示向量或图谱来源。"""
    content: str
    source: str  # 如 "文档A.pdf 第3页" 或 "知识图谱: 实体X - 关系 - 实体Y"
    score: Optional[float] = None  # 向量相似度时可用
    source_type: str = "vector"  # "vector" | "graph"

def normalize_query_for_search(query: str) -> str:
    """
    归一化查询串，减少英文/数字 token 因空格/连字符被“拆开”导致的误匹配。

    例：
      - "Sky Campus" / "Sky-Campus" / "Sky_Campus" -> "SkyCampus"
      - "AI OPS" -> "AIOPS"
    """
    q = (query or "").strip()
    if not q:
        return ""
    # 把 ASCII 字母/数字之间的空白、下划线、连字符合并掉（重复替换直到稳定）
    prev = None
    while prev != q:
        prev = q
        q = re.sub(r"(?i)([a-z0-9])[\s_-]+([a-z0-9])", r"\1\2", q)
    return q


def query_stat_numbers(query: str) -> list[str]:
    """
    问题里的统计类数字（小数、三位及以上整数），用于查询扩展与 chunk 匹配加权。
    排除单独出现的 20xx 年份，避免所有报道都被动命中。
    """
    q = query or ""
    found: list[str] = []
    for m in re.finditer(r"\d+\.\d+", q):
        found.append(m.group(0))
    for m in re.finditer(r"\d{3,}", q):
        s = m.group(0)
        if re.fullmatch(r"20\d{2}", s):
            continue
        found.append(s)
    out: list[str] = []
    seen: set[str] = set()
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:12]


# 演示/业务里「英文产品名 ↔ 中文文档用词」常见不一致，用于字面过滤扩展（小写键 = 主词小写）
_LEXICAL_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "skycampus": (
        "sky campus",
        "sky-campus",
        "智慧校园",
        "智慧校园与设备管理",
        "智慧校园与设备",
    ),
}


def query_distinctive_lexical_tokens(query: str) -> list[str]:
    """
    从问题中提取「强区分度」字面串，用于锚点加权与误召回过滤。
    典型：英文 PascalCase 产品名（SkyCampus），避免只靠「系统/模块」等泛词命中教材。
    """
    out: list[str] = []
    seen: set[str] = set()
    # 同时扫原文与归一化串，避免 Sky Campus 被拆开后抽不到驼峰
    for qq in {query or "", normalize_query_for_search(query or "")}:
        # SkyCampus、OpenStack 等：至少两段大小写驼峰且总长≥5
        for m in re.finditer(r"(?<![A-Za-z0-9])([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", qq):
            t = m.group(1)
            if len(t) >= 5 and t not in seen:
                seen.add(t)
                out.append(t)
    return out[:8]


def distinctive_filter_needles(query: str) -> list[str]:
    """用于子串匹配的 needle 列表（小写），含别名扩展。"""
    needles: list[str] = []
    seen: set[str] = set()
    for t in query_distinctive_lexical_tokens(query):
        k = t.lower()
        for n in (k,) + tuple(x.lower() for x in _LEXICAL_QUERY_ALIASES.get(k, ())):
            if n and n not in seen:
                seen.add(n)
                needles.append(n)
    return needles


def expand_query_for_hybrid(query: str) -> str:
    """
    为混合检索拼接问题中的关键短语（书名、引号内短语、日期、XX市），
    便于向量/BM25/关键词更盯住具体实体，减轻同主题不同报道混淆。
    """
    q = (query or "").strip()
    if not q:
        return q
    spans: list[str] = []
    for pat in (r"《[^》]{1,120}》", r"「[^」]{1,120}」", r"“[^”]{1,120}”"):
        for m in re.finditer(pat, q):
            spans.append(m.group(0))
    for m in re.finditer(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}[日号])?", q):
        spans.append(m.group(0))
    for t in query_distinctive_lexical_tokens(q):
        spans.append(t)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    for n in query_stat_numbers(q):
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    if not uniq:
        return q
    return q + "\n" + " ".join(uniq[:20])


def query_anchor_phrases(query: str) -> list[str]:
    """从问题中提取用于与 chunk 字面匹配的锚点（书名、日期、地级市名等）。"""
    q = query or ""
    anchors: list[str] = []
    for pat in (r"《[^》]{1,120}》", r"「[^」]{1,120}」"):
        for m in re.finditer(pat, q):
            anchors.append(m.group(0))
    for m in re.finditer(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}[日号])?", q):
        anchors.append(m.group(0))
    for m in re.finditer(r"[\u4e00-\u9fff]{2,10}市", q):
        anchors.append(m.group(0))
    anchors.extend(query_distinctive_lexical_tokens(q))
    out: list[str] = []
    seen: set[str] = set()
    for a in anchors:
        if a not in seen and len(a) >= 2:
            seen.add(a)
            out.append(a)
    return out


def query_match_tokens(query: str) -> list[str]:
    """锚点短语 + 统计数字 + 专有词字面，供融合后重排。"""
    seen: set[str] = set()
    out: list[str] = []
    for t in query_anchor_phrases(query) + query_stat_numbers(query):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def filter_doc_chunks_by_distinctive_lexical(
    query: str,
    chunks: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """
    若问题含 PascalCase 等强区分词，则只保留正文或来源中含该词（及配置别名）的文档块，
    避免「系统/模块」泛词误召回无关教材。
    默认在过滤为空时不回退（见 RETRIEVAL_DISTINCTIVE_STRICT）；否则回退为原列表。
    """
    needles = distinctive_filter_needles(query)
    if not needles or not chunks:
        return chunks
    filtered: List[RetrievedChunk] = []
    for c in chunks:
        hay = ((c.content or "") + "\n" + (c.source or "")).lower()
        if any(n in hay for n in needles):
            filtered.append(c)
    if filtered:
        return filtered
    if RETRIEVAL_DISTINCTIVE_STRICT:
        return []
    return chunks


def boost_chunks_by_query_anchors(
    chunks: List[RetrievedChunk],
    query: str,
) -> List[RetrievedChunk]:
    """融合后按「chunk 含问题锚点/统计数字数量」升权重排，稳定排序保留同分相对顺序。"""
    tokens = query_match_tokens(query)
    if not tokens or not chunks:
        return chunks

    def hit_count(c: RetrievedChunk) -> int:
        body = c.content or ""
        return sum(1 for t in tokens if t in body)

    indexed = list(enumerate(chunks))
    indexed.sort(key=lambda ic: (-hit_count(ic[1]), ic[0]))
    return [c for _, c in indexed]


def vector_search(
    vectorstore: PGVector,
    query: str,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """在 PostgreSQL(pgvector) 中做相似度检索，返回 Top-K 文本块及来源。"""
    if vectorstore is None:
        return []
    q = normalize_query_for_search(query)
    docs = vectorstore.similarity_search_with_relevance_scores(q, k=top_k)
    return [
        RetrievedChunk(
            content=d.page_content,
            source=d.metadata.get("source", "未知文档"),
            score=score,
            source_type="vector",
        )
        for d, score in docs
    ]


def keyword_search(query: str, top_k: int = 5) -> List[RetrievedChunk]:
    """在 PostgreSQL 中做关键词/字符串相似检索（用于对比实验）。"""
    q = normalize_query_for_search(query)
    rows = pg_keyword_search(q, top_k=top_k)
    return [
        RetrievedChunk(
            content=r.content,
            source=r.source,
            score=r.score,
            source_type="keyword",
        )
        for r in rows
    ]


def bm25_search(query: str, top_k: int = 5) -> List[RetrievedChunk]:
    """在 PostgreSQL 中做 BM25-like 全文检索（tsvector + ts_rank_cd）。"""
    q = normalize_query_for_search(query)
    rows = pg_bm25_search(q, top_k=top_k)
    return [
        RetrievedChunk(
            content=r.content,
            source=r.source,
            score=r.score,
            source_type="bm25",
        )
        for r in rows
    ]


def extract_query_keywords(query: str, min_len: int = 2) -> List[str]:
    """从用户问题中提取关键词（简单按长度与停用过滤）。"""
    stop = {
        "的", "是", "在", "和", "与", "或", "什么", "如何", "怎样", "哪些", "吗", "呢", "了",
        "说说", "介绍", "一下", "关系", "有关", "关于",
    }
    q0 = (query or "").strip()
    q = normalize_query_for_search(q0)
    # 同时保留原 query 与归一化 query 的 token，避免过拟合某一种写法
    words = re.findall(r"[\u4e00-\u9fff\w]+", q0) + re.findall(r"[\u4e00-\u9fff\w]+", q)
    out: List[str] = []
    seen = set()
    for w in words:
        if not w:
            continue
        # 对“操作系统和cpu的关系”这类长串，按连接词再切一次，避免只保留整句 token。
        parts = [p for p in re.split(r"[的和与及、,，\s]+", w) if p]
        for p in parts:
            if len(p) < min_len:
                continue
            if p in stop:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def graph_search(
    G: nx.DiGraph,
    query: str,
    max_neighbors: int = 10,
    max_relation_length: int = 3,
) -> List[RetrievedChunk]:
    """
    用问题中的关键词在图谱中找相关节点及其邻居，转为结构化文本片段。
    """
    keywords = extract_query_keywords(query)
    if not keywords or G.number_of_nodes() == 0:
        return []

    seen = set()
    chunks: List[RetrievedChunk] = []

    keywords_l = [kw.lower() for kw in keywords]
    def edge_match_score(u: str, rel: str, v: str) -> int:
        txt = f"{u} {rel} {v}".lower()
        return sum(1 for kw in keywords_l if kw and kw in txt)

    for node in G.nodes():
        node_str = str(node).lower() if isinstance(node, str) else ""
        if not any(kw in node_str or node_str in kw for kw in keywords_l):
            continue
        edge_rows: List[tuple[str, str, str]] = []
        # 出边
        for u, v, data in list(G.out_edges(node, data=True)):
            rel = data.get("relation", "相关")
            edge_rows.append((str(u), str(rel), str(v)))
        # 入边
        for u, v, data in list(G.in_edges(node, data=True)):
            rel = data.get("relation", "相关")
            edge_rows.append((str(u), str(rel), str(v)))

        # 先按与问题关键词的匹配度排序，再截断，避免关键边被“前几条边”随机淹没。
        edge_rows.sort(key=lambda t: edge_match_score(t[0], t[1], t[2]), reverse=True)
        for u, rel, v in edge_rows[: max_relation_length * 2]:
            text = f"{u} - {rel} - {v}"
            key = (u, rel, v)
            if key not in seen:
                seen.add(key)
                chunks.append(
                    RetrievedChunk(
                        content=text,
                        source=f"知识图谱节点: {u}",
                        source_type="graph",
                    )
                )
        if len(chunks) >= max_neighbors:
            break

    return chunks[:max_neighbors]


def merge_vector_keyword_chunks(
    vector_chunks: List[RetrievedChunk],
    keyword_chunks: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """向量结果在前、关键词在后，按 content 去重（保留先出现的条目）。"""
    seen: set[str] = set()
    out: List[RetrievedChunk] = []
    for c in vector_chunks + keyword_chunks:
        if c.content not in seen:
            seen.add(c.content)
            out.append(c)
    return out


def fuse_doc_chunks_rrf(
    vector_chunks: List[RetrievedChunk],
    keyword_chunks: List[RetrievedChunk],
    bm25_chunks: List[RetrievedChunk],
    k: int = 60,
) -> List[RetrievedChunk]:
    """RRF 融合三路结果，按 content 去重并累计名次分数。"""
    grouped: Dict[str, RetrievedChunk] = {}
    scores: Dict[str, float] = {}

    streams = [vector_chunks, keyword_chunks, bm25_chunks]
    for stream in streams:
        for rank, chunk in enumerate(stream, start=1):
            key = chunk.content
            if key not in grouped:
                grouped[key] = chunk
            scores[key] = scores.get(key, 0.0) + (1.0 / float(k + rank))

    ordered = sorted(grouped.keys(), key=lambda x: scores.get(x, 0.0), reverse=True)
    return [grouped[k_] for k_ in ordered]


def hybrid_retrieve(
    query: str,
    vectorstore: Optional[PGVector] = None,
    graph: Optional[nx.DiGraph] = None,
    vector_top_k: int = 5,
    keyword_top_k: int = 5,
    graph_max: int = 5,
) -> List[RetrievedChunk]:
    """
    混合检索：向量检索 + 关键词检索(pg_trgm) + BM25-like + 图谱检索，合并去重后返回。
    去重按 content 近似（此处简化为精确 content 去重）。
    若开启 RERANK_ENABLED：向量/关键词召回量按 RERANK_RECALL_MULT 放大，交叉编码器重排后
    仅保留至多 `cap` 条文档块（默认 vector_top_k + keyword_top_k；可用 RERANK_DOC_CAP 覆盖），
    再拼接图谱结果。
    """
    mult = RERANK_RECALL_MULT if RERANK_ENABLED else 1
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

    if FUSION_METHOD == "rrf":
        doc_chunks = fuse_doc_chunks_rrf(vector_chunks, keyword_chunks, bm25_chunks, k=RRF_K)
    else:
        # fallback: keep previous behavior, then append BM25 and deduplicate by content
        doc_chunks = merge_vector_keyword_chunks(vector_chunks, keyword_chunks)
        doc_chunks = merge_vector_keyword_chunks(doc_chunks, bm25_chunks)

    doc_chunks = filter_doc_chunks_by_distinctive_lexical(query, doc_chunks)
    doc_chunks = boost_chunks_by_query_anchors(doc_chunks, query)

    if RERANK_ENABLED and doc_chunks:
        from .reranker import rerank_doc_chunks

        cap_default = max(1, vector_top_k + keyword_top_k + (BM25_TOP_K if BM25_ENABLED else 0))
        cap = min(cap_default, RERANK_DOC_CAP) if (RERANK_DOC_CAP and RERANK_DOC_CAP > 0) else cap_default
        doc_chunks = rerank_doc_chunks(query, doc_chunks, top_n=cap)

    graph_chunks: List[RetrievedChunk] = []
    if graph is not None:
        graph_chunks = graph_search(graph, query, max_neighbors=graph_max)

    seen_content: set[str] = {c.content for c in doc_chunks}
    merged: List[RetrievedChunk] = list(doc_chunks)
    for c in graph_chunks:
        if c.content not in seen_content:
            seen_content.add(c.content)
            merged.append(c)

    return merged
