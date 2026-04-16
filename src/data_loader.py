# 知识获取模块：文档解析、文本切片、向量化、简易知识图谱构建
from __future__ import annotations

import io
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore

from .generator import get_llm
from .config import (
    DATA_DIR,
    EMBEDDING_LOCAL_ONLY,
    EMBEDDING_MODEL,
    INGEST_CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    KG_LLM_INGEST_RATE,
    KG_PERSIST_PATH,
    PROJECT_ROOT,
    TAVILY_API_KEY,
    WEB_FETCH_VERIFY_SSL,
)
from .doc_store import add_docs
from .pg_db import upsert_keyword_chunks, ensure_db_objects


@dataclass
class IngestStats:
    """上传/入库结果摘要，便于界面提示（例如扫描版 PDF 无文字层）。"""

    chunk_count: int
    """可写入向量库与关键词表的非空文本块数量。"""
    files_no_extractable_text: List[str]
    """解析后完全没有抽出文字的原始文件名列表。"""


# ---------- 文档解析 ----------
# 单份 PDF 最多解析页数，避免异常大文件卡死
PDF_MAX_PAGES = 500


def _extract_text_from_pdf_pypdf2(file_content: bytes, filename: str) -> List[Tuple[str, str]]:
    """用 PyPDF2 提取，失败或无文字时返回空列表。"""
    import PyPDF2

    out: List[Tuple[str, str]] = []
    pdf = PyPDF2.PdfReader(io.BytesIO(file_content))
    n = min(len(pdf.pages), PDF_MAX_PAGES)
    for i in range(n):
        try:
            text = (pdf.pages[i].extract_text() or "").strip()
            if text:
                source = f"{filename} 第{i + 1}页" if filename else f"PDF 第{i + 1}页"
                out.append((text, source))
        except Exception:
            continue
    return out


def _extract_text_from_pdf_pymupdf(file_content: bytes, filename: str) -> List[Tuple[str, str]]:
    """用 PyMuPDF 提取（对扫描版/复杂版式兼容更好）。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    out: List[Tuple[str, str]] = []
    doc = fitz.open(stream=file_content, filetype="pdf")
    try:
        n = min(len(doc), PDF_MAX_PAGES)
        for i in range(n):
            page = doc[i]
            text = (page.get_text() or "").strip()
            if text:
                source = f"{filename} 第{i + 1}页" if filename else f"PDF 第{i + 1}页"
                out.append((text, source))
    finally:
        doc.close()
    return out


def _looks_mojibake(text: str) -> bool:
    """
    经验性判定：PyPDF2 在某些 CMap/编码（如 GBK-EUC-H）下会抽出类似 latin-1 的乱码。
    若文本中几乎没有中文，却充满高位/控制字符，则判为“疑似乱码”，触发回退提取或纠错。
    """
    s = (text or "").strip()
    if len(s) < 60:
        return False
    # 中文比例过低
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    if cjk / max(1, len(s)) > 0.02:
        return False
    # 乱码常见特征：大量 0x80-0xFF 范围字符（在 Python str 中表现为 U+0080~U+00FF）
    high = sum(1 for ch in s if "\u0080" <= ch <= "\u00ff")
    # 以及一些非常规控制字符
    ctrl = sum(1 for ch in s if ord(ch) < 9 or (13 < ord(ch) < 32))
    return (high / max(1, len(s)) > 0.08) or (ctrl > 10)


def _fix_gbk_mojibake(text: str) -> str:
    """
    尝试修复典型“latin-1/CP1252 误解码的 GBK 文本”。
    只在修复结果明显更像中文时才替换。
    """
    s = (text or "").strip()
    if not s:
        return s
    try:
        repaired = s.encode("latin1", errors="ignore").decode("gbk", errors="ignore").strip()
    except Exception:
        return s
    if not repaired:
        return s
    # 若修复后中文比例显著提升，则采用修复文本
    cjk_before = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    cjk_after = sum(1 for ch in repaired if "\u4e00" <= ch <= "\u9fff")
    if cjk_after >= max(5, cjk_before * 3):
        return repaired
    return s


def extract_text_from_pdf(file_content: bytes, filename: str = "") -> List[Tuple[str, str]]:
    """
    从 PDF 提取文本，返回 (page_text, source_label) 列表。
    策略：
    - 先用 PyPDF2（快、依赖轻）
    - 若无结果或疑似乱码，则回退 PyMuPDF（对 CMap/复杂版式更稳）
    - 对疑似 GBK 乱码做一次轻量修复
    """
    chunks = _extract_text_from_pdf_pypdf2(file_content, filename)
    if chunks:
        # 若多数页疑似乱码，则整体回退 PyMuPDF
        bad = sum(1 for t, _ in chunks if _looks_mojibake(t))
        if bad >= max(1, len(chunks) // 2):
            alt = _extract_text_from_pdf_pymupdf(file_content, filename)
            if alt:
                chunks = alt
        # 对仍疑似乱码的页尝试修复
        fixed: List[Tuple[str, str]] = []
        for t, src in chunks:
            if _looks_mojibake(t):
                t2 = _fix_gbk_mojibake(t)
                fixed.append((t2, src))
            else:
                fixed.append((t, src))
        chunks = fixed
    if not chunks:
        chunks = _extract_text_from_pdf_pymupdf(file_content, filename)
    return chunks


def extract_text_from_docx(file_content: bytes, filename: str = "") -> List[Tuple[str, str]]:
    """从 Word 文档提取文本，返回 (paragraph_block, source_label) 列表。"""
    from docx import Document

    doc = Document(io.BytesIO(file_content))
    blocks: List[Tuple[str, str]] = []
    source = filename or "Word文档"
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            blocks.append((text, source))
    return blocks


def load_document(file_content: bytes, filename: str) -> List[Tuple[str, str]]:
    """根据后缀选择解析器，返回 (text, source) 列表。"""
    suf = Path(filename).suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(file_content, filename)
    if suf in (".docx", ".doc"):
        return extract_text_from_docx(file_content, filename)
    if suf == ".txt":
        text = file_content.decode("utf-8", errors="replace").strip()
        if not text:
            return []
        label = filename or "txt"
        return [(text, label)]
    raise ValueError(f"暂不支持的文件类型: {suf}")


# ---------- 网络地址解析 ----------
def extract_text_from_url(url: str, timeout_s: float = 20.0) -> List[Tuple[str, str]]:
    """
    从网页 URL 抽取正文文本，返回 (text, source_label) 列表。
    说明：该实现以“可用/稳定”为优先，使用 BeautifulSoup 提取段落文本。
    """
    url = (url or "").strip()
    if not url:
        return []
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL 必须以 http:// 或 https:// 开头")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    }
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout_s,
        headers=headers,
        verify=WEB_FETCH_VERIFY_SSL,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "lxml")
    # 移除脚本/样式/无关节点
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # 优先 article/main，其次 body
    container = soup.find("article") or soup.find("main") or soup.body or soup

    texts: List[str] = []
    for p in container.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if t and len(t) >= 20:
            texts.append(t)

    # 回退：如果没有段落，取全部可见文本
    if not texts:
        t = container.get_text("\n", strip=True)
        if t:
            texts = [t]

    merged = "\n".join(texts).strip()
    if not merged:
        return []
    return [(merged, url)]


def _web_search_for_url(url: str, page_summary: str = "", max_results: int = 3) -> List[Tuple[str, str]]:
    """
    上传网页分析时使用的联网搜索：根据 URL 或页面摘要搜索相关结果，返回 (snippet, source_label) 列表。
    未配置 TAVILY_API_KEY 时返回空列表。
    """
    if not TAVILY_API_KEY or not (url or page_summary):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(k=max_results, tavily_api_key=TAVILY_API_KEY)
        query = (page_summary[:80] + "...") if len(page_summary or "") > 80 else (page_summary or url)
        results = tool.invoke({"query": query})
        out: List[Tuple[str, str]] = []
        for item in (results or [])[:max_results]:
            content = (item.get("content") or item.get("snippet") or "").strip()
            title = (item.get("title") or "").strip()
            if content:
                out.append((content[:800], f"网页分析-联网补充: {title or url}"))
        return out
    except Exception:
        return []


# ---------- 文本切片 ----------
def get_text_splitter(
    chunk_size: int = INGEST_CHUNK_SIZE,
    chunk_overlap: int = INGEST_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


# ---------- 简易实体关系抽取 (规则/正则，MVP 用) ----------
# 简单模式：A 是 B、A 与 B 有关、A 包括 B 等
REL_PATTERNS = [
    (re.compile(r"([^\s，。；]+)\s+是\s+([^\s，。；]+)"), "是"),
    (re.compile(r"([^\s，。；]+)\s+与\s+([^\s，。；]+)\s+"), "相关"),
    (re.compile(r"([^\s，。；]+)\s+包括\s+([^\s，。；]+)"), "包括"),
    (re.compile(r"([^\s，。；]+)\s+属于\s+([^\s，。；]+)"), "属于"),
    (re.compile(r"([^\s，。；]+)\s+的\s+([^\s，。；]+)"), "的"),
]


GENERIC_ENTITY_TERMS = {
    "系统", "模块", "功能", "机制", "方法", "问题", "过程", "部分", "结构", "资源"
}

# 主客体过长多为整句误抽，入图会污染 2-hop 推导前缀
KG_ENTITY_MAX_CHARS = 48


def _refine_entity(entity: str, context: str) -> str:
    """
    实体轻量规范化：
    - 去除两端噪声符号
    - 若实体过泛（如“系统”），优先在上下文中回填更具体的复合实体（如“操作系统”）
    """
    e = (entity or "").strip().strip("，。；：、,.!?()[]{}\"'“”‘’")
    if len(e) < 2:
        return ""
    if e in GENERIC_ENTITY_TERMS:
        # 在上下文中找“XX系统/XX模块...”这类复合词，优先最长词
        cands = re.findall(rf"[\u4e00-\u9fffA-Za-z0-9_]{{1,12}}{re.escape(e)}", context or "")
        cands = [c for c in cands if len(c) > len(e)]
        if cands:
            cands.sort(key=len, reverse=True)
            return cands[0]
    return e


def extract_entity_relations(text: str) -> List[Tuple[str, str, str]]:
    """从一段文本中抽取 (Subject, Relation, Object) 三元组。"""
    triples: List[Tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pattern, rel in REL_PATTERNS:
        for m in pattern.finditer(text):
            s0, o0 = m.group(1).strip(), m.group(2).strip()
            s = _refine_entity(s0, text)
            o = _refine_entity(o0, text)
            if len(s) >= 2 and len(o) >= 2 and s != o:  # 过滤过短或相同
                t = (s, rel, o)
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
    return triples


def _looks_relation_rich(text: str) -> bool:
    """
    轻量候选判定：文本是否“可能包含实体关系描述”。
    用于两阶段抽取：先规则抽，若没抽到再对候选段落调用 LLM。
    """
    s = (text or "").strip()
    if len(s) < 30:
        return False
    # 关系触发词（尽量偏通用，避免每段都触发）
    triggers = (
        "是",
        "属于",
        "包括",
        "导致",
        "引起",
        "位于",
        "负责",
        "关系",
        "关联",
        "原因",
        "影响",
        "由",
        "包含",
    )
    if any(t in s for t in triggers):
        return True
    # 轻量规则：出现多个专名样式（中文/数字字母连续串）也可能适合抽取
    words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", s)
    return len(words) >= 8


def llm_extract_entity_relations(
    text: str,
    *,
    llm=None,
    max_triples: int = 30,
) -> List[Tuple[str, str, str]]:
    """
    使用大模型从文本中抽取 (Subject, Relation, Object) 三元组。
    - 输出要求为 JSON，便于稳定解析
    - 失败/解析异常时返回空列表，由上层回退到规则抽取
    """
    llm = llm or get_llm()
    if llm is None:
        return []

    # 控制 token：只取前一段做抽取（图谱主要用于关系增强，不需要逐字覆盖）
    snippet = (text or "").strip()
    if not snippet:
        return []
    snippet = snippet[:2500]

    prompt = f"""你是信息抽取助手。请从给定文本中抽取“实体-关系-实体”的三元组。

要求：
1) 只输出 JSON 数组，不要输出任何解释或多余文本
2) 数组元素是对象，字段必须为: s, r, o
3) r 用尽量简短的中文动词/关系短语（如“属于/包含/导致/负责/位于/是/相关”）
4) 去除无意义的代词（如“它/他们/本公司”），实体尽量具体；s 与 o 须为简短名词短语（建议各不超过 16 字），禁止把整句、半句、标点串作主客体
5) 最多输出 {max_triples} 条；没有则输出 []

文本：
{snippet}
"""
    try:
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = (raw or "").strip()
        import json

        data = json.loads(raw)
        out: List[Tuple[str, str, str]] = []
        if isinstance(data, list):
            for item in data[:max_triples]:
                if not isinstance(item, dict):
                    continue
                s = _refine_entity(str(item.get("s", "")).strip(), snippet)
                r = str(item.get("r", "")).strip()
                o = _refine_entity(str(item.get("o", "")).strip(), snippet)
                if len(s) >= 2 and len(o) >= 2 and s != o and r:
                    out.append((s, r[:20], o))
        return out[:max_triples]
    except Exception:
        return []


def extract_entity_relations_auto(text: str) -> List[Tuple[str, str, str]]:
    """
    两阶段知识图谱三元组抽取（更快）：
    1) 先用规则正则快速抽取；若抽到则直接返回（不调用 LLM）
    2) 若规则未抽到：仅当文本“疑似含关系描述”时，按 KG_LLM_INGEST_RATE 概率调用 LLM；
       LLM 未配置/失败/返回空则返回空（不再强行回退规则，避免无意义开销）。
    """
    # Stage 1: fast rule extraction
    rule_triples = extract_entity_relations(text)
    if rule_triples:
        return rule_triples

    # Stage 2: LLM extraction only on candidates
    if not _looks_relation_rich(text):
        return []

    rate = KG_LLM_INGEST_RATE
    try_llm = rate >= 1.0 or (rate > 0.0 and random.random() < rate)
    if not try_llm:
        return []

    llm_triples = llm_extract_entity_relations(text)
    return llm_triples or []


# ---------- 向量化并存入 PostgreSQL(pgvector) ----------
_hf_embeddings_singleton: HuggingFaceEmbeddings | None = None


def get_embeddings():
    """获取 HuggingFace 嵌入模型（单例，避免批量入库时重复加载权重）。"""
    global _hf_embeddings_singleton
    if _hf_embeddings_singleton is not None:
        return _hf_embeddings_singleton
    model_kwargs: dict = {"device": "cpu"}
    if EMBEDDING_LOCAL_ONLY or (isinstance(EMBEDDING_MODEL, str) and Path(EMBEDDING_MODEL).exists()):
        model_kwargs["local_files_only"] = True
    _hf_embeddings_singleton = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    return _hf_embeddings_singleton


def build_vectorstore(
    texts: List[str],
    metadatas: List[dict],
    collection_name: str = "rag_docs",
    persist_directory: str | Path | None = None,
) -> PGVector:
    """将文本列表向量化并存入 PostgreSQL(pgvector)，返回 VectorStore。"""
    # persist_directory 仅为兼容旧接口，pgvector 不需要
    ensure_db_objects()
    from .config import POSTGRES_DSN, POSTGRES_SQLALCHEMY_URL

    if not (POSTGRES_DSN or "").strip():
        raise RuntimeError("未配置 POSTGRES_DSN（.env），无法写入 PostgreSQL/pgvector。")

    embeddings = get_embeddings()
    # 无块时禁止 from_texts：SQLAlchemy 会对空 VALUES 生成 INSERT DEFAULT VALUES，触发 id NOT NULL。
    if not texts:
        return PGVector.from_existing_index(
            embedding=embeddings,
            connection=POSTGRES_SQLALCHEMY_URL,
            collection_name=collection_name,
            use_jsonb=True,
        )
    return PGVector.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        connection=POSTGRES_SQLALCHEMY_URL,
        use_jsonb=True,
    )


def add_documents_to_vectorstore(
    vectorstore: PGVector,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """向已有 PGVector 集合追加文档。"""
    vectorstore.add_texts(texts=texts, metadatas=metadatas)


# ---------- 知识图谱 (NetworkX) ----------
def build_or_load_graph(path: str | Path | None = None) -> nx.DiGraph:
    """加载已有图或返回新的有向图。"""
    path = path or KG_PERSIST_PATH
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph, path: str | Path | None = None) -> None:
    path = path or KG_PERSIST_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def add_triples_to_graph(G: nx.DiGraph, triples: List[Tuple]) -> None:
    """
    写入三元组到图谱。
    triples 支持两种形态：
    - (s, r, o)
    - (s, r, o, doc_name)  # provenance：来自哪个文档/URL（用于后续按文档删除图谱边）
    """
    for t in triples:
        if not t or len(t) < 3:
            continue
        s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
        doc = str(t[3]).strip() if len(t) >= 4 and t[3] is not None else ""
        if not s or not o or not r:
            continue
        if len(s) > KG_ENTITY_MAX_CHARS or len(o) > KG_ENTITY_MAX_CHARS:
            continue

        G.add_node(s, label=s)
        G.add_node(o, label=o)

        if G.has_edge(s, o):
            data = G.get_edge_data(s, o) or {}
            # relation：尽量保留第一个；若不同则拼接（避免丢信息）
            rel0 = str(data.get("relation", "") or "").strip()
            if rel0 and rel0 != r and r not in rel0.split("|"):
                data["relation"] = rel0 + "|" + r
            elif not rel0:
                data["relation"] = r
            # provenance sources：集合化
            if doc:
                srcs = data.get("sources")
                if isinstance(srcs, list):
                    if doc not in srcs:
                        srcs.append(doc)
                    data["sources"] = srcs
                elif isinstance(srcs, set):
                    srcs.add(doc)
                    data["sources"] = srcs
                elif isinstance(srcs, str) and srcs:
                    data["sources"] = list({srcs, doc})
                else:
                    data["sources"] = [doc]
            G.add_edge(s, o, **data)
        else:
            edge_data = {"relation": r}
            if doc:
                edge_data["sources"] = [doc]
            G.add_edge(s, o, **edge_data)


# ---------- 主流程：上传文件 -> 解析、切片、向量化、图谱 ----------
def process_uploaded_files(
    files: List[Tuple[bytes, str]],
    collection_name: str = "rag_docs",
    chunk_size: int = INGEST_CHUNK_SIZE,
    chunk_overlap: int = INGEST_CHUNK_OVERLAP,
    existing_vectorstore: PGVector | None = None,
    existing_graph: nx.DiGraph | None = None,
) -> Tuple[PGVector, nx.DiGraph, IngestStats]:
    """
    处理上传的文件：解析 -> 切片 -> 向量化入 PostgreSQL(pgvector)，实体关系入 NetworkX。
    若传入 existing_vectorstore / existing_graph 则在其上追加；否则新建。
    返回 (vectorstore, knowledge_graph, ingest_stats)。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_texts: List[str] = []
    all_metadatas: List[dict] = []
    all_triples: List[Tuple[str, str, str, str]] = []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    filenames_for_index: List[str] = []
    files_no_extractable_text: List[str] = []
    for file_content, filename in files:
        raw_blocks = load_document(file_content, filename)
        if not raw_blocks:
            files_no_extractable_text.append(filename)
        else:
            filenames_for_index.append(filename)
        for block_text, source in raw_blocks:
            triples = extract_entity_relations_auto(block_text)
            # provenance：使用“文档名/URL”，与 docs_index 中的 name 一致，便于删除时匹配
            all_triples.extend([(s, r, o, filename) for (s, r, o) in triples])
            chunks = splitter.split_text(block_text)
            for ch in chunks:
                if ch.strip():
                    all_texts.append(ch)
                    all_metadatas.append({"source": source, "filename": filename})

    from .config import POSTGRES_DSN, POSTGRES_SQLALCHEMY_URL

    if not (POSTGRES_DSN or "").strip():
        raise RuntimeError("未配置 POSTGRES_DSN（.env），无法写入 PostgreSQL/pgvector。")

    ensure_db_objects()
    if all_texts:
        if existing_vectorstore is not None:
            existing_vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)
            vectorstore = existing_vectorstore
        else:
            vectorstore = PGVector.from_texts(
                texts=all_texts,
                embedding=get_embeddings(),
                metadatas=all_metadatas,
                collection_name=collection_name,
                connection=POSTGRES_SQLALCHEMY_URL,
                use_jsonb=True,
            )
    else:
        if existing_vectorstore is not None:
            vectorstore = existing_vectorstore
        else:
            vectorstore = PGVector.from_existing_index(
                embedding=get_embeddings(),
                connection=POSTGRES_SQLALCHEMY_URL,
                collection_name=collection_name,
                use_jsonb=True,
            )

    # 同步写入关键词检索表（用于 keyword search 对比实验）
    upsert_keyword_chunks(
        [
            {"content": t, **(m or {})}
            for t, m in zip(all_texts, all_metadatas)
            if (t or "").strip()
        ]
    )

    G = existing_graph if existing_graph is not None else build_or_load_graph()
    add_triples_to_graph(G, all_triples)
    save_graph(G)

    if filenames_for_index:
        add_docs(filenames_for_index, doc_type="file", source="upload")

    stats = IngestStats(
        chunk_count=len(all_texts),
        files_no_extractable_text=files_no_extractable_text,
    )
    return vectorstore, G, stats


def process_web_urls(
    urls: List[str],
    collection_name: str = "rag_docs",
    chunk_size: int = INGEST_CHUNK_SIZE,
    chunk_overlap: int = INGEST_CHUNK_OVERLAP,
    existing_vectorstore: PGVector | None = None,
    existing_graph: nx.DiGraph | None = None,
) -> Tuple[PGVector, nx.DiGraph]:
    """
    处理网页 URL：抓取 -> 抽取文本 -> 切片 -> 向量化入 PostgreSQL(pgvector)，实体关系入 NetworkX。
    返回 (vectorstore, knowledge_graph)。
    """
    url_list = [u.strip() for u in (urls or []) if u and u.strip()]
    if not url_list:
        raise ValueError("未提供有效 URL")

    all_texts: List[str] = []
    all_metadatas: List[dict] = []
    all_triples: List[Tuple[str, str, str, str]] = []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for url in url_list:
        blocks = extract_text_from_url(url)
        first_text = ""
        for block_text, source in blocks:
            if not first_text:
                first_text = block_text[:200] if block_text else ""
            triples = extract_entity_relations_auto(block_text)
            all_triples.extend([(s, r, o, url) for (s, r, o) in triples])
            chunks = splitter.split_text(block_text)
            for ch in chunks:
                if ch.strip():
                    all_texts.append(ch)
                    all_metadatas.append({"source": source, "url": url, "filename": url})
        # 上传网页分析时使用联网搜索，补充与该页相关的检索结果
        for snippet, search_source in _web_search_for_url(url, first_text):
            if snippet.strip():
                all_texts.append(snippet)
                all_metadatas.append({"source": search_source, "url": url, "filename": url})
                for (s, r, o) in extract_entity_relations_auto(snippet):
                    all_triples.append((s, r, o, url))

    from .config import POSTGRES_DSN, POSTGRES_SQLALCHEMY_URL

    if not (POSTGRES_DSN or "").strip():
        raise RuntimeError("未配置 POSTGRES_DSN（.env），无法写入 PostgreSQL/pgvector。")

    ensure_db_objects()
    if all_texts:
        if existing_vectorstore is not None:
            existing_vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)
            vectorstore = existing_vectorstore
        else:
            vectorstore = PGVector.from_texts(
                texts=all_texts,
                embedding=get_embeddings(),
                metadatas=all_metadatas,
                collection_name=collection_name,
                connection=POSTGRES_SQLALCHEMY_URL,
                use_jsonb=True,
            )
    else:
        if existing_vectorstore is not None:
            vectorstore = existing_vectorstore
        else:
            vectorstore = PGVector.from_existing_index(
                embedding=get_embeddings(),
                connection=POSTGRES_SQLALCHEMY_URL,
                collection_name=collection_name,
                use_jsonb=True,
            )

    # 同步写入关键词检索表（用于 keyword search 对比实验）
    upsert_keyword_chunks(
        [
            {"content": t, **(m or {})}
            for t, m in zip(all_texts, all_metadatas)
            if (t or "").strip()
        ]
    )

    G = existing_graph if existing_graph is not None else build_or_load_graph()
    add_triples_to_graph(G, all_triples)
    save_graph(G)

    # 以 URL 作为“文档名”登记到知识库索引
    add_docs(url_list, doc_type="url", source="url")

    return vectorstore, G
