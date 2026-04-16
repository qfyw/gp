# 交叉编码器重排序（query-passage），依赖 sentence-transformers 已装包
from __future__ import annotations

import threading
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from .retriever import RetrievedChunk

from .config import RERANK_MAX_PASSAGE_CHARS, RERANK_MODEL

_ce_lock = threading.Lock()
_ce_model: Optional["CrossEncoder"] = None


def get_cross_encoder() -> "CrossEncoder":
    global _ce_model
    with _ce_lock:
        if _ce_model is None:
            from sentence_transformers import CrossEncoder as CE

            _ce_model = CE(RERANK_MODEL, max_length=512)
        return _ce_model


def rerank_doc_chunks(
    query: str,
    chunks: List["RetrievedChunk"],
    top_n: int,
) -> List["RetrievedChunk"]:
    """
    按 query 与 passage 相关性重排，保留 source/source_type/score 等字段，仅重排顺序；
    截断用于打分的正文长度，避免超长段拖慢推理。
    """
    if not chunks or top_n <= 0:
        return []
    q = (query or "").strip()
    if not q:
        return chunks[:top_n]

    model = get_cross_encoder()
    limit = max(256, int(RERANK_MAX_PASSAGE_CHARS))
    pairs: list[tuple[str, str]] = []
    for c in chunks:
        text = (c.content or "").strip()
        if len(text) > limit:
            text = text[:limit]
        pairs.append((q, text))

    scores = model.predict(pairs, show_progress_bar=False)
    indexed = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in indexed[:top_n]]
