from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from .config import DATA_DIR, KB_NAMESPACE


DOC_INDEX_PATH = DATA_DIR / f"docs_index_{KB_NAMESPACE}.json"


@dataclass
class DocEntry:
    id: str
    name: str
    doc_type: Literal["file", "url"]
    source: str  # "upload" | "url"
    created_at: str


def _ensure_parent():
    DOC_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_docs() -> List[DocEntry]:
    _ensure_parent()
    if not DOC_INDEX_PATH.exists():
        return []
    try:
        data = json.loads(DOC_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    docs: List[DocEntry] = []
    for item in data:
        try:
            docs.append(
                DocEntry(
                    id=item["id"],
                    name=item["name"],
                    doc_type=item.get("doc_type", "file"),
                    source=item.get("source", "upload"),
                    created_at=item.get("created_at", ""),
                )
            )
        except Exception:
            continue
    return docs


def save_docs(docs: List[DocEntry]) -> None:
    _ensure_parent()
    DOC_INDEX_PATH.write_text(
        json.dumps([asdict(d) for d in docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_docs(names: List[str], doc_type: str = "file", source: str = "upload") -> None:
    docs = load_docs()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_type_fixed: Literal["file", "url"] = "url" if doc_type == "url" else "file"
    for n in names:
        docs.append(
            DocEntry(
                id=str(uuid.uuid4()),
                name=n,
                doc_type=doc_type_fixed,
                source=source,
                created_at=now,
            )
        )
    save_docs(docs)


def remove_doc_by_name(name: str) -> None:
    docs = load_docs()
    docs = [d for d in docs if d.name != name]
    save_docs(docs)


def find_doc(name: str) -> Optional[DocEntry]:
    for d in load_docs():
        if d.name == name:
            return d
    return None

