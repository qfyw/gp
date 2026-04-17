#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check database status"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pg_db import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM langchain_pg_collection WHERE name = %s', ('crud_eval',))
count = cur.fetchone()[0]
print(f'Collection count: {count}')

if count > 0:
    cur.execute('SELECT uuid FROM langchain_pg_collection WHERE name = %s', ('crud_eval',))
    cid = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s', (cid,))
    emb_count = cur.fetchone()[0]
    print(f'Embedding count: {emb_count}')

    cur.execute('SELECT COUNT(*) FROM rag_keyword_chunks WHERE namespace = %s', ('crud_eval',))
    kw_count = cur.fetchone()[0]
    print(f'Keyword chunks count: {kw_count}')

    cur.execute('SELECT COUNT(*) FROM docs_index WHERE namespace = %s', ('crud_eval',))
    docs_count = cur.fetchone()[0]
    print(f'Docs index count: {docs_count}')

conn.close()