#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check CRUD-RAG data structure"""

import json

data = json.load(open('CRUD_RAG/data/crud_split/split_merged.json', 'r', encoding='utf-8'))
qa = data['questanswer_1doc']

print(f'Type: {type(qa)}')
print(f'Length: {len(qa) if isinstance(qa, list) else "N/A"}')
print(f'First sample keys: {list(qa[0].keys()) if isinstance(qa, list) and len(qa) > 0 else "N/A"}')
print(f'First sample:')
if isinstance(qa, list) and len(qa) > 0:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sample = qa[0]
    print(f'  ID: {sample["ID"]}')
    print(f'  Event: {sample["event"]}')
    print(f'  Questions: {sample["questions"]}')
    print(f'  Answers: {sample["answers"]}')
    print(f'  News1 (first 200 chars): {sample["news1"][:200]}')