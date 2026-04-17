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
    import pprint
    pprint.pprint(qa[0], width=100)