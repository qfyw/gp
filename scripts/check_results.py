#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check results summary"""

import json

data = json.load(open('datasets/ragquesteval_results_1776.json', 'r', encoding='utf-8'))

print(f'Total samples: {len(data["details"])}')
print(f'Mean F1: {data["quest_avg_f1_mean"]}')
print(f'Mean Recall: {data["quest_recall_mean"]}')
print(f'STD F1: {data["quest_avg_f1_std"]}')
print(f'STD Recall: {data["quest_recall_std"]}')