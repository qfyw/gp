#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Quick check of full_hybrid results"""

import json

data = json.load(open('datasets/optimized_test_15/ragquesteval_evaluation/ragquesteval_comparison_results_50.json', 'r', encoding='utf-8'))
full_hybrid = next(s for s in data['strategies'] if s['strategy'] == 'full_hybrid')

print(f'Full Hybrid (原版):')
print(f'  F1: {full_hybrid["avg_quest_avg_f1"]:.4f} ({full_hybrid["avg_quest_avg_f1"]*100:.2f}%)')
print(f'  Recall: {full_hybrid["avg_quest_recall"]:.4f} ({full_hybrid["avg_quest_recall"]*100:.2f}%)')
print(f'  F1标准差: {full_hybrid["std_quest_avg_f1"]:.4f}')