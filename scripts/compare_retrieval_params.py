#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare results: top_k=8 vs top_k=15"""

import json

# 加载优化后的结果（top_k=15）
with open('datasets/optimized_test_15/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    results_15 = json.load(f)

# 加载之前的结果（top_k=8）
with open('datasets/final_test_50/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    results_8 = json.load(f)

print("="*80)
print("检索参数优化对比（Full Hybrid策略，50样本）")
print("="*80)

print("\n【top_k=8（之前）】")
print(f"  Quest Avg F1: {results_8['quest_avg_f1_mean']:.4f} ({results_8['quest_avg_f1_mean']*100:.2f}%)")
print(f"  Quest Recall: {results_8['quest_recall_mean']:.4f} ({results_8['quest_recall_mean']*100:.2f}%)")
print(f"  F1标准差: {results_8['quest_avg_f1_std']:.4f}")

print("\n【top_k=15（优化后）】")
print(f"  Quest Avg F1: {results_15['quest_avg_f1_mean']:.4f} ({results_15['quest_avg_f1_mean']*100:.2f}%)")
print(f"  Quest Recall: {results_15['quest_recall_mean']:.4f} ({results_15['quest_recall_mean']*100:.2f}%)")
print(f"  F1标准差: {results_15['quest_avg_f1_std']:.4f}")

print("\n" + "="*80)
print("性能变化")
print("="*80)

f1_change = (results_15['quest_avg_f1_mean'] - results_8['quest_avg_f1_mean'])
recall_change = (results_15['quest_recall_mean'] - results_8['quest_recall_mean'])

print(f"F1变化: {f1_change:+.4f} ({f1_change/results_8['quest_avg_f1_mean']*100:+.2f}%)")
print(f"Recall变化: {recall_change:+.4f} ({recall_change/results_8['quest_recall_mean']*100:+.2f}%)")
print(f"F1标准差变化: {(results_15['quest_avg_f1_std'] - results_8['quest_avg_f1_std']):+.4f}")

# 距离70%目标
target_f1 = 0.70
gap_8 = target_f1 - results_8['quest_avg_f1_mean']
gap_15 = target_f1 - results_15['quest_avg_f1_mean']

print(f"\n距离70%目标:")
print(f"  top_k=8: {gap_8:.4f} ({gap_8/target_f1*100:.2f}%)")
print(f"  top_k=15: {gap_15:.4f} ({gap_15/target_f1*100:.2f}%)")

# 样本分布对比
print("\n" + "="*80)
print("样本表现分布对比")
print("="*80)

f1_scores_8 = [item['quest_avg_f1'] for item in results_8['details']]
f1_scores_15 = [item['quest_avg_f1'] for item in results_15['details']]

excellent_8 = sum(1 for score in f1_scores_8 if score >= 0.8)
good_8 = sum(1 for score in f1_scores_8 if 0.6 <= score < 0.8)
poor_8 = sum(1 for score in f1_scores_8 if score < 0.6)

excellent_15 = sum(1 for score in f1_scores_15 if score >= 0.8)
good_15 = sum(1 for score in f1_scores_15 if 0.6 <= score < 0.8)
poor_15 = sum(1 for score in f1_scores_15 if score < 0.6)

print(f"\ntop_k=8:")
print(f"  优秀 (≥0.8): {excellent_8}/50 ({excellent_8/50*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good_8}/50 ({good_8/50*100:.1f}%)")
print(f"  较差 (<0.6): {poor_8}/50 ({poor_8/50*100:.1f}%)")

print(f"\ntop_k=15:")
print(f"  优秀 (≥0.8): {excellent_15}/50 ({excellent_15/50*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good_15}/50 ({good_15/50*100:.1f}%)")
print(f"  较差 (<0.6): {poor_15}/50 ({poor_15/50*100:.1f}%)")