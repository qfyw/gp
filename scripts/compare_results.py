#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare evaluation results between different frameworks"""

import json

# 加载优化后的test_ragquesteval.py结果
with open('datasets/baseline_20/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    optimized_results = json.load(f)

# 加载evaluate_ragquesteval.py结果
with open('datasets/baseline_20/ragquesteval_evaluation/ragquesteval_comparison_results_20.json', 'r', encoding='utf-8') as f:
    original_results = json.load(f)

# 获取vector_keyword策略的结果
vector_keyword = next(s for s in original_results['strategies'] if s['strategy'] == 'vector_keyword')

print("="*60)
print("RAGQuestEval 评估结果对比（20个样本）")
print("="*60)

print("\n【evaluate_ragquesteval.py - vector_keyword策略】")
print(f"  Quest Avg F1: {vector_keyword['avg_quest_avg_f1']:.4f}")
print(f"  Quest Recall: {vector_keyword['avg_quest_recall']:.4f}")
print(f"  F1标准差: {vector_keyword['std_quest_avg_f1']:.4f}")
print(f"  Recall标准差: {vector_keyword['std_quest_recall']:.4f}")

print("\n【test_ragquesteval.py - 优化版】")
print(f"  Quest Avg F1: {optimized_results['quest_avg_f1_mean']:.4f}")
print(f"  Quest Recall: {optimized_results['quest_recall_mean']:.4f}")
print(f"  F1标准差: {optimized_results['quest_avg_f1_std']:.4f}")
print(f"  Recall标准差: {optimized_results['quest_recall_std']:.4f}")

print("\n" + "="*60)
print("提升分析")
print("="*60)

f1_improvement = (optimized_results['quest_avg_f1_mean'] - vector_keyword['avg_quest_avg_f1'])
recall_change = (optimized_results['quest_recall_mean'] - vector_keyword['avg_quest_recall'])

print(f"F1提升: {f1_improvement:+.4f} ({f1_improvement/vector_keyword['avg_quest_avg_f1']*100:+.2f}%)")
print(f"Recall变化: {recall_change:+.4f} ({recall_change/vector_keyword['avg_quest_recall']*100:+.2f}%)")

# 计算达到70%目标的情况
target_f1 = 0.70
gap = target_f1 - optimized_results['quest_avg_f1_mean']
print(f"\n距离70%目标差距: {gap:.4f} ({gap/target_f1*100:.2f}%)")

# 分析样本表现
print("\n" + "="*60)
print("样本表现分布")
print("="*60)

# 优化后的样本F1分数分布
optimized_f1_scores = [item['quest_avg_f1'] for item in optimized_results['details']]
excellent = sum(1 for score in optimized_f1_scores if score >= 0.8)
good = sum(1 for score in optimized_f1_scores if 0.6 <= score < 0.8)
medium = sum(1 for score in optimized_f1_scores if 0.4 <= score < 0.6)
poor = sum(1 for score in optimized_f1_scores if score < 0.4)

print(f"\n优化后test_ragquesteval.py:")
print(f"  优秀 (≥0.8): {excellent}/20 ({excellent/20*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good}/20 ({good/20*100:.1f}%)")
print(f"  中等 (0.4-0.6): {medium}/20 ({medium/20*100:.1f}%)")
print(f"  较差 (<0.4): {poor}/20 ({poor/20*100:.1f}%)")

# 原版样本F1分数分布
original_f1_scores = vector_keyword['quest_avg_f1_scores']
excellent_orig = sum(1 for score in original_f1_scores if score >= 0.8)
good_orig = sum(1 for score in original_f1_scores if 0.6 <= score < 0.8)
medium_orig = sum(1 for score in original_f1_scores if 0.4 <= score < 0.6)
poor_orig = sum(1 for score in original_f1_scores if score < 0.4)

print(f"\n原版evaluate_ragquesteval.py:")
print(f"  优秀 (≥0.8): {excellent_orig}/20 ({excellent_orig/20*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good_orig}/20 ({good_orig/20*100:.1f}%)")
print(f"  中等 (0.4-0.6): {medium_orig}/20 ({medium_orig/20*100:.1f}%)")
print(f"  较差 (<0.4): {poor_orig}/20 ({poor_orig/20*100:.1f}%)")