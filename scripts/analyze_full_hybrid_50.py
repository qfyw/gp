#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze full_hybrid 50-sample test results"""

import json

# 加载优化后的test_ragquesteval.py结果
with open('datasets/final_test_50/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    optimized_results = json.load(f)

# 加载原版evaluate_ragquesteval.py结果
with open('datasets/final_test_50/ragquesteval_evaluation/ragquesteval_comparison_results_50.json', 'r', encoding='utf-8') as f:
    original_results = json.load(f)

# 获取full_hybrid策略的结果
full_hybrid = next(s for s in original_results['strategies'] if s['strategy'] == 'full_hybrid')

print("="*80)
print("Full Hybrid 策略 - 50样本评估结果对比")
print("="*80)

print("\n【原版 evaluate_ragquesteval.py】")
print(f"  Quest Avg F1: {full_hybrid['avg_quest_avg_f1']:.4f} ({full_hybrid['avg_quest_avg_f1']*100:.2f}%)")
print(f"  Quest Recall: {full_hybrid['avg_quest_recall']:.4f} ({full_hybrid['avg_quest_recall']*100:.2f}%)")
print(f"  F1标准差: {full_hybrid['std_quest_avg_f1']:.4f}")

print("\n【优化版 test_ragquesteval.py】")
print(f"  Quest Avg F1: {optimized_results['quest_avg_f1_mean']:.4f} ({optimized_results['quest_avg_f1_mean']*100:.2f}%)")
print(f"  Quest Recall: {optimized_results['quest_recall_mean']:.4f} ({optimized_results['quest_recall_mean']*100:.2f}%)")
print(f"  F1标准差: {optimized_results['quest_avg_f1_std']:.4f}")

print("\n" + "="*80)
print("性能提升")
print("="*80)

f1_improvement = (optimized_results['quest_avg_f1_mean'] - full_hybrid['avg_quest_avg_f1'])
recall_improvement = (optimized_results['quest_recall_mean'] - full_hybrid['avg_quest_recall'])

print(f"F1提升: {f1_improvement:+.4f} ({f1_improvement/full_hybrid['avg_quest_avg_f1']*100:+.2f}%)")
print(f"Recall提升: {recall_improvement:+.4f} ({recall_improvement/full_hybrid['avg_quest_recall']*100:+.2f}%)")
print(f"F1标准差变化: {(optimized_results['quest_avg_f1_std'] - full_hybrid['std_quest_avg_f1']):+.4f}")

# 计算达到70%目标的情况
target_f1 = 0.70
gap = target_f1 - optimized_results['quest_avg_f1_mean']
print(f"\n距离70%目标差距: {gap:.4f} ({gap/target_f1*100:.2f}%)")

# 分析样本表现分布
print("\n" + "="*80)
print("样本表现分布")
print("="*80)

optimized_f1_scores = [item['quest_avg_f1'] for item in optimized_results['details']]
original_f1_scores = full_hybrid['quest_avg_f1_scores']

excellent_opt = sum(1 for score in optimized_f1_scores if score >= 0.8)
good_opt = sum(1 for score in optimized_f1_scores if 0.6 <= score < 0.8)
medium_opt = sum(1 for score in optimized_f1_scores if 0.4 <= score < 0.6)
poor_opt = sum(1 for score in optimized_f1_scores if score < 0.4)
zero_opt = sum(1 for score in optimized_f1_scores if score == 0.0)

excellent_orig = sum(1 for score in original_f1_scores if score >= 0.8)
good_orig = sum(1 for score in original_f1_scores if 0.6 <= score < 0.8)
medium_orig = sum(1 for score in original_f1_scores if 0.4 <= score < 0.6)
poor_orig = sum(1 for score in original_f1_scores if score < 0.4)
zero_orig = sum(1 for score in original_f1_scores if score == 0.0)

print(f"\n优化后 test_ragquesteval.py:")
print(f"  优秀 (≥0.8): {excellent_opt}/50 ({excellent_opt/50*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good_opt}/50 ({good_opt/50*100:.1f}%)")
print(f"  中等 (0.4-0.6): {medium_opt}/50 ({medium_opt/50*100:.1f}%)")
print(f"  较差 (<0.4): {poor_opt}/50 ({poor_opt/50*100:.1f}%)")
print(f"  失败 (0.0): {zero_opt}/50 ({zero_opt/50*100:.1f}%)")

print(f"\n原版 evaluate_ragquesteval.py:")
print(f"  优秀 (≥0.8): {excellent_orig}/50 ({excellent_orig/50*100:.1f}%)")
print(f"  良好 (0.6-0.8): {good_orig}/50 ({good_orig/50*100:.1f}%)")
print(f"  中等 (0.4-0.6): {medium_orig}/50 ({medium_orig/50*100:.1f}%)")
print(f"  较差 (<0.4): {poor_orig}/50 ({poor_orig/50*100:.1f}%)")
print(f"  失败 (0.0): {zero_orig}/50 ({zero_orig/50*100:.1f}%)")

# 分析失败样本
print("\n" + "="*80)
print(f"失败样本分析 (F1=0, 共{zero_opt}个)")
print("="*80)

failed_samples = [item for item in optimized_results['details'] if item['quest_avg_f1'] == 0.0]
for i, sample in enumerate(failed_samples, 1):
    print(f"\n【失败样本 {i}】")
    print(f"  ID: {sample['id'][:60]}...")
    print(f"  Recall: {sample['quest_recall']:.2f}")
    print(f"  问题数: {len(sample['detail']['questions_gt'])}")
    print(f"  样本回答: {sample['detail']['answers_gm4gt'][:3]}")