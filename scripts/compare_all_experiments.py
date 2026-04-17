#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare all experimental results"""

import json

# 加载所有实验结果
results = {
    'top_k=8, rerank=4': {
        'file': 'datasets/final_test_50/ragquesteval_results_1776.json',
        'name': '基础配置（top_k=8, RERANK_RECALL_MULT=4）'
    },
    'top_k=15, rerank=4': {
        'file': 'datasets/optimized_test_15/ragquesteval_results_1776.json',
        'name': '增加检索参数（top_k=15, RERANK_RECALL_MULT=4）'
    },
    'top_k=8, rerank=5': {
        'file': 'datasets/final_optimized/ragquesteval_results_1776.json',
        'name': '优化重排序（top_k=8, RERANK_RECALL_MULT=5）'
    }
}

# 加载数据
for key, info in results.items():
    try:
        with open(info['file'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            info['data'] = data
    except:
        info['data'] = None

print("="*80)
print("Full Hybrid 策略 - 所有实验结果对比（50样本）")
print("="*80)

for key, info in results.items():
    if info['data']:
        print(f"\n【{info['name']}】")
        print(f"  Quest Avg F1: {info['data']['quest_avg_f1_mean']:.4f} ({info['data']['quest_avg_f1_mean']*100:.2f}%)")
        print(f"  Quest Recall: {info['data']['quest_recall_mean']:.4f} ({info['data']['quest_recall_mean']*100:.2f}%)")
        print(f"  F1标准差: {info['data']['quest_avg_f1_std']:.4f}")

# 性能对比
print("\n" + "="*80)
print("性能对比（相对于基础配置）")
print("="*80)

base_f1 = results['top_k=8, rerank=4']['data']['quest_avg_f1_mean']
base_recall = results['top_k=8, rerank=4']['data']['quest_recall_mean']

for key, info in results.items():
    if info['data'] and key != 'top_k=8, rerank=4':
        f1_change = (info['data']['quest_avg_f1_mean'] - base_f1)
        recall_change = (info['data']['quest_recall_mean'] - base_recall)
        print(f"\n{info['name']}:")
        print(f"  F1变化: {f1_change:+.4f} ({f1_change/base_f1*100:+.2f}%)")
        print(f"  Recall变化: {recall_change:+.4f} ({recall_change/base_recall*100:+.2f}%)")

# 距离70%目标
print("\n" + "="*80)
print("距离70%目标")
print("="*80)

target_f1 = 0.70
for key, info in results.items():
    if info['data']:
        gap = target_f1 - info['data']['quest_avg_f1_mean']
        print(f"{info['name']}: {gap:.4f} ({gap/target_f1*100:.2f}%)")

# 样本分布对比
print("\n" + "="*80)
print("样本表现分布对比")
print("="*80)

for key, info in results.items():
    if info['data']:
        f1_scores = [item['quest_avg_f1'] for item in info['data']['details']]
        excellent = sum(1 for score in f1_scores if score >= 0.8)
        good = sum(1 for score in f1_scores if 0.6 <= score < 0.8)
        poor = sum(1 for score in f1_scores if score < 0.6)
        zero = sum(1 for score in f1_scores if score == 0.0)
        
        print(f"\n{info['name']}:")
        print(f"  优秀 (≥0.8): {excellent}/50 ({excellent/50*100:.1f}%)")
        print(f"  良好 (0.6-0.8): {good}/50 ({good/50*100:.1f}%)")
        print(f"  较差 (<0.6): {poor}/50 ({poor/50*100:.1f}%)")
        print(f"  失败 (0.0): {zero}/50 ({zero/50*100:.1f}%)")