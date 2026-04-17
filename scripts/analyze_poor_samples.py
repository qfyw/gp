#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze poor performing samples"""

import json

# 加载优化后的test_ragquesteval.py结果
with open('datasets/baseline_20/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 按F1分数排序
sorted_samples = sorted(results['details'], key=lambda x: x['quest_avg_f1'])

print("="*80)
print("表现较差的样本分析（F1 < 0.5）")
print("="*80)

for i, sample in enumerate(sorted_samples[:5], 1):
    print(f"\n【样本 {i}】F1 = {sample['quest_avg_f1']:.4f}, Recall = {sample['quest_recall']:.4f}")
    print(f"ID: {sample['id'][:60]}...")
    print(f"\n问题生成:")
    for q in sample['detail']['questions_gt'][:3]:
        print(f"  - {q}")
    print(f"\n标准答案:")
    for a in sample['detail']['answers_gt4gt'][:3]:
        print(f"  - {a}")
    print(f"\n生成答案:")
    for a in sample['detail']['answers_gm4gt'][:3]:
        print(f"  - {a}")

# 加载原始数据查看检索情况
print("\n" + "="*80)
print("原始检索情况分析")
print("="*80)

with open('datasets/baseline_20/retrieval_comparison/comparison_results_20.json', 'r', encoding='utf-8') as f:
    retrieval_data = json.load(f)

vector_keyword = next(s for s in retrieval_data['strategies'] if s['strategy'] == 'vector_keyword')

print("\n检索文档数量分析:")
for result in vector_keyword['results']:
    if result['retrieved_docs_count'] > 0:
        print(f"  文档数: {result['retrieved_docs_count']}")
        # 简短显示生成答案
        answer = result['generated_answer'][:100]
        if '无法确定' in answer:
            print(f"    回答: 无法确定")
        else:
            print(f"    回答: {answer}...")
        print()