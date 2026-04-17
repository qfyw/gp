#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze retrieval quality for failed samples"""

import json

# 加载数据
with open('datasets/final_test_50/retrieval_comparison/comparison_results_50.json', 'r', encoding='utf-8') as f:
    retrieval_data = json.load(f)

with open('datasets/final_test_50/ragquesteval_results_1776.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

# 获取full_hybrid策略
full_hybrid_retrieval = next(s for s in retrieval_data['strategies'] if s['strategy'] == 'full_hybrid')

# 找出F1=0的失败样本ID
failed_ids = [item['id'][:60] for item in eval_data['details'] if item['quest_avg_f1'] == 0.0]

print("="*80)
print(f"失败样本检索分析 (共{len(failed_ids)}个)")
print("="*80)

for result in full_hybrid_retrieval['results']:
    # 检查是否是失败样本
    question_id = result['question'][:60]
    if question_id in failed_ids:
        print(f"\n【失败样本】")
        print(f"问题: {result['question'][:80]}...")
        print(f"检索文档数: {result['retrieved_docs_count']}")
        print(f"回答前150字: {result['generated_answer'][:150]}...")
        print(f"是否'无法确定': {'是' if '无法确定' in result['generated_answer'] else '否'}")

# 统计所有样本的检索文档数分布
print("\n" + "="*80)
print("检索文档数分布")
print("="*80)

doc_counts = [r['retrieved_docs_count'] for r in full_hybrid_retrieval['results']]
print(f"最小文档数: {min(doc_counts)}")
print(f"最大文档数: {max(doc_counts)}")
print(f"平均文档数: {sum(doc_counts)/len(doc_counts):.2f}")
print(f"中位数文档数: {sorted(doc_counts)[len(doc_counts)//2]}")

# 统计回答是否包含"无法确定"
print("\n" + "="*80)
print("回答质量分析")
print("="*80)

cannot_determine_count = sum(1 for r in full_hybrid_retrieval['results'] if '无法确定' in r['generated_answer'])
print(f"包含'无法确定'的样本: {cannot_determine_count}/50 ({cannot_determine_count/50*100:.1f}%)")

# 对比成功和失败样本的文档数
print("\n" + "="*80)
print("成功 vs 失败样本的检索文档数对比")
print("="*80)

failed_counts = [r['retrieved_docs_count'] for i, r in enumerate(full_hybrid_retrieval['results'])
                 if eval_data['details'][i]['quest_avg_f1'] == 0.0]

success_counts = [r['retrieved_docs_count'] for i, r in enumerate(full_hybrid_retrieval['results'])
                  if eval_data['details'][i]['quest_avg_f1'] > 0.0]

if failed_counts:
    print(f"\n失败样本 (F1=0):")
    print(f"  样本数: {len(failed_counts)}")
    print(f"  平均文档数: {sum(failed_counts)/len(failed_counts):.2f}")
    print(f"  文档数范围: {min(failed_counts)} - {max(failed_counts)}")

if success_counts:
    print(f"\n成功样本 (F1>0):")
    print(f"  样本数: {len(success_counts)}")
    print(f"  平均文档数: {sum(success_counts)/len(success_counts):.2f}")
    print(f"  文档数范围: {min(success_counts)} - {max(success_counts)}")