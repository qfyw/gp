import json
from pathlib import Path
from collections import Counter

# 加载 rrf_k_25 的结果
data = json.load(open('datasets/rrf_k_25/ragquesteval_results_1776.json', encoding='utf-8'))

# 找出失败样本（F1 < 0.3 或 Recall < 0.5）
failed_samples = []
details = data['details']

for i, sample in enumerate(details):
    f1 = sample['quest_avg_f1']
    recall = sample['quest_recall']
    if f1 < 0.3 or recall < 0.5:
        failed_samples.append({
            'index': i,
            'id': sample['id'][:100],  # 截断长ID
            'f1': f1,
            'recall': recall,
            'detail': sample['detail']
        })

print(f"=== 失败样本分析 (共 {len(failed_samples)} 个) ===\n")

# 统计特征
cant_infer_count = 0
low_f1_high_recall = 0  # F1低但Recall高（能回答但答错了）
low_f1_low_recall = 0   # F1低且Recall低（无法回答）
category_counts = Counter()

print(f"{'索引':<6s} {'F1':<8s} {'Recall':<8s} {'问题类型':<20s} {'ID (前100字符)'}")
print("-" * 120)

for sample in failed_samples:
    f1 = sample['f1']
    recall = sample['recall']
    detail = sample['detail']

    # 分析特征
    answers_gm4gt = detail.get('answers_gm4gt', [])
    cant_infer = any(ans.strip() == '无法推断' for ans in answers_gm4gt)

    if cant_infer:
        category = "完全无法回答"
        cant_infer_count += 1
        category_counts[category] += 1
    elif recall > 0.7:
        category = "高召回但低F1"
        low_f1_high_recall += 1
        category_counts[category] += 1
    elif recall < 0.5:
        category = "低召回低F1"
        low_f1_low_recall += 1
        category_counts[category] += 1
    else:
        category = "部分回答"
        category_counts[category] += 1

    print(f"{sample['index']:<6d} {f1*100:>6.2f}%  {recall*100:>6.2f}%  {category:<20s} {sample['id']}")

print("\n" + "=" * 80)
print("=== 失败样本分类统计 ===")
for cat, count in category_counts.most_common():
    print(f"{cat:<20s} {count:>3d} 个 ({count/len(failed_samples)*100:>5.1f}%)")

print(f"\n完全无法回答（无法推断）: {cant_infer_count} 个")
print(f"高召回但低F1（答错了）: {low_f1_high_recall} 个")
print(f"低召回低F1（无法回答）: {low_f1_low_recall} 个")

# 详细分析每个失败样本
print("\n" + "=" * 80)
print("=== 详细失败样本分析 ===\n")

for i, sample in enumerate(failed_samples[:10]):  # 显示前10个
    print(f"--- 样本 #{sample['index']} ---")
    print(f"ID: {sample['id']}")
    print(f"F1: {sample['f1']*100:.2f}%, Recall: {sample['recall']*100:.2f}%")

    detail = sample['detail']
    questions_gt = detail.get('questions_gt', [])
    answers_gt4gt = detail.get('answers_gt4gt', [])
    answers_gm4gt = detail.get('answers_gm4gt', [])

    print(f"\n问题示例 (共 {len(questions_gt)} 个):")
    for q in questions_gt[:3]:
        print(f"  Q: {q}")

    print(f"\n参考答案 vs 生成答案 (前5个):")
    for j in range(min(5, len(answers_gt4gt), len(answers_gm4gt))):
        print(f"  Q{j+1}: 参考={answers_gt4gt[j]}, 生成={answers_gm4gt[j]}")

    print()