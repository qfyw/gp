import json
from pathlib import Path
from collections import Counter

# 设置输出编码
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 加载 rrf_k_25 的结果
data = json.load(open('datasets/rrf_k_25/ragquesteval_results_1776.json', encoding='utf-8'))

# 找出失败样本（F1 < 0.4 或 Recall < 0.3）
failed_samples = []
details = data['details']

for i, sample in enumerate(details):
    f1 = sample['quest_avg_f1']
    recall = sample['quest_recall']
    if f1 < 0.4:
        failed_samples.append({
            'index': i,
            'id': sample['id'][:80],
            'f1': f1,
            'recall': recall,
            'detail': sample['detail']
        })

print(f"=== Failed Samples Analysis (Total: {len(failed_samples)}) ===\n")

# 分类统计
category_counts = Counter()
issue_types = []

for sample in failed_samples:
    answers_gm4gt = sample['detail'].get('answers_gm4gt', [])
    cant_infer = any('无法推断' in ans or '无法确定' in ans for ans in answers_gm4gt)

    if cant_infer and sample['recall'] == 0:
        category = "完全无法检索"
    elif cant_infer:
        category = "检索到但无法回答"
    elif sample['recall'] > 0.8 and sample['f1'] < 0.3:
        category = "高召回但低F1"
    elif sample['recall'] < 0.3:
        category = "低召回低F1"
    else:
        category = "其他"

    category_counts[category] += 1
    issue_types.append({
        'category': category,
        'f1': sample['f1'],
        'recall': sample['recall'],
        'id': sample['id']
    })

print(f"{'Index':<6s} {'F1':<8s} {'Recall':<8s} {'Category':<20s} {'Question ID'}")
print("-" * 120)

for sample in failed_samples:
    print(f"{sample['index']:<6d} {sample['f1']*100:>6.2f}%  {sample['recall']*100:>6.2f}%  ",
          end='')

    answers_gm4gt = sample['detail'].get('answers_gm4gt', [])
    cant_infer = any('无法推断' in ans or '无法确定' in ans for ans in answers_gm4gt)

    if cant_infer and sample['recall'] == 0:
        category = "完全无法检索"
    elif cant_infer:
        category = "检索到但无法回答"
    elif sample['recall'] > 0.8 and sample['f1'] < 0.3:
        category = "高召回低F1"
    elif sample['recall'] < 0.3:
        category = "低召回低F1"
    else:
        category = "其他"

    print(f"{category:<20s} {sample['id']}")

print("\n" + "=" * 100)
print("=== Category Statistics ===")
for cat, count in category_counts.most_common():
    pct = count / len(failed_samples) * 100
    print(f"{cat:<25s} {count:>2d} samples ({pct:>5.1f}%)")

# 深入分析每个失败样本的特征
print("\n" + "=" * 100)
print("=== Detailed Analysis of Failed Samples ===\n")

problematic_types = {
    "无法检索列表型数据": 0,
    "无法检索具体数值": 0,
    "无法检索人名/机构名": 0,
    "无法检索地点": 0,
    "无法检索时间": 0,
    "无法检索多个项目": 0,
    "无法推断答案": 0
}

for sample in failed_samples:
    detail = sample['detail']
    questions_gt = detail.get('questions_gt', [])
    answers_gt4gt = detail.get('answers_gt4gt', [])
    answers_gm4gt = detail.get('answers_gm4gt', [])

    print(f"--- Sample #{sample['index']} (F1: {sample['f1']*100:.2f}%, Recall: {sample['recall']*100:.2f}%) ---")

    # 分析问题类型
    question_type = []
    for q in questions_gt[:3]:
        if any(kw in q for kw in ['多少', '几', '哪些', '具体']):
            if '元' in q or '万' in q:
                question_type.append('具体数值')
            elif '家' in q or '个' in q or '项' in q:
                question_type.append('列表型数据')
            elif '谁' in q or '哪个' in q or '哪所' in q:
                question_type.append('人名/机构名')
            elif '哪' in q and '地' in q:
                question_type.append('地点')
            elif '何时' in q or '时间' in q:
                question_type.append('时间')

    # 检查生成答案
    cant_infer = any('无法推断' in ans or '无法确定' in ans for ans in answers_gm4gt)

    if cant_infer:
        print(f"  Problem: Cannot infer answers (检索失败)")
        print(f"  Question type hints: {', '.join(set(question_type))}")
    elif sample['recall'] > 0.8 and sample['f1'] < 0.3:
        print(f"  Problem: High recall but low F1 (检索到了但答错了)")
    else:
        print(f"  Problem: Partial recall (只检索到部分信息)")

    # 显示对比
    print(f"\n  Reference vs Generated answers (first 3):")
    for j in range(min(3, len(answers_gt4gt), len(answers_gm4gt))):
        gt = answers_gt4gt[j]
        gm = answers_gm4gt[j]
        if gt != gm and '无法推断' not in gm:
            match = "✗ MISMATCH"
        elif '无法推断' in gm:
            match = "✗ NO ANSWER"
        else:
            match = "✓ MATCH"
        print(f"    Q{j+1}: GT={gt[:30]:<30s} GM={gm[:30]:<30s} {match}")

    print()

# 统计问题类型
print("\n" + "=" * 100)
print("=== Common Problem Patterns ===")

all_questions = []
for sample in failed_samples:
    questions_gt = sample['detail'].get('questions_gt', [])
    all_questions.extend(questions_gt)

# 分析关键词
keywords_counter = Counter()
for q in all_questions:
    for kw in ['多少', '哪些', '具体', '几', '谁', '哪个', '何时', '哪', '金额', '元', '家', '个', '项']:
        if kw in q:
            keywords_counter[kw] += 1

print("\nTop keywords in failed questions:")
for kw, count in keywords_counter.most_common(15):
    print(f"  {kw:<8s} {count:>3d} occurrences")