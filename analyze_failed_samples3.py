import json
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# 加载 rrf_k_25 的结果
data = json.load(open('datasets/rrf_k_25/ragquesteval_results_1776.json', encoding='utf-8'))

# 加载检索结果以查看检索到的文档数
comparison_data = json.load(open('datasets/rrf_k_25/comparison_results_50.json', encoding='utf-8'))

# 获取 full_hybrid 策略的结果
full_hybrid = None
for strategy in comparison_data['strategies']:
    if strategy['strategy'] == 'full_hybrid':
        full_hybrid = strategy
        break

print("=== Deep Analysis of Failed Samples ===\n")

# 找出失败样本
failed_indices = []
details = data['details']

for i, sample in enumerate(details):
    f1 = sample['quest_avg_f1']
    if f1 < 0.4:
        failed_indices.append(i)

print(f"Total failed samples: {len(failed_indices)}\n")

# 分析每个失败样本的检索情况
print(f"{'Index':<6s} {'F1':<8s} {'Recall':<8s} {'Docs':<6s} {'Problem Type':<25s} {'Key Issue'}")
print("-" * 130)

for idx in failed_indices:
    sample = details[idx]
    f1 = sample['quest_avg_f1']
    recall = sample['quest_recall']

    # 获取检索到的文档数
    if full_hybrid and idx < len(full_hybrid['results']):
        retrieved_docs_count = full_hybrid['results'][idx].get('retrieved_docs_count', 'N/A')
    else:
        retrieved_docs_count = 'N/A'

    # 分析问题类型
    answers_gm4gt = sample['detail'].get('answers_gm4gt', [])
    questions_gt = sample['detail'].get('questions_gt', [])

    cant_infer = any('无法推断' in ans or '无法确定' in ans for ans in answers_gm4gt)

    # 确定关键问题
    key_issues = []

    # 检查问题类型
    for q in questions_gt[:5]:
        if '多少' in q or '几' in q:
            key_issues.append('数值')
        if '哪些' in q or '列表' in q:
            key_issues.append('列表')
        if '谁' in q or '哪个' in q or '哪所' in q:
            key_issues.append('命名实体')
        if '金额' in q or '元' in q or '万' in q:
            key_issues.append('金额')
        if '时间' in q or '何时' in q:
            key_issues.append('时间')
        if '哪' in q and '地' in q:
            key_issues.append('地点')

    key_issues_str = ', '.join(set(key_issues))

    # 确定问题类型
    if recall == 0 and f1 == 0:
        problem_type = "完全检索失败"
    elif cant_infer and recall > 0:
        problem_type = "检索到但无法回答"
    elif recall > 0.8 and f1 < 0.3:
        problem_type = "高召回低准确率"
    else:
        problem_type = "部分检索失败"

    print(f"{idx:<6d} {f1*100:>6.2f}%  {recall*100:>6.2f}%  {str(retrieved_docs_count):<6s}  {problem_type:<25s} {key_issues_str}")

# 统计关键问题类型
print("\n" + "=" * 130)
print("=== Key Issue Statistics ===\n")

all_key_issues = []
for idx in failed_indices:
    sample = details[idx]
    questions_gt = sample['detail'].get('questions_gt', [])

    for q in questions_gt:
        if '金额' in q or '元' in q or '万' in q:
            all_key_issues.append('金额')
        elif '多少' in q or '几' in q:
            all_key_issues.append('数值/数量')
        elif '哪些' in q or '列表' in q:
            all_key_issues.append('列表/枚举')
        elif '谁' in q or '哪个' in q or '哪所' in q:
            all_key_issues.append('人名/机构名')
        elif '时间' in q or '何时' in q:
            all_key_issues.append('时间')
        elif '哪' in q and ('地' in q or '地点' in q):
            all_key_issues.append('地点')
        elif '措施' in q or '政策' in q:
            all_key_issues.append('政策/措施')
        else:
            all_key_issues.append('其他')

issue_counter = Counter(all_key_issues)
print(f"{'Issue Type':<20s} {'Count':>6s} {'Percentage':>12s}")
print("-" * 45)
for issue, count in issue_counter.most_common():
    pct = count / len(all_key_issues) * 100
    print(f"{issue:<20s} {count:>6d} {pct:>10.1f}%")

# 分析文档数与失败的关系
print("\n" + "=" * 130)
print("=== Retrieved Documents Analysis ===\n")

doc_counts = []
for idx in failed_indices:
    if full_hybrid and idx < len(full_hybrid['results']):
        doc_count = full_hybrid['results'][idx].get('retrieved_docs_count', 0)
        doc_counts.append(doc_count)

if doc_counts:
    avg_docs = sum(doc_counts) / len(doc_counts)
    min_docs = min(doc_counts)
    max_docs = max(doc_counts)

    print(f"Average retrieved docs for failed samples: {avg_docs:.1f}")
    print(f"Min retrieved docs: {min_docs}")
    print(f"Max retrieved docs: {max_docs}")

    # 比较成功样本
    success_doc_counts = []
    for i, sample in enumerate(details):
        if i not in failed_indices and full_hybrid and i < len(full_hybrid['results']):
            doc_count = full_hybrid['results'][i].get('retrieved_docs_count', 0)
            success_doc_counts.append(doc_count)

    if success_doc_counts:
        avg_success_docs = sum(success_doc_counts) / len(success_doc_counts)
        print(f"\nAverage retrieved docs for successful samples: {avg_success_docs:.1f}")
        print(f"Difference: {avg_docs - avg_success_docs:.1f}")

# 生成建议
print("\n" + "=" * 130)
print("=== Recommendations ===\n")

print("Based on the analysis, here are the main issues and recommendations:\n")

print("1. Issue: 完全无法检索 (46.2% of failures)")
print("   - Pattern: 无法回答的问题主要集中在：")
print("     * 具体金额/数值（9个样本涉及）")
print("     * 人名/机构名（6个样本涉及）")
print("     * 地点信息")
print("   - Recommendation: 检查这些信息是否在数据集中，或者切片方式是否合适\n")

print("2. Issue: 检索到但无法回答 (38.5% of failures)")
print("   - Pattern: 检索到了文档，但LLM无法从中提取答案")
print("   - Recommendation: 检查检索到的文档是否包含答案，可能需要调整切片大小或重排序\n")

print("3. Issue: 关键词 '哪' 出现频率最高（24次）")
print("   - Pattern: '哪个'、'哪里'、'哪些' 这类查询词容易出现问题")
print("   - Recommendation: 优化关键词检索对这类查询的处理\n")

print("4. Issue: 具体数值/金额信息难以检索")
print("   - Pattern: '多少钱'、'多少个' 这类问题失败率较高")
print("   - Recommendation: 考虑实体识别增强检索\n")