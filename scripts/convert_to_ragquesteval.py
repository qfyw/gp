"""
将评测结果CSV转换为RAGQuestEval格式（不依赖pandas）
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def convert_to_ragquesteval(input_csv, output_csv, max_rows=20):
    """转换CSV格式"""
    data = []

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break

            data.append({
                'ID': row['id'],
                'ground_truth_text': row['gold_answer'],
                'generated_text': row['pred_answer']
            })

    # 写入新CSV
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['ID', 'ground_truth_text', 'generated_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"已转换 {len(data)} 条数据到 {output_csv}")

    # 显示前几行
    print("\n数据预览:")
    for i, row in enumerate(data[:3], 1):
        print(f"\n样本 {i}:")
        print(f"  ID: {row['ID']}")
        print(f"  原文: {row['ground_truth_text'][:50]}...")
        print(f"  生成: {row['generated_text'][:50]}...")

if __name__ == "__main__":
    input_file = PROJECT_ROOT / "datasets" / "eval_crud_best_n20_1776252216.csv"
    output_file = PROJECT_ROOT / "datasets" / "ragquesteval_test_n20.csv"

    convert_to_ragquesteval(input_file, output_file, max_rows=20)