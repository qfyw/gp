#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert retrieval comparison results to RAGQuestEval test format"""

import json
import pandas as pd
from pathlib import Path

# 加载数据
with open('datasets/optimized_test_15/retrieval_comparison/comparison_results_50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 选择策略
strategy_name = 'full_hybrid'
strategy = next(s for s in data['strategies'] if s['strategy'] == strategy_name)

# 转换数据
data_points = []
for result in strategy['results']:
    data_points.append({
        'ID': result.get('question', '')[:50],  # 使用问题前50字符作为ID
        'ground_truth_text': result['reference_answer'],
        'generated_text': result['generated_answer']
    })

# 创建DataFrame
df = pd.DataFrame(data_points)

# 保存为CSV
output_file = Path('datasets/optimized_test_15/ragquesteval_input_full_hybrid.csv')
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"转换完成，共 {len(df)} 条数据")
print(f"输出文件: {output_file}")