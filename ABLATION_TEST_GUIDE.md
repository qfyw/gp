# 消融实验和对照测试指南

## 概述

本测试框架支持运行多种对照测试，用于验证 RAG 系统各个组件的有效性。测试结果可以用于论文中的表格和对比分析。

## 测试脚本

### 1. 简化版消融实验（推荐）

运行最基础的对比测试：

```bash
python scripts/run_simple_ablation.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests
```

**特点**:
- 直接运行 RAGQuestEval 评估
- 支持多个测试场景对比
- 自动生成对比表格和报告

**输出文件**:
- `ablation_results.json` - JSON 格式结果
- `comparison_table.csv` - CSV 格式对比表
- `ablation_report.md` - Markdown 报告

### 2. 完整版消融实验

支持配置文件驱动的复杂测试场景：

```bash
python scripts/run_ablation_tests.py \
  --config configs/test_scenarios.json \
  --output-dir datasets/ablation_tests \
  --scenarios vector_only full_hybrid
```

**参数**:
- `--config`: 测试场景配置文件
- `--output-dir`: 输出目录
- `--scenarios`: 指定要运行的场景（可选，默认运行所有）

### 3. 论文表格生成器

将测试结果格式化为论文表格：

```bash
python scripts/generate_paper_tables.py \
  --results datasets/ablation_tests/ablation_results.json \
  --output-dir datasets/paper_tables \
  --format all
```

**参数**:
- `--results`: 测试结果文件（JSON 或 CSV）
- `--output-dir`: 输出目录
- `--format`: 输出格式（markdown/latex/all）

**输出表格**:
- `table1_retrieval_comparison.md/.tex` - 检索策略对比
- `table2_ablation_study.md/.tex` - 消融实验
- `table3_baseline_comparison.md/.tex` - 基线对比

## 测试场景配置

### 检索策略对比

| 场景名称 | 描述 | 目的 |
|---------|------|------|
| vector_only | 仅向量检索 | 基线 |
| vector_keyword | 向量 + 关键词 | 验证关键词作用 |
| vector_keyword_bm25 | 向量 + 关键词 + BM25 | 验证 BM25 作用 |
| full_hybrid_no_kg | 完整混合（无图谱） | 验证图谱作用 |
| full_hybrid | 完整混合检索 | 完整方法 |

### 推荐的最小测试集（写论文用）

```python
# 1. 向量基线
vector_only: Quest Avg F1 = 0.XXXX

# 2. 简单融合
vector_keyword: Quest Avg F1 = 0.XXXX

# 3. 完整混合（无图谱）
full_hybrid_no_kg: Quest Avg F1 = 0.XXXX

# 4. 完整方法
full_hybrid: Quest Avg F1 = 0.7254
```

## 论文表格示例

### Table 1: 检索策略对比

| Method | Vector | Keyword | BM25 | Knowledge Graph | Quest Avg F1 | Quest Recall |
|--------|--------|---------|------|----------------|-------------|-------------|
| Vector Only | ✓ | ✗ | ✗ | ✗ | 0.XXXX | 0.XXXX |
| Vector + Keyword | ✓ | ✓ | ✗ | ✗ | 0.XXXX | 0.XXXX |
| Vector + Keyword + BM25 | ✓ | ✓ | ✓ | ✗ | 0.XXXX | 0.XXXX |
| Full Hybrid | ✓ | ✓ | ✓ | ✓ | **0.7254** | **0.7171** |

### Table 2: 消融实验

| Method | Description | Quest Avg F1 | Δ | Quest Recall | Δ |
|--------|-------------|-------------|---|-------------|---|
| Full Model | All components | 0.7254 | - | 0.7171 | - |
| -KG | Without Knowledge Graph | 0.XXXX | -X% | 0.XXXX | -X% |
| -BM25 | Without BM25 | 0.XXXX | -X% | 0.XXXX | -X% |
| -Keyword | Without Keyword Search | 0.XXXX | -X% | 0.XXXX | -X% |

### Table 3: 基线对比

| Method | Quest Avg F1 | Quest Recall |
|--------|-------------|-------------|
| No Retrieval (LLM only) | 0.XXXX | 0.XXXX |
| Simple RAG | 0.XXXX | 0.XXXX |
| **Our Method** | **0.7254** | **0.7171** |

## 完整工作流程

### 步骤 1: 准备测试数据

```bash
# 使用现有的测试数据
python scripts/convert_to_ragquesteval.py

# 或使用自定义数据
# 准备 CSV 文件，格式：ID, ground_truth_text, generated_text
```

### 步骤 2: 运行对比测试

```bash
# 运行简化版测试
python scripts/run_simple_ablation.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests
```

### 步骤 3: 生成论文表格

```bash
# 生成所有格式的表格
python scripts/generate_paper_tables.py \
  --results datasets/ablation_tests/ablation_results.json \
  --output-dir datasets/paper_tables \
  --format all
```

### 步骤 4: 查看结果

```bash
# 查看对比表格
cat datasets/ablation_tests/comparison_table.csv

# 查看详细报告
cat datasets/ablation_tests/ablation_report.md

# 查看 LaTeX 表格
cat datasets/paper_tables/table1_retrieval_comparison.tex
```

## 自定义测试场景

### 修改配置文件

编辑 `configs/test_scenarios.json`，添加新的测试场景：

```json
{
  "name": "my_custom_test",
  "description": "我的自定义测试",
  "config": {
    "RETRIEVAL_VECTOR_TOP_K": 10,
    "RETRIEVAL_KEYWORD_TOP_K": 10,
    "BM25_ENABLED": "1",
    "RERANK_ENABLED": "1"
  }
}
```

### 运行特定场景

```bash
python scripts/run_ablation_tests.py \
  --scenarios my_custom_test \
  --output-dir datasets/my_test
```

## 结果分析

### 查看性能提升

```python
import json

# 加载结果
with open('datasets/ablation_tests/ablation_results.json') as f:
    results = json.load(f)['results']

# 计算性能提升
baseline = results[0]['metrics']['quest_avg_f1']
full = results[-1]['metrics']['quest_avg_f1']
improvement = (full - baseline) / baseline * 100

print(f"性能提升: {improvement:.2f}%")
```

### 绘制性能曲线

使用 Python 或 R 绘制性能曲线：

```python
import matplotlib.pyplot as plt

# 准备数据
names = [r['name'] for r in results]
f1_scores = [r['metrics']['quest_avg_f1'] for r in results]
recall_scores = [r['metrics']['quest_recall'] for r in results]

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(names, f1_scores, marker='o', label='Quest Avg F1')
plt.plot(names, recall_scores, marker='s', label='Quest Recall')
plt.xticks(rotation=45)
plt.xlabel('Method')
plt.ylabel('Score')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('datasets/ablation_tests/performance_curve.png')
```

## 注意事项

1. **成本控制**: RAGQuestEval 需要多次调用 LLM，建议控制样本量
2. **测试时间**: 完整测试可能需要数小时，建议分批运行
3. **结果一致性**: 使用相同的数据集和随机种子确保可重复性
4. **数据备份**: 保存测试结果和配置文件，便于后续分析

## 常见问题

### Q: 如何添加新的测试场景？

A: 编辑 `configs/test_scenarios.json`，添加新场景的配置。

### Q: 如何只运行部分测试？

A: 使用 `--scenarios` 参数指定场景名称。

### Q: 测试速度慢怎么办？

A: 减少样本量或使用更快的 LLM。

### Q: 如何在论文中使用这些表格？

A: 使用 `--format latex` 生成 LaTeX 表格，直接复制到论文中。

## 参考资料

- [RAGQuestEval 详细总结](../RAGQUESTEVAL_SUMMARY.md)
- [项目总结](../SUMMARY.md)
- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)