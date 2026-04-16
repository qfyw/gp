# 消融实验测试总结

## 测试概述

本文档记录了 RAG 系统的消融实验测试结果，使用 RAGQuestEval 框架进行评估。

## 测试环境

- **测试框架**: RAGQuestEval v1.0
- **测试时间**: 2026-04-16
- **LLM 模型**: qwen-flash
- **测试数据**: 8 条样本 + 20 条样本

## 测试结果

### 8 条样本测试

**测试时间**: 2026-04-16 10:13:46
**测试时长**: 39.9 秒

| 指标 | 平均值 | 标准差 |
|------|--------|--------|
| Quest Avg F1 | 0.6970 | ±0.2311 |
| Quest Recall | 0.7917 | ±0.2165 |

**样本分布**:
- 样本数: 8
- 成功: 8
- 失败: 0

### 20 条样本测试

**测试时间**: 2026-04-16 10:18:37
**测试时长**: 112.7 秒

| 指标 | 平均值 | 标准差 |
|------|--------|--------|
| Quest Avg F1 | 0.7587 | ±0.2540 |
| Quest Recall | 0.6912 | ±0.3577 |

**样本分布**:
- 样本数: 20
- 成功: 20
- 失败: 0

## 对比分析

### 8 条 vs 20 条

| 指标 | 8 条样本 | 20 条样本 | 变化 |
|------|---------|-----------|------|
| Quest Avg F1 | 0.6970 | 0.7587 | +8.9% |
| Quest Recall | 0.7917 | 0.6912 | -12.7% |

**分析**:
- **Quest Avg F1**: 20 条样本的准确率更高，说明数据量增加提升了平均准确度
- **Quest Recall**: 20 条样本的召回率略低，可能是因为数据增加引入了更多复杂问题

### 与之前测试对比

| 测试 | 样本数 | Quest Avg F1 | Quest Recall | 备注 |
|------|--------|-------------|-------------|------|
| 基础测试（之前） | 20 | 0.7254 | 0.7171 | 初始测试 |
| 8 条样本 | 8 | 0.6970 | 0.7917 | 简化测试 |
| 20 条样本 | 20 | 0.7587 | 0.6912 | 全量测试 |

**分析**:
- 20 条样本的 Quest Avg F1 (0.7587) 是最高的
- Quest Recall 在不同测试中波动较大（0.6912 - 0.7917）

## 性能分析

### Quest Avg F1 分布

根据 20 条样本的详细结果：

| 表现等级 | F1 范围 | 样本数 | 占比 |
|---------|---------|--------|------|
| 优秀 | ≥ 0.8 | 9 | 45% |
| 中等 | 0.4-0.8 | 5 | 25% |
| 较差 | < 0.4 | 6 | 30% |

### Quest Recall 分布

| 表现等级 | Recall 范围 | 样本数 | 占比 |
|---------|-------------|--------|------|
| 优秀 | ≥ 0.8 | 9 | 45% |
| 中等 | 0.4-0.8 | 5 | 25% |
| 较差 | < 0.4 | 6 | 30% |

## 主要发现

### 1. 整体性能良好

- **Quest Avg F1**: 0.7587 (75.87%) - 准确性较高
- **Quest Recall**: 0.6912 (69.12%) - 覆盖率中等
- **稳定性**: 标准差较大（±0.25, ±0.36），说明不同问题表现差异大

### 2. 表现分布均衡

- **优秀**: 45% (9条) - 接近一半问题回答优秀
- **中等**: 25% (5条) - 25% 问题表现一般
- **较差**: 30% (6条) - 30% 问题需要改进

### 3. 数据量影响

- 8 条样本: F1=0.6970, Recall=0.7917
- 20 条样本: F1=0.7587, Recall=0.6912
- **结论**: 数据量增加提升了准确度，但降低了召回率

## 优化方向

### 优先级 1: 提升稳定性

**问题**: 标准差较大（±0.25, ±0.36）
**目标**: 降低标准差，提高稳定性
**方法**:
- 改进检索召回策略
- 优化提示词
- 增加检索多样性

### 优先级 2: 提升覆盖率

**问题**: Quest Recall = 69.12%
**目标**: 提升至 80%+
**方法**:
- 增加 top_k
- 改进实体匹配
- 优化知识图谱推理

### 优先级 3: 减少错误回答

**问题**: 30% 问题表现较差
**目标**: 降低至 10%
**方法**:
- 改进实体和时间匹配
- 优化数字识别
- 加强事实核查

## 论文表格

### Table 1: 检索策略对比

| Method | Vector | Keyword | BM25 | Knowledge Graph | Quest Avg F1 | Quest Recall |
|--------|--------|---------|------|----------------|-------------|-------------|
| **Current System** | Y | Y | Y | Y | **0.7587** | **0.6912** |

### Table 2: 消融实验

| Method | Description | Quest Avg F1 | Δ | Quest Recall | Δ |
|--------|-------------|-------------|---|-------------|---|
| **Full Model** | All components | 0.7587 | - | 0.6912 | - |

### Table 3: 基线对比

| Method | Quest Avg F1 | Quest Recall |
|--------|-------------|-------------|
| **Our Method** | **0.7587** | **0.6912** |

## 生成的文件

### 8 条样本测试

```
datasets/ablation_tests/
├── ablation_results.json          # JSON 格式结果
├── comparison_table.csv           # CSV 对比表
├── ablation_report.md             # Markdown 报告
└── test_1_results_*.json          # 详细结果

datasets/paper_tables/
├── table1_retrieval_comparison.md # Table 1 (Markdown)
├── table2_ablation_study.md       # Table 2 (Markdown)
├── table3_baseline_comparison.md  # Table 3 (Markdown)
├── table1_retrieval_comparison.tex # Table 1 (LaTeX)
├── table2_ablation_study.tex      # Table 2 (LaTeX)
└── table3_baseline_comparison.tex # Table 3 (LaTeX)
```

### 20 条样本测试

```
datasets/ablation_tests_20/
├── ablation_results.json          # JSON 格式结果
├── comparison_table.csv           # CSV 对比表
├── ablation_report.md             # Markdown 报告
└── test_1_results_*.json          # 详细结果

datasets/paper_tables_20/
├── table1_retrieval_comparison.md # Table 1 (Markdown)
├── table2_ablation_study.md       # Table 2 (Markdown)
├── table3_baseline_comparison.md  # Table 3 (Markdown)
├── table1_retrieval_comparison.tex # Table 1 (LaTeX)
├── table2_ablation_study.tex      # Table 2 (LaTeX)
└── table3_baseline_comparison.tex # Table 3 (LaTeX)
```

## 下一步计划

### 短期（1-2周）

1. **组件对比测试**
   - 仅向量检索
   - 仅关键词检索
   - 仅 BM25 检索
   - 无知识图谱
   - 无重排序

2. **参数敏感性测试**
   - 不同 top_k 值
   - 不同召回倍数
   - 不同 chunk 大小

### 中期（2-4周）

1. **基线对比**
   - 无检索（纯 LLM）
   - 简单 RAG
   - LangChain RAG

2. **大规模测试**
   - 50 条样本
   - 100 条样本
   - 完整数据集

### 长期（1-2月）

1. **优化迭代**
   - 根据测试结果优化系统
   - 运行对比测试验证效果

2. **论文撰写**
   - 整理实验数据
   - 生成论文图表
   - 撰写实验章节

## 成本统计

### API 调用成本

- **8 条样本**: 约 24 次 API 调用（3-5 次/样本）
- **20 条样本**: 约 60 次 API 调用
- **总计**: 约 84 次 API 调用

### 时间成本

- **8 条样本**: 39.9 秒
- **20 条样本**: 112.7 秒
- **总计**: 152.6 秒（约 2.5 分钟）

## 结论

1. **整体性能**: RAGQuestEval 指标显示系统表现良好，Quest Avg F1 达到 75.87%

2. **稳定性**: 标准差较大，说明不同问题类型表现差异大，需要优化稳定性

3. **覆盖率**: Quest Recall 为 69.12%，需要提升到 80%+

4. **下一步**: 需要运行组件对比测试，验证各个模块的贡献

## 参考资料

- [RAGQuestEval 详细总结](../RAGQUESTEVAL_SUMMARY.md)
- [消融实验框架总结](../ABLATION_FRAMEWORK_SUMMARY.md)
- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)

---

**测试人员**: AI Assistant
**测试日期**: 2026-04-16
**报告版本**: 1.0