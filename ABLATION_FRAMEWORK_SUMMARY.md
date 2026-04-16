# 消融实验框架使用总结

## 创建的文件

### 1. 测试脚本
- `scripts/run_simple_ablation.py` - 简化版消融实验（推荐）
- `scripts/run_ablation_tests.py` - 完整版消融实验
- `scripts/run_all_tests.py` - 一键运行完整流程
- `scripts/generate_paper_tables.py` - 论文表格生成器

### 2. 配置文件
- `configs/test_scenarios.json` - 测试场景配置

### 3. 测试数据
- `datasets/ablation_test_samples.csv` - 示例测试数据（8条）

### 4. 文档
- `ABLATION_TEST_GUIDE.md` - 详细使用指南

## 快速开始

### 方法 1: 一键运行（推荐）

```bash
python scripts/run_all_tests.py \
  --test-data datasets/ablation_test_samples.csv \
  --output-dir datasets/ablation_tests
```

这将自动：
1. 运行消融实验
2. 生成对比表格
3. 生成论文用的 Markdown 和 LaTeX 表格

### 方法 2: 分步运行

#### 步骤 1: 运行消融实验

```bash
python scripts/run_simple_ablation.py \
  --test-data datasets/ablation_test_samples.csv \
  --output-dir datasets/ablation_tests
```

#### 步骤 2: 生成论文表格

```bash
python scripts/generate_paper_tables.py \
  --results datasets/ablation_tests/ablation_results.json \
  --output-dir datasets/paper_tables \
  --format all
```

## 测试场景

当前支持的测试场景：

| 场景名称 | 描述 | 配置 |
|---------|------|------|
| vector_only | 仅向量检索 | Vector Only |
| vector_keyword | 向量 + 关键词 | + Keyword |
| vector_keyword_bm25 | 向量 + 关键词 + BM25 | + BM25 |
| full_hybrid_no_kg | 完整混合（无图谱） | - KG |
| full_hybrid_no_rerank | 完整混合（无重排序） | - Rerank |
| full_hybrid | 完整混合检索 | All components |
| full_hybrid_larger_topk | 完整混合（增大top_k） | top_k=8 |
| full_hybrid_high_recall | 完整混合（高召回） | recall_mult=4 |

## 论文表格

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

### Table 3: 基线对比

| Method | Quest Avg F1 | Quest Recall |
|--------|-------------|-------------|
| No Retrieval (LLM only) | 0.XXXX | 0.XXXX |
| Simple RAG | 0.XXXX | 0.XXXX |
| **Our Method** | **0.7254** | **0.7171** |

## 输出文件

运行测试后，会生成以下文件：

```
datasets/ablation_tests/
├── ablation_results.json          # JSON 格式结果
├── comparison_table.csv           # CSV 格式对比表
├── ablation_report.md             # Markdown 报告
└── test_1_results_*.json          # 每个测试的详细结果

datasets/paper_tables/
├── table1_retrieval_comparison.md # Table 1 (Markdown)
├── table2_ablation_study.md       # Table 2 (Markdown)
├── table3_baseline_comparison.md  # Table 3 (Markdown)
├── table1_retrieval_comparison.tex # Table 1 (LaTeX)
├── table2_ablation_study.tex      # Table 2 (LaTeX)
└── table3_baseline_comparison.tex # Table 3 (LaTeX)
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

### 使用自己的数据

准备 CSV 文件，格式如下：

```csv
ID,ground_truth_text,generated_text
1,原文内容...,生成内容...
2,原文内容...,生成内容...
```

然后运行测试：

```bash
python scripts/run_simple_ablation.py \
  --test-data your_data.csv \
  --output-dir datasets/your_test
```

## 论文中的使用建议

### 最小测试集（写论文用）

如果时间有限，至少运行这 4 个测试：

1. **vector_only** - 向量基线
2. **vector_keyword_bm25** - 多路融合
3. **full_hybrid_no_kg** - 无图谱
4. **full_hybrid** - 完整方法

### 表格选择

论文中至少包含这 3 个表格：

1. **Table 1**: 检索策略对比 - 验证各组件作用
2. **Table 2**: 消融实验 - 量化每个组件的贡献
3. **Table 3**: 基线对比 - 与现有方法对比

### LaTeX 表格使用

生成的 LaTeX 表格可以直接复制到论文中：

```latex
% 复制 datasets/paper_tables/table1_retrieval_comparison.tex
\begin{table}[h]
\centering
\caption{Comparison of Retrieval Strategies}
\label{tab:retrieval_comparison}
\begin{tabular}{lccccc}
\toprule
Method & Vector & Keyword & BM25 & Knowledge Graph & Quest Avg F1 \\\\
& & & & & & Quest Recall \\\\
\midrule
...
\bottomrule
\end{tabular}
\end{table}
```

## 注意事项

1. **成本控制**: RAGQuestEval 需要多次调用 LLM
   - 开发阶段：5-10 条数据
   - 评测阶段：20-30 条数据
   - 论文中途：50 条数据

2. **测试时间**: 完整测试可能需要数小时
   - 建议分批运行
   - 使用快速 LLM（如 qwen-flash）

3. **结果一致性**:
   - 使用相同的数据集
   - 使用相同的随机种子
   - 保存测试配置

4. **数据备份**:
   - 保存测试结果
   - 保存配置文件
   - 便于后续分析

## Git 提交

代码已提交到本地仓库：
- Commit: `cb4e5a6`
- 状态: 已提交，未推送（网络问题）

推送到 GitHub（网络恢复后）：

```bash
git push origin master
```

## 参考资料

- [详细使用指南](./ABLATION_TEST_GUIDE.md)
- [RAGQuestEval 总结](./RAGQUESTEVAL_SUMMARY.md)
- [项目总结](./SUMMARY.md)
- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)

---

**创建时间**: 2026-04-16
**框架版本**: 1.0
**维护者**: RAG 系统优化团队