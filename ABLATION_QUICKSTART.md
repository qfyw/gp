# 消融实验框架 - 快速开始

## 一键运行测试

```bash
python scripts/run_all_tests.py \
  --test-data datasets/ablation_test_samples.csv \
  --output-dir datasets/ablation_tests
```

## 分步运行

### 1. 运行消融实验

```bash
python scripts/run_simple_ablation.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests
```

### 2. 生成论文表格

```bash
python scripts/generate_paper_tables.py \
  --results datasets/ablation_tests/ablation_results.json \
  --output-dir datasets/paper_tables \
  --format all
```

## 输出文件

### 测试结果
- `datasets/ablation_tests/ablation_results.json` - JSON 格式结果
- `datasets/ablation_tests/comparison_table.csv` - CSV 对比表
- `datasets/ablation_tests/ablation_report.md` - Markdown 报告

### 论文表格（Markdown）
- `datasets/paper_tables/table1_retrieval_comparison.md` - 检索策略对比
- `datasets/paper_tables/table2_ablation_study.md` - 消融实验
- `datasets/paper_tables/table3_baseline_comparison.md` - 基线对比

### 论文表格（LaTeX）
- `datasets/paper_tables/table1_retrieval_comparison.tex` - 检索策略对比
- `datasets/paper_tables/table2_ablation_study.tex` - 消融实验
- `datasets/paper_tables/table3_baseline_comparison.tex` - 基线对比

## 查看结果

```bash
# 查看对比表格
cat datasets/ablation_tests/comparison_table.csv

# 查看详细报告
cat datasets/ablation_tests/ablation_report.md

# 查看 LaTeX 表格
cat datasets/paper_tables/table1_retrieval_comparison.tex
```

## 更多信息

- [详细使用指南](./ABLATION_TEST_GUIDE.md)
- [框架总结](./ABLATION_FRAMEWORK_SUMMARY.md)