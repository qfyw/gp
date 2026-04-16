# RAGQuestEval 快速指南

## 什么是 RAGQuestEval？

RAGQuestEval 是一种通过问答方式评估 RAG 系统生成质量的方法。

## 核心指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **Quest Avg F1** | 问答答案 F1 分数平均值 | 准确性 |
| **Quest Recall** | 1 - (无法推断比例) | 覆盖率 |

## 快速开始

### 1. 快速测试（3条示例数据）

```bash
python scripts/quick_test_ragquesteval.py
```

### 2. 完整测试（20条数据）

```bash
python scripts/test_ragquesteval.py \
  --result-file datasets/ragquesteval_test_n20.csv \
  --output-dir datasets \
  --save-quest-gt
```

### 3. 自定义数据

准备 CSV 文件：

```csv
ID,ground_truth_text,generated_text
1,原文内容...,生成内容...
```

然后运行测试。

## 测试结果

### 最新结果（20条数据）

| 指标 | 结果 | 标准差 |
|------|------|--------|
| Quest Avg F1 | 0.7254 | ±0.3193 |
| Quest Recall | 0.7171 | ±0.3445 |

## 常见问题

### 评估速度慢
- RAGQuestEval 需要多次调用 LLM
- 每条数据约 3-5 次调用
- 建议使用快速模型（如 qwen-flash）

### 结果全为0
- 检查 LLM API 配置
- 检查环境变量
- 查看错误日志

### 成本控制
- 开发阶段: 5-10 条数据
- 评测阶段: 20-30 条数据
- 保存问题缓存避免重复生成

## 更多信息

- [详细总结](./RAGQUESTEVAL_SUMMARY.md)
- [项目总结](./SUMMARY.md)
- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)