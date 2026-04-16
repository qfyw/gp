# 项目上下文总结

**最后更新**: 2026-04-16 10:30

## 项目概述

这是一个 RAG + 知识图谱智能问答系统，使用 RAGQuestEval 框架进行评估。目标是在 CRUD-RAG 评测基准上提升准确率。

**GitHub 仓库**: https://github.com/qfyw/gp

## 当前状态

### 项目结构

```
graduation project/
├── app.py                        # Streamlit 主应用
├── README.md                     # 项目文档（已更新）
├── SUMMARY.md                    # 项目总结
├── RAGQUESTEVAL_SUMMARY.md      # RAGQuestEval 详细总结
├── ABLATION_TEST_GUIDE.md       # 消融实验使用指南
├── ABLATION_FRAMEWORK_SUMMARY.md # 框架总结
├── ABLATION_QUICKSTART.md        # 快速开始
├── ABLATION_TEST_RESULTS.md     # 测试结果总结
├── ABLATION_TEST_COMPLETE.md     # 完成报告
├── PROJECT_CLEANUP.md           # 清理记录
├── src/                          # 核心代码
│   ├── agents/                  # 多智能体工作流
│   ├── config.py                # 配置文件
│   ├── retriever.py             # 混合检索器
│   └── generator.py             # 生成器
├── scripts/                     # 工具脚本
│   ├── test_ragquesteval.py          # RAGQuestEval 主脚本
│   ├── quick_test_ragquesteval.py   # 快速测试
│   ├── convert_to_ragquesteval.py   # CSV 转换
│   ├── run_simple_ablation.py       # 简化版消融实验（主要使用）
│   ├── generate_paper_tables.py    # 论文表格生成器
│   ├── run_all_tests.py             # 一键运行完整流程
│   ├── clear_rag_kb.py              # 清空知识库
│   └── ingest_crud_news.py          # 入库 CRUD 数据
├── datasets/                     # 测试数据
│   ├── ragquesteval_test_n20.csv      # 20 条测试数据
│   ├── ablation_test_samples.csv      # 8 条示例数据
│   ├── ablation_tests/               # 8 条测试结果
│   ├── ablation_tests_20/            # 20 条测试结果
│   ├── paper_tables/                 # 8 条论文表格
│   └── paper_tables_20/              # 20 条论文表格
├── configs/                      # 配置文件
│   └── test_scenarios.json       # 测试场景配置
└── .env                         # 环境变量（已配置）
```

## 已完成的工作

### 1. 项目清理（已完成）

- ✅ 删除旧的测试数据（40+ 文件）
- ✅ 删除旧的评测脚本（15+ 文件）
- ✅ 删除旧的优化记录和报告（11 文件）
- ✅ 精简文档，删除重复内容
- ✅ 聚焦 RAGQuestEval 评估框架

**提交**: `e4656ba`

### 2. 消融实验框架搭建（已完成）

- ✅ 创建测试脚本
  - `run_simple_ablation.py` - 简化版消融实验
  - `run_ablation_tests.py` - 完整版消融实验
  - `generate_paper_tables.py` - 论文表格生成器
  - `run_all_tests.py` - 一键运行完整流程

- ✅ 创建配置和数据
  - `test_scenarios.json` - 测试场景配置
  - `ablation_test_samples.csv` - 示例测试数据（8 条）

- ✅ 创建文档
  - `ABLATION_TEST_GUIDE.md` - 详细使用指南
  - `ABLATION_FRAMEWORK_SUMMARY.md` - 框架总结
  - `ABLATION_QUICKSTART.md` - 快速开始

**提交**: `cb4e5a6`

### 3. RAGQuestEval 测试（已完成）

#### 8 条样本测试

- **Quest Avg F1**: 0.6970 ± 0.2311
- **Quest Recall**: 0.7917 ± 0.2165
- **测试时长**: 39.9 秒
- **测试时间**: 2026-04-16 10:13:46

**提交**: `d57c465`

#### 20 条样本测试

- **Quest Avg F1**: 0.7587 ± 0.2540
- **Quest Recall**: 0.6912 ± 0.3577
- **测试时长**: 112.7 秒
- **测试时间**: 2026-04-16 10:18:37

**提交**: `6392728`

### 4. 文档更新（已完成）

- ✅ `ABLATION_TEST_RESULTS.md` - 测试结果总结
- ✅ `ABLATION_TEST_COMPLETE.md` - 完成报告
- ✅ `README.md` - 更新测试结果
- ✅ `PROJECT_CLEANUP.md` - 清理记录

**提交**: `71dfd74`, `e754d25`, `b78ed3a`

### 5. Git 推送（已完成）

- ✅ 成功推送到 GitHub
- ✅ 最新提交: `b78ed3a`
- ✅ 仓库: https://github.com/qfyw/gp

## 测试结果总结

### 性能指标

| 测试 | 样本数 | Quest Avg F1 | Quest Recall | 说明 |
|------|--------|-------------|-------------|------|
| 初始测试 | 20 | 0.7254 | 0.7171 | 早期测试 |
| 8 条样本 | 8 | 0.6970 | 0.7917 | 简化测试 |
| 20 条样本 | 20 | **0.7587** | 0.6912 | **当前最佳** |

### 表现分布（20 条样本）

- **优秀** (≥0.8): 45% (9条)
- **中等** (0.4-0.8): 25% (5条)
- **较差** (<0.4): 30% (6条)

### 论文表格（已生成）

#### Table 1: 检索策略对比

| Method | Vector | Keyword | BM25 | Knowledge Graph | Quest Avg F1 | Quest Recall |
|--------|--------|---------|------|----------------|-------------|-------------|
| Current System | Y | Y | Y | Y | **0.7587** | **0.6912** |

#### Table 2: 消融实验

| Method | Description | Quest Avg F1 | Δ | Quest Recall | Δ |
|--------|-------------|-------------|---|-------------|---|
| **Full Model** | All components | 0.7587 | - | 0.6912 | - |

#### Table 3: 基线对比

| Method | Quest Avg F1 | Quest Recall |
|--------|-------------|-------------|
| **Our Method** | **0.7587** | **0.6912** |

## 生成的关键文件

### 测试结果文件

1. **20 条样本（主要使用）**
   - `datasets/ablation_tests_20/ablation_results.json`
   - `datasets/ablation_tests_20/comparison_table.csv`
   - `datasets/ablation_tests_20/ablation_report.md`

2. **论文表格（20 条样本）**
   - `datasets/paper_tables_20/table1_retrieval_comparison.md`
   - `datasets/paper_tables_20/table2_ablation_study.md`
   - `datasets/paper_tables_20/table3_baseline_comparison.md`
   - `datasets/paper_tables_20/table1_retrieval_comparison.tex`
   - `datasets/paper_tables_20/table2_ablation_study.tex`
   - `datasets/paper_tables_20/table3_baseline_comparison.tex`

### 测试数据

- `datasets/ragquesteval_test_n20.csv` - 20 条测试数据
- `datasets/ablation_test_samples.csv` - 8 条示例数据

## 环境配置

### .env 配置（已设置）

```env
# LLM 配置
OPENAI_API_KEY=sk-de027d7c15ba4756b871857e3262644a
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash

# PostgreSQL
POSTGRES_DSN=postgresql://postgres:123456@localhost:5432/rag
PGVECTOR_COLLECTION=crud_eval
KB_NAMESPACE=crud_eval

# 检索参数
RETRIEVAL_VECTOR_TOP_K=8
RETRIEVAL_KEYWORD_TOP_K=8
RETRIEVAL_GRAPH_MAX=10

# Rerank
RERANK_ENABLED=1
RERANK_RECALL_MULT=3

# 其他
BM25_ENABLED=1
DOC_ONLY_MODE=0
INTERNAL_DOC_ONLY_ANSWER=1
```

## 下一步计划

### 优先级 1: 组件对比测试（重要！）

**目的**: 验证各个模块的贡献

**测试场景**:
1. 仅向量检索（无关键词、无 BM25、无图谱、无重排序）
2. 向量 + 关键词（无 BM25、无图谱、无重排序）
3. 向量 + 关键词 + BM25（无图谱、无重排序）
4. 完整混合（无图谱）
5. 完整混合（无重排序）
6. 完整混合（当前系统）

**实现方法**:
- 修改 `configs/test_scenarios.json`
- 或修改 `src/config.py` 中的参数
- 运行 `python scripts/run_simple_ablation.py`

**预期输出**:
- 论文 Table 1: 检索策略对比
- 各组件贡献的量化分析

### 优先级 2: 基线对比测试

**测试场景**:
1. 无检索（纯 LLM 生成）
2. 简单 RAG（向量检索 + 直接生成）
3. LangChain RAG 标准实现
4. 我们的完整系统

### 优先级 3: 参数优化测试

**测试参数**:
- top_k 值：3, 5, 8, 10
- Rerank 召回倍数：2, 3, 4
- Chunk 大小：128, 256, 512, 600

### 优先级 4: 大规模测试

- 50 条样本
- 100 条样本
- 完整数据集

## 关键命令

### 运行消融实验

```bash
# 简化版（推荐）
python scripts/run_simple_ablation.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests

# 一键运行完整流程
python scripts/run_all_tests.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests
```

### 生成论文表格

```bash
python scripts/generate_paper_tables.py \
  --results datasets/ablation_tests/ablation_results.json \
  --output-dir datasets/paper_tables \
  --format all
```

### 快速测试

```bash
# 3 条示例数据
python scripts/quick_test_ragquesteval.py

# 8 条示例数据
python scripts/test_ragquesteval.py \
  --result-file datasets/ablation_test_samples.csv \
  --output-dir datasets
```

### Git 操作

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "描述"
git push origin master

# 查看提交历史
git log --oneline -10
```

## 重要提示

### 测试成本

- RAGQuestEval 需要多次调用 LLM（3-5 次/样本）
- 20 条样本 ≈ 60 次 API 调用
- 成本相对较高，注意控制样本量

### 测试时间

- 20 条样本 ≈ 2 分钟
- 50 条样本 ≈ 5 分钟
- 100 条样本 ≈ 10 分钟

### 数据备份

- 重要的测试结果已提交到 Git
- 配置文件已保存
- 测试数据已保存

## 待解决的问题

### 1. 性能稳定性

**问题**: 标准差较大（±0.25）
**影响**: 不同问题表现差异大
**解决**: 改进检索召回策略，优化提示词

### 2. 覆盖率不足

**问题**: Quest Recall = 69.12%
**影响**: 部分问题无法回答
**解决**: 增加 top_k，改进实体匹配

### 3. 错误回答

**问题**: 30% 问题表现较差
**影响**: 用户体验不佳
**解决**: 改进数字识别，加强事实核查

## 文档索引

### 核心文档

1. **项目文档**
   - `README.md` - 项目介绍
   - `SUMMARY.md` - 项目总结

2. **评估框架**
   - `RAGQUESTEVAL_SUMMARY.md` - RAGQuestEval 详细总结
   - `RAGQUESTEVAL_GUIDE.md` - 快速指南

3. **消融实验**
   - `ABLATION_TEST_GUIDE.md` - 详细使用指南
   - `ABLATION_FRAMEWORK_SUMMARY.md` - 框架总结
   - `ABLATION_QUICKSTART.md` - 快速开始
   - `ABLATION_TEST_RESULTS.md` - 测试结果总结
   - `ABLATION_TEST_COMPLETE.md` - 完成报告

4. **其他**
   - `PROJECT_CLEANUP.md` - 清理记录

## Git 状态

### 最新提交

```
b78ed3a 全量消融实验测试完成
e754d25 更新 README 添加最新测试结果
71dfd74 添加消融实验测试总结
6392728 20条样本消融实验测试
d57c465 首次消融实验测试（8条样本）
271af54 添加消融实验快速开始指南
8234a92 添加消融实验框架使用总结
cb4e5a6 添加消融实验和对照测试框架
e4656ba 项目清理：聚焦RAGQuestEval评估框架
56201fd Initial commit: RAG+KG intelligent Q&A system
```

### 当前状态

- **分支**: master
- **状态**: 领先 origin/master 8 个提交
- **已推送**: ✅ 是

## Python 环境

### 已安装的包

- jieba
- numpy
- pandas
- networkx
- psycopg[binary]
- python-dotenv
- openai
- langchain
- langchain-openai
- langchain-core
- langgraph

### Python 版本

- Python 3.12.1
- 路径: C:\Users\qfyw\AppData\Local\Programs\Python\Python312\python.exe

## 快速恢复工作

### 如果要继续测试组件对比：

1. 修改配置
```bash
# 编辑 src/config.py 或 .env
# 修改检索参数
```

2. 运行测试
```bash
python scripts/run_simple_ablation.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/component_tests
```

3. 生成表格
```bash
python scripts/generate_paper_tables.py \
  --results datasets/component_tests/ablation_results.json \
  --output-dir datasets/component_tables
```

### 如果要查看现有结果：

```bash
# 查看测试报告
cat datasets/ablation_tests_20/ablation_report.md

# 查看对比表格
cat datasets/ablation_tests_20/comparison_table.csv

# 查看论文表格
cat datasets/paper_tables_20/table1_retrieval_comparison.md
```

## 联系方式

- **GitHub**: https://github.com/qfyw/gp
- **项目路径**: D:\graduation project

---

**最后更新**: 2026-04-16 10:30
**文档版本**: 1.0
**维护者**: RAG 系统优化团队