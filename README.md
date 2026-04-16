# RAG + Knowledge Graph 智能问答系统

基于 CRUD-RAG 评估基准的智能问答系统，使用 RAGQuestEval 框架进行评估。

## 项目简介

本项目实现了一个 RAG（检索增强生成）+ 知识图谱的智能问答系统，在 CRUD-RAG 评测基准上进行优化和评估。

## 核心特性

- **混合检索系统**: 向量检索 + 关键词检索 + BM25 + 知识图谱检索
- **多智能体工作流**: Router → KGQuery → Join → CheckRelevance → Synthesizer
- **RAGQuestEval 评估**: 基于 CRUD-RAG 论文的问答式评估框架
- **Streamlit 界面**: Web 端交互式问答和文档管理

## RAGQuestEval 评估框架

### 核心指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **Quest Avg F1** | 问答对 F1 分数的平均值 | 衡量答案的准确性 |
| **Quest Recall** | 1 - (无法推断回答的比例) | 衡量信息的覆盖率 |

### 工作原理

1. 从 ground truth 提取关键信息并生成问题
2. 用 ground truth 回答问题（参考答案）
3. 用生成文本回答同样问题（预测答案）
4. 计算 F1 分数和 Recall

### 快速测试

```bash
# 使用示例数据快速测试
python scripts/quick_test_ragquesteval.py

# 使用自定义 CSV 测试
python scripts/test_ragquesteval.py \
  --result-file datasets/ragquesteval_test_n20.csv \
  --output-dir datasets \
  --save-quest-gt
```

### 数据格式

CSV 文件需要包含以下列：

```csv
ID,ground_truth_text,generated_text
1,原文内容...,生成内容...
2,原文内容...,生成内容...
```

## 项目结构

```
.
├── app.py                    # Streamlit 主应用
├── src/                      # 核心代码
│   ├── agents/              # 多智能体工作流
│   ├── config.py            # 配置文件
│   ├── retriever.py         # 混合检索器
│   └── generator.py         # 生成器
├── scripts/                 # 工具脚本
│   ├── test_ragquesteval.py      # RAGQuestEval 主脚本
│   ├── quick_test_ragquesteval.py # 快速测试
│   ├── convert_to_ragquesteval.py # CSV 格式转换
│   ├── clear_rag_kb.py           # 清空知识库
│   ├── ingest_crud_news.py       # 入库 CRUD 数据
│   └── ingest_txt_dir.py         # 入库文本目录
├── datasets/                # 测试数据和结果
│   ├── ragquesteval_test_n20.csv # 测试数据
│   ├── ragquesteval_results_*.json # 评估结果
│   └── quest_gt_save_*.json       # 问题答案缓存
├── configs/                 # 配置文件
├── requirements.txt         # Python 依赖
└── .env                     # 环境变量
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```env
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash

# PostgreSQL
POSTGRES_DSN=postgresql://postgres:password@localhost:5432/rag
PGVECTOR_COLLECTION=crud_eval

# 检索参数
RETRIEVAL_VECTOR_TOP_K=8
RETRIEVAL_KEYWORD_TOP_K=8
RETRIEVAL_GRAPH_MAX=10
```

### 3. 运行 RAGQuestEval 评估

#### 快速测试（3条示例数据）

```bash
python scripts/quick_test_ragquesteval.py
```

#### 完整测试（20条数据）

```bash
python scripts/test_ragquesteval.py \
  --result-file datasets/ragquesteval_test_n20.csv \
  --output-dir datasets \
  --save-quest-gt
```

### 4. 运行消融实验（论文用）

#### 一键运行完整流程

```bash
python scripts/run_all_tests.py \
  --test-data datasets/ragquesteval_test_n20.csv \
  --output-dir datasets/ablation_tests
```

这将自动：
1. 运行消融实验
2. 生成对比表格
3. 生成论文用的 Markdown 和 LaTeX 表格

#### 详细使用指南

查看 [消融实验指南](./ABLATION_TEST_GUIDE.md)

### 5. 入库数据

```bash
# 入库 CRUD 新闻数据
python scripts/ingest_crud_news.py

# 入库文本目录
python scripts/ingest_txt_dir.py data/documents
```

### 6. 运行 Web 应用

```bash
streamlit run app.py
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```env
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash

# PostgreSQL
POSTGRES_DSN=postgresql://postgres:password@localhost:5432/rag
PGVECTOR_COLLECTION=crud_eval

# 检索参数
RETRIEVAL_VECTOR_TOP_K=8
RETRIEVAL_KEYWORD_TOP_K=8
RETRIEVAL_GRAPH_MAX=10
```

### 3. 入库数据

```bash
# 入库 CRUD 新闻数据
python scripts/ingest_crud_news.py

# 入库文本目录
python scripts/ingest_txt_dir.py data/documents
```

### 4. 运行 Web 应用

```bash
streamlit run app.py
```

### 5. 运行评估

```bash
# 快速测试（3条示例数据）
python scripts/quick_test_ragquesteval.py

# 完整测试（20条数据）
python scripts/test_ragquesteval.py \
  --result-file datasets/ragquesteval_test_n20.csv \
  --output-dir datasets
```

## 评估结果

### 最新测试结果（20条数据）

| 指标 | 结果 | 标准差 |
|------|------|--------|
| Quest Avg F1 | 0.7254 | ±0.3193 |
| Quest Recall | 0.7171 | ±0.3445 |

### 表现分布

- **优秀** (≥0.8): 45% (9条)
- **中等** (0.4-0.8): 25% (5条)
- **较差** (<0.4): 30% (6条)

### 主要问题

1. **数字准确性**: 不同时间/地点的数据混淆
2. **检索失败**: 关键词召回不足
3. **部分信息缺失**: 多子问只回答部分

## 优化方向

1. **实体和时间匹配**: 实现更精确的数字识别和时间范围过滤
2. **检索召回优化**: 增加 top_k，优化关键词权重
3. **提示词优化**: 强调完整性要求，增加多子问示例

## 参考资料

- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)
- [RAGQuestEval 评估指南](./RAGQUESTEVAL_GUIDE.md)
- [RAGQuestEval 详细总结](./RAGQUESTEVAL_SUMMARY.md)

## 许可证

MIT License