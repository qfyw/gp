# RAG + 知识图谱智能问答系统

基于 CRUD-RAG 评估基准的智能问答系统，使用 RAGQuestEval 框架进行评估。

## 项目简介

本项目实现了一个 RAG（检索增强生成）+ 知识图谱的智能问答系统，在 CRUD-RAG 评测基准上进行优化和评估。

## 核心特性

- **混合检索系统**: 向量检索 + 关键词检索 + BM25 + 知识图谱检索
- **融合策略**: RRF (Reciprocal Rank Fusion)
- **重排序**: CrossEncoder (BAAI/bge-reranker-base)
- **RAGQuestEval 评估**: 基于 CRUD-RAG 论文的问答式评估框架

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

## 实验结果

### 最新测试结果（2条样本示例）

| 策略 | Quest Avg F1 | Quest Recall | 样本数 |
|------|-------------|-------------|-------|
| vector_only | 0.4762 ± 0.4762 | 0.5000 ± 0.5000 | 2 |
| vector_keyword | 0.4615 ± 0.4615 | 0.5000 ± 0.5000 | 2 |
| full_hybrid | 0.4762 ± 0.4762 | 0.5000 ± 0.5000 | 2 |

**注意**: 这是一个小规模测试（2个样本），结果仅供参考。建议使用更大的样本数（10-50）来获得更可靠的评估结果。

### 1. 激活虚拟环境

```bash
cd "D:\graduation project"
.venv\Scripts\Activate.ps1
```

### 2. 运行检索策略对比实验

```bash
# 小规模测试（20条样本，约30分钟）
python scripts/run_experiment.py \
  --data-path CRUD_RAG/data/crud_split/split_merged.json \
  --max-samples 20 \
  --output-dir datasets/experiment_results
```

### 3. 查看结果

```bash
# 查看检索结果
cat datasets/experiment_results/retrieval_comparison/comparison_results_20.json

# 查看评估结果
cat datasets/experiment_results/ragquesteval_evaluation/*.json
```

## 实验脚本

### 1. run_experiment.py

一键运行完整的检索策略对比实验 + RAGQuestEval 评估。

**参数**:
- `--data-path`: CRUD-RAG 数据集路径
- `--max-samples`: 最大样本数（建议20-50）
- `--output-dir`: 输出目录

**示例**:
```bash
python scripts/run_experiment.py --max-samples 20
```

### 2. run_retrieval_comparison.py

运行检索策略对比实验。

**测试策略**:
1. vector_only - 仅向量检索
2. vector_keyword - 向量 + 关键词检索
3. vector_keyword_bm25 - 向量 + 关键词 + BM25
4. full_hybrid_no_kg - 完整混合（无知识图谱）
5. full_hybrid - 完整混合检索

**参数**:
- `--data-path`: 数据集路径
- `--max-samples`: 最大样本数
- `--output-dir`: 输出目录
- `--custom-config`: 自定义策略配置文件（JSON）

**示例**:
```bash
python scripts/run_retrieval_comparison.py \
  --data-path CRUD_RAG/data/crud_split/split_merged.json \
  --max-samples 20
```

### 3. evaluate_ragquesteval.py

使用 RAGQuestEval 评估测试结果。

**参数**:
- `--results-file`: 测试结果文件路径
- `--output-dir`: 输出目录

**示例**:
```bash
python scripts/evaluate_ragquesteval.py \
  --results-file datasets/retrieval_comparison/comparison_results_20.json
```

## 自定义实验

### 自定义检索策略

创建一个 JSON 配置文件（如 `custom_strategies.json`）:

```json
[
    {
        "name": "my_strategy",
        "description": "我的自定义策略",
        "config": {
            "use_keyword_search": true,
            "use_bm25": true,
            "use_knowledge_graph": false,
            "use_rerank": false
        }
    }
]
```

然后运行:
```bash
python scripts/run_retrieval_comparison.py \
  --custom-config custom_strategies.json
```

## 项目结构

```
.
├── app.py                                    # Streamlit 主应用
├── README.md                                 # 项目文档
├── src/                                      # 核心代码
│   ├── config.py                            # 配置文件
│   ├── retriever.py                         # 混合检索器
│   └── generator.py                         # 生成器
├── scripts/                                  # 实验脚本
│   ├── run_experiment.py                    # 一键运行实验
│   ├── run_retrieval_comparison.py          # 检索策略对比
│   ├── evaluate_ragquesteval.py             # RAGQuestEval 评估
│   ├── test_ragquesteval.py                 # RAGQuestEval 测试
│   ├── quick_test_ragquesteval.py           # 快速测试
│   ├── clear_rag_kb.py                      # 清空知识库
│   ├── ingest_crud_news.py                  # 入库 CRUD 数据
│   └── convert_to_ragquesteval.py           # CSV 格式转换
├── datasets/                                 # 测试数据和结果
│   └── experiment_results/                  # 实验结果
├── configs/                                  # 配置文件
├── CRUD_RAG/                                 # CRUD-RAG 数据集
│   └── data/crud_split/
│       └── split_merged.json                # 测试数据（800个问答对）
├── requirements.txt                          # Python 依赖
└── .env                                      # 环境变量
```

## 实验流程

### 完整实验流程

```bash
# 1. 准备数据（首次运行）
python scripts/ingest_crud_news.py

# 2. 运行实验
python scripts/run_experiment.py --max-samples 20

# 3. 查看结果
cat datasets/experiment_results/ragquesteval_evaluation/*.json
```

### 分步实验流程

```bash
# 步骤1: 运行检索策略对比
python scripts/run_retrieval_comparison.py \
  --data-path CRUD_RAG/data/crud_split/split_merged.json \
  --max-samples 20 \
  --output-dir datasets/my_experiment

# 步骤2: 评估结果
python scripts/evaluate_ragquesteval.py \
  --results-file datasets/my_experiment/comparison_results_20.json \
  --output-dir datasets/my_experiment/evaluation
```

## 环境配置

### .env 配置

```env
# LLM 配置
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash

# PostgreSQL
POSTGRES_DSN=postgresql://postgres:password@localhost:5432/rag
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

## 注意事项

### 1. 虚拟环境

务必使用虚拟环境运行所有实验：

```powershell
cd "D:\graduation project"
.venv\Scripts\Activate.ps1
```

### 2. 数据库连接

确保 PostgreSQL 数据库正在运行，并且已安装 pgvector 扩展。

### 3. 测试时间

| 样本数 | 预计时间 |
|--------|----------|
| 10 | 15-20分钟 |
| 20 | 30-40分钟 |
| 50 | 1.5-2小时 |
| 100 | 3-4小时 |

### 4. 成本控制

RAGQuestEval 需要多次调用 LLM（每条样本约 6-10 次调用）：
- 10 条样本：约 60-100 次 API 调用
- 20 条样本：约 120-200 次 API 调用

## 参考资料

- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)
- [CRUD-RAG GitHub](https://github.com/IAAR-Shanghai/CRUD_RAG)
- [GitHub 仓库](https://github.com/qfyw/gp)

## 许可证

MIT License