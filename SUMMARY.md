# 项目总结

## 项目概述

本项目实现了一个 RAG（检索增强生成）+ 知识图谱的智能问答系统，在 CRUD-RAG 评测基准上进行优化和评估。

## 核心技术

### 1. 混合检索系统
- **向量检索**: 基于 BGE embedding
- **关键词检索**: PostgreSQL pg_trgm
- **BM25**: PostgreSQL 全文检索
- **知识图谱**: NetworkX 图谱检索
- **融合策略**: RRF (Reciprocal Rank Fusion)
- **重排序**: CrossEncoder (BAAI/bge-reranker-base)

### 2. 多智能体工作流
- **Router**: 路由分发
- **KGQuery**: 知识图谱查询
- **Join**: 结果合并
- **CheckRelevance**: 相关性检查
- **Synthesizer**: 答案合成

### 3. RAGQuestEval 评估
- **Quest Avg F1**: 问答答案准确性
- **Quest Recall**: 信息覆盖率

## 评估结果

### RAGQuestEval 指标（20条数据）

| 指标 | 结果 | 标准差 |
|------|------|--------|
| Quest Avg F1 | 0.7254 | ±0.3193 |
| Quest Recall | 0.7171 | ±0.3445 |

### 表现分布

- **优秀** (≥0.8): 45% (9条)
- **中等** (0.4-0.8): 25% (5条)
- **较差** (<0.4): 30% (6条)

## 主要问题

### 1. 数字准确性问题 (30%)
- 不同时间/地点的数据混淆
- 需要更精确的实体和时间匹配

### 2. 检索失败 (15%)
- 关键词召回不足
- 需要优化检索策略

### 3. 部分信息缺失 (20%)
- 多子问只回答部分
- 需要改进提示词完整性

## 优化方向

### 优先级1: 实体和时间匹配
- 实现更精确的数字识别
- 添加时间范围过滤
- 区分不同来源的数据

### 优先级2: 检索召回优化
- 增加 top_k
- 优化关键词权重
- 提升检索多样性

### 优先级3: 提示词优化
- 强调完整性要求
- 增加多子问示例
- 改进回答格式约束

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
└── requirements.txt         # Python 依赖
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```env
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash

POSTGRES_DSN=postgresql://postgres:password@localhost:5432/rag
PGVECTOR_COLLECTION=crud_eval
```

### 3. 运行评估
```bash
# 快速测试
python scripts/quick_test_ragquesteval.py

# 完整测试
python scripts/test_ragquesteval.py \
  --result-file datasets/ragquesteval_test_n20.csv \
  --output-dir datasets
```

## 参考资料

- [CRUD-RAG 论文](https://arxiv.org/abs/2401.17043)
- [RAGQuestEval 评估指南](./RAGQUESTEVAL_GUIDE.md)
- [RAGQuestEval 详细总结](./RAGQUESTEVAL_SUMMARY.md)
- [GitHub 仓库](https://github.com/qfyw/gp)

---

**最后更新**: 2026-04-16