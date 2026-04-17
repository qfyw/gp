# 优化入库参数实验 - 待完成步骤

## 📋 实验背景

**目标**: 将 Full Hybrid 策略的 RAGQuestEval 指标从 59.68% 提升到 70%

**当前最优配置** (datasets/final_test_50):
- Quest Avg F1: 59.68%
- Quest Recall: 68.83%
- 失败样本: 10个
- 检索参数: top_k=8,8,10, rerank=4
- 入库参数: chunk_size=128, overlap=0

**优化方案**: 改进入库参数
- chunk_size: 128 → 256
- overlap: 0 → 50
- 使用新的 collection: crud_eval_optimized

## 🔧 配置修改

### 已完成的修改
- ✅ PGVECTOR_COLLECTION: crud_eval → crud_eval_optimized
- ✅ KB_NAMESPACE: crud_eval → crud_eval_optimized
- ✅ INGEST_CHUNK_SIZE: 128 → 256
- ✅ INGEST_CHUNK_OVERLAP: 0 → 50
- ✅ RERANK_RECALL_MULT: 4 (保持不变)

### .env 配置关键参数
```
INGEST_CHUNK_SIZE=256
INGEST_CHUNK_OVERLAP=50
PGVECTOR_COLLECTION=crud_eval_optimized
KB_NAMESPACE=crud_eval_optimized
RERANK_RECALL_MULT=4
```

## 📌 当前状态

**正在执行**: 步骤2 - 重新入库数据
- 命令: `scripts/ingest_crud_news.py --docs-dir "D:\graduation project\CRUD_RAG\data\80000_docs" --max-docs 0 --glob-pattern "documents_dup*" --upload-batch 100`
- 预计时间: 30-40分钟
- 状态: 进行中...

## 🚀 待完成步骤

### 步骤3: 运行测试
```powershell
# 运行50个样本的测试（只测试full_hybrid策略）
.venv\Scripts\python.exe scripts/run_retrieval_comparison.py --data-path CRUD_RAG/data/crud_split/split_merged.json --max-samples 50 --output-dir datasets/optimized_ingest --custom-config configs/full_hybrid_only.json
```
- 预计时间: 5-8分钟

### 步骤4: 评估结果
```powershell
# 转换数据格式
.venv\Scripts\python.exe -c "import json, pandas as pd; data = json.load(open('datasets/optimized_ingest/comparison_results_50.json', 'r', encoding='utf-8')); strategy = next(s for s in data['strategies'] if s['strategy'] == 'full_hybrid'); df = pd.DataFrame([{'ID': r['question'][:50], 'ground_truth_text': r['reference_answer'], 'generated_text': r['generated_answer']} for r in strategy['results']]); df.to_csv('datasets/optimized_ingest/ragquesteval_input.csv', index=False, encoding='utf-8'); print(f'转换完成，共 {len(df)} 条数据')"

# 使用优化后的test_ragquesteval.py评估
.venv\Scripts\python.exe scripts/test_ragquesteval.py --result-file datasets/optimized_ingest/ragquesteval_input.csv --output-dir datasets/optimized_ingest --save-quest-gt
```
- 预计时间: 5-8分钟

### 步骤5: 对比结果
对比 datasets/optimized_ingest 和 datasets/final_test_50 的结果

## 📊 预期结果

- **当前最优**: F1 = 59.68%
- **优化后预期**: F1 = 64-67%
- **距离70%目标**: 差距缩小到3-6%

## 💾 重要的数据文件

### 保留的实验结果（用于对比）
- `datasets/final_test_50/` - 当前最优结果（50样本，top_k=8, rerank=4）
- `datasets/baseline_20/` - 20样本基线结果
- `datasets/optimized_ingest/` - 新的优化入库测试（即将生成）

### 配置文件
- `.env.backup` - 原始配置备份
- `configs/full_hybrid_only.json` - full_hybrid策略配置

## ⏱️ 总耗时估算（步骤2之后）

- 步骤3: 5-8分钟
- 步骤4: 5-8分钟
- **总计: 约15-20分钟**

## 🔄 恢复配置（可选）

如果需要恢复到原始配置：
```powershell
Copy-Item "D:\graduation project\.env.backup" "D:\graduation project\.env"
```

---

**创建时间**: 2026-04-16
**状态**: 步骤2进行中
**下一步**: 等待步骤2完成后执行步骤3