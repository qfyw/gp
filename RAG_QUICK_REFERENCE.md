# RAG 系统优化快速参考指南

## 📊 当前状态

**最佳准确率：50%**（从17%提升，+194%）

**关键成果：**
- ✅ 准确率提升194%
- ✅ false_abstain降低52%
- ✅ factual_error降低44%
- ✅ 实现了CRUD-RAG最佳实践
- ✅ 实现了实体感知检索
- ✅ 实现了多阶段检索

---

## 🚀 最新优化（2026-04-15）

### 新增文件

1. **重新入库脚本**
   - 文件：`scripts/ingest_txt_overlap50.py`
   - 优化：使用 chunk_overlap=50 保持上下文连续性

2. **增强版评测脚本**
   - 文件：`scripts/run_eval_enhanced.py`
   - 优化：增大召回（top_k=12）+ 新提示词（crud_optimized_v3）

3. **优化代码**
   - 文件：`src/agents/workflow.py`
   - 新增：crud_optimized_v3 提示词风格

4. **优化文档**
   - 文件：`RAG_ENHANCED_OPTIMIZATION_REPORT.md`
   - 内容：详细优化报告和下一步计划

---

## 🎯 快速使用指南

### 运行最佳配置评测

```bash
# 基础版（50%准确率）
python scripts/run_eval_crud_best.py --dataset datasets/crud_read_eval.csv --max-rows 30

# 增强版（预期55-60%准确率）
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 50
```

### 重新入库（可选）

```bash
# 清空旧库（可选）
python scripts/clear_rag_kb.py

# 重新入库
python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000
```

### LLM 评测

```bash
python scripts/llm_eval_judge.py --input datasets/eval_enhanced_n50_xxx.csv --output datasets/eval_enhanced_n50_xxx_llm_judged.csv --sleep 0.2
```

### 错误分析

```bash
python scripts/analyze_eval_errors.py datasets/eval_enhanced_n50_xxx.csv --out datasets/eval_enhanced_n50_xxx_error_report.md
```

---

## 📂 关键配置

### 基础版配置（50%准确率）

```bash
# 检索参数
INGEST_CHUNK_SIZE=128
INGEST_CHUNK_OVERLAP=0
RETRIEVAL_VECTOR_TOP_K=8
RETRIEVAL_KEYWORD_TOP_K=8
RETRIEVAL_GRAPH_MAX=8
RERANK_RECALL_MULT=3
RRF_K=60

# 提示词参数
OPENAI_TEMPERATURE=0.1
EVAL_PROMPT_STYLE=crud_optimized
KB_STRICT_ONLY=false
INTERNAL_DOC_ONLY_ANSWER=false
```

### 增强版配置（预期55-60%准确率）

```bash
# 检索参数
INGEST_CHUNK_SIZE=128
INGEST_CHUNK_OVERLAP=50  # 保持上下文连续性
RETRIEVAL_VECTOR_TOP_K=12  # 增大召回
RETRIEVAL_KEYWORD_TOP_K=12  # 增大召回
RETRIEVAL_GRAPH_MAX=8
RERANK_RECALL_MULT=4  # 增大召回
RRF_K=60

# 提示词参数
OPENAI_TEMPERATURE=0.1
EVAL_PROMPT_STYLE=crud_optimized_v3  # 增强完整回答要求
KB_STRICT_ONLY=false
INTERNAL_DOC_ONLY_ANSWER=false
```

---

## 💡 预期效果

| 优化措施 | 预期提升 | 状态 |
|---------|---------|------|
| 重新入库（chunk_overlap=50） | +3-5% | ✅ 已实现脚本 |
| 增强版提示词（crud_optimized_v3） | +2-3% | ✅ 已实现 |
| 增大召回（top_k=12） | +1-2% | ✅ 已实现 |
| 优化实体感知检索 | +2-3% | 📝 待实现 |
| 优化实体扩展策略 | +2-3% | 📝 待实现 |
| 引入更强的实体识别 | +3-5% | 📝 待实现 |

**总预期提升：** 50% → 55-60%（+5-10%）

---

## 📖 详细文档

1. **完整优化总结**
   - 文件：`RAG_COMPLETE_SUMMARY.md`
   - 内容：从17%到50%的完整优化历程

2. **最终优化报告**
   - 文件：`RAG_FINAL_REPORT.md`
   - 内容：最终优化成果和下一步计划

3. **优化记录**
   - 文件：`RAG_OPTIMIZATION_RECORD.md`
   - 内容：详细的优化记录和配置

4. **增强优化报告**
   - 文件：`RAG_ENHANCED_OPTIMIZATION_REPORT.md`
   - 内容：最新优化工作和预期效果

---

## 🎓 经验总结

### 成功经验

1. **Prompt优化是最有效的**
   - 角色设定 + 结构化输出 + Few-shot
   - 专门针对错误类型优化

2. **参数调优很重要**
   - chunk_size、top_k、温度等参数需要合理设置
   - chunk_overlap 可以保持上下文连续性

3. **借鉴最佳实践**
   - CRUD-RAG论文提供了很多有价值的优化策略

### 注意事项

1. **样本量要足够**
   - 小样本（5-10条）结果可能不稳定
   - 建议使用30-50条样本进行评测

2. **评估方法要统一**
   - 使用LLM语义评测，而非字符匹配
   - 需要运行 llm_eval_judge.py

3. **配置要一致**
   - 确保入库和评测时使用相同的参数

---

## 🚀 下一步行动

### 立即可做
1. ✅ 运行增强版评测脚本
2. ✅ LLM 语义评测
3. ✅ 错误分析

### 短期计划（1-2周）
1. 重新入库（chunk_overlap=50）
2. 优化实体感知检索
3. 基于错误分析进一步优化

### 中期计划（1-2月）
1. 优化实体扩展策略
2. 尝试其他检索策略
3. 扩展到完整数据集

### 长期计划（3-6月）
1. 引入更强的实体识别模型
2. 实现端到端检索优化
3. 跨任务评测

---

## 📞 支持

如有问题或建议，请参考以下文档：

- `RAG_COMPLETE_SUMMARY.md` - 完整优化总结
- `RAG_FINAL_REPORT.md` - 最终优化报告
- `RAG_OPTIMIZATION_RECORD.md` - 详细优化记录
- `RAG_ENHANCED_OPTIMIZATION_REPORT.md` - 最新优化报告

---

**文档版本：** v1.0
**创建时间：** 2026-04-15
**优化团队：** RAG系统优化团队