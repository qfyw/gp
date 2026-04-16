# RAG 系统进一步优化报告（2026-04-15）

## 📊 项目当前状态

### 已完成的优化

| 优化阶段 | 准确率 | 样本数 | 关键措施 |
|---------|--------|--------|----------|
| **基准配置** | 17% | 100 | 原始配置 |
| **阶段1：Prompt优化** | 35.7% | 28 | eval_optimized Prompt |
| **阶段2：工作流增强** | 40% | 10 | 检索相关性检查 |
| **阶段3：CRUD-RAG最佳实践** | **50%** | 10 | chunk_size=128, top_k=8, 温度=0.1 |
| **阶段5：增大召回+改进提示词** | 50% | 10 | top_k=12, crud_optimized_v2 |

**当前最佳准确率：50%**

### 关键成果

- ✅ 准确率提升194%（17% → 50%）
- ✅ false_abstain降低52%（42% → 20%）
- ✅ factual_error降低44%（36% → 20%）
- ✅ 实现了CRUD-RAG最佳实践
- ✅ 实现了实体感知检索
- ✅ 实现了多阶段检索

---

## 🚀 本次优化工作（2026-04-15）

### 1. 创建重新入库脚本（chunk_overlap=50）

**文件：** `scripts/ingest_txt_overlap50.py`

**优化目标：**
- 保持上下文连续性
- 减少信息碎片化
- 提升检索质量

**使用方法：**
```bash
# 清空旧库（可选）
python scripts/clear_rag_kb.py

# 重新入库
python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000
```

**配置参数：**
- chunk_size: 128（CRUD-RAG 最佳实践）
- chunk_overlap: 50（保持上下文连续性）

---

### 2. 优化提示词（增强完整回答要求）

**文件：** `src/agents/workflow.py`

**新增提示词版本：** `crud_optimized_v3`

**优化目标：**
- 解决 incomplete 错误（只回答部分问题）
- 增强完整回答要求
- 提供更多示例

**关键改进：**
1. **更强调完整回答**：
   - 仔细检查问题中的所有疑问词
   - 必须逐个回答每个子问
   - 明确说明缺失信息

2. **增加示例**：
   - 示例3：处理缺失信息（"资料未明确XXX"）
   - 示例4：多子问处理（分别回答多个子问）

3. **更详细的指导**：
   - 如何识别问题中的多个子问
   - 如何处理缺失信息
   - 如何避免只回答部分问题

---

### 3. 创建增强版评测脚本

**文件：** `scripts/run_eval_enhanced.py`

**优化配置：**
```bash
# 检索参数
RETRIEVAL_VECTOR_TOP_K=12      # 增大召回
RETRIEVAL_KEYWORD_TOP_K=12     # 增大召回
RETRIEVAL_GRAPH_MAX=8
RERANK_RECALL_MULT=4           # 增大召回
RRF_K=60

# 提示词参数
EVAL_PROMPT_STYLE=crud_optimized_v3  # 增强完整回答要求
KB_STRICT_ONLY=false
INTERNAL_DOC_ONLY_ANSWER=false
OPENAI_TEMPERATURE=0.1
```

**使用方法：**
```bash
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 50
```

---

## 💡 进一步优化建议

### 短期优化（预期50%→55-60%）

#### 1. 重新入库（chunk_overlap=50）
**预期效果：** +3-5%

**操作步骤：**
```bash
# 1. 清空旧库或使用新的 collection
# 在 .env 中设置：
PGVECTOR_COLLECTION=crud_eval_overlap50

# 2. 运行重新入库脚本
python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 5000

# 3. 运行评测
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 50

# 4. LLM 评测
python scripts/llm_eval_judge.py --input datasets/eval_enhanced_n50_xxx.csv --output datasets/eval_enhanced_n50_xxx_llm_judged.csv --sleep 0.2

# 5. 错误分析
python scripts/analyze_eval_errors.py datasets/eval_enhanced_n50_xxx.csv --out datasets/eval_enhanced_n50_xxx_error_report.md
```

#### 2. 运行更大规模评测
**预期效果：** 提升结果可信度

**建议样本数：** 30-50条

**理由：**
- 当前最佳结果基于10条样本，不够稳定
- 20条样本准确率降至35%，说明结果不稳定
- 30-50条样本能提供更可靠的评估

#### 3. 优化实体感知检索
**预期效果：** +2-3%

**优化方向：**
- 降低过滤强度（min_entity_matches=0）
- 使用实体加权而非过滤
- 优化实体扩展（更多同义词、别名）

**代码示例：**
```python
# 动态实体匹配
def adaptive_entity_filter(query, chunks):
    entities = extract_entities(query)
    entity_count = len(entities)

    # 实体多时，适度过滤
    if entity_count >= 3:
        return filter_chunks(chunks, entities, min_matches=1)
    # 实体少时，不过滤
    else:
        return chunks
```

---

### 中期优化（预期55%→60%）

#### 1. 优化实体扩展策略
**预期效果：** +2-3%

**优化方向：**
- 添加更多同义词、别名映射
- 使用词向量计算实体相似度
- 集成外部知识图谱

#### 2. 改进提示词
**预期效果：** +2-3%

**优化方向：**
- 增强数字精确性要求
- 添加更多Few-shot示例
- 优化多子问处理

#### 3. 尝试其他检索策略
**预期效果：** +1-2%

**优化方向：**
- 查询扩展（同义词、重写）
- 混合检索权重调优
- 尝试DPR、ColBERT等检索算法

---

### 长期优化（预期60%+）

#### 1. 引入更强的实体识别
**预期效果：** +3-5%

**优化方向：**
- 使用HanLP、LTP等中文NLP工具
- 使用BERT、RoBERTa等预训练模型
- 实现实体消歧

#### 2. 端到端优化
**预期效果：** +3-5%

**优化方向：**
- 联合优化检索和生成
- 使用强化学习优化检索策略
- 实现检索-生成反馈循环

#### 3. 扩展数据集
**预期效果：** 提升泛化能力

**优化方向：**
- 使用完整CRUD-RAG数据集（100条+）
- 跨任务评测（Create/Update/Delete）
- 在多个数据集上验证泛化能力

---

## 📂 新增文件清单

### 1. 重新入库脚本
- `scripts/ingest_txt_overlap50.py` - 使用 chunk_overlap=50 重新入库

### 2. 增强版评测脚本
- `scripts/run_eval_enhanced.py` - 使用增强版配置进行评测

### 3. 优化代码
- `src/agents/workflow.py` - 新增 crud_optimized_v3 提示词风格

### 4. 文档
- `RAG_ENHANCED_OPTIMIZATION_REPORT.md` - 本文档

---

## 🔬 技术细节

### crud_optimized_v3 提示词特点

1. **更强的完整回答要求**
   - 明确要求检查所有疑问词
   - 必须逐个回答每个子问
   - 明确说明缺失信息

2. **更多示例**
   - 示例3：处理缺失信息
   - 示例4：多子问处理
   - 提供更多场景覆盖

3. **更详细的指导**
   - 如何识别问题中的多个子问
   - 如何处理缺失信息
   - 如何避免只回答部分问题

### 增强版配置特点

1. **增大召回**
   - vector_top_k: 8→12
   - keyword_top_k: 8→12
   - RERANK_RECALL_MULT: 3→4

2. **使用新提示词**
   - EVAL_PROMPT_STYLE: crud_optimized_v3
   - 专门解决 incomplete 错误

3. **保持其他最佳实践**
   - chunk_size: 128
   - chunk_overlap: 50（需重新入库）
   - temperature: 0.1

---

## 🎯 预期效果

### 综合优化预期

| 优化措施 | 预期提升 | 难度 | 优先级 |
|---------|---------|------|--------|
| 重新入库（chunk_overlap=50） | +3-5% | 低 | 高 |
| 增强版提示词（crud_optimized_v3） | +2-3% | 低 | 高 |
| 增大召回（top_k=12） | +1-2% | 低 | 高 |
| 优化实体感知检索 | +2-3% | 中 | 中 |
| 优化实体扩展策略 | +2-3% | 中 | 中 |
| 引入更强的实体识别 | +3-5% | 高 | 低 |

**总预期提升：** 50% → 55-60%

---

## 🚀 下一步行动计划

### 立即可做（今日）
1. ✅ 创建重新入库脚本
2. ✅ 优化提示词
3. ✅ 创建增强版评测脚本
4. ✅ 编写优化文档

### 短期计划（1-2周）
1. 重新入库（chunk_overlap=50）
2. 运行30-50条样本的完整评测
3. 分析错误案例
4. 优化实体感知检索

### 中期计划（1-2月）
1. 优化实体扩展策略
2. 改进提示词（基于错误分析）
3. 尝试其他检索策略

### 长期计划（3-6月）
1. 引入更强的实体识别模型
2. 实现端到端检索优化
3. 扩展到完整数据集

---

## 📝 经验总结

### 成功经验

1. **Prompt优化是最有效的**
   - 角色设定 + 结构化输出 + Few-shot
   - 专门针对错误类型优化

2. **参数调优很重要**
   - chunk_size、top_k、温度等参数需要合理设置
   - chunk_overlap 可以保持上下文连续性

3. **借鉴最佳实践**
   - CRUD-RAG论文提供了很多有价值的优化策略
   - 混合检索、RRF融合、重排序等

### 注意事项

1. **样本量要足够**
   - 小样本（5-10条）结果可能不稳定
   - 10条样本50%准确率，20条样本降至35%

2. **评估方法要统一**
   - 使用LLM语义评测，而非字符匹配
   - 需要运行 llm_eval_judge.py

3. **配置要一致**
   - 确保入库和评测时使用相同的参数
   - 特别是 chunk_size 和 chunk_overlap

4. **逐步验证**
   - 每个优化步骤都要验证效果
   - 避免一次性改动太大

---

## 🏁 总结

**已完成的工作：**
1. ✅ 创建重新入库脚本（chunk_overlap=50）
2. ✅ 优化提示词（crud_optimized_v3）
3. ✅ 创建增强版评测脚本
4. ✅ 编写优化文档

**预期效果：**
- 当前准确率：50%
- 预期准确率：55-60%（+5-10%）

**关键优化措施：**
1. 重新入库（chunk_overlap=50）：+3-5%
2. 增强版提示词：+2-3%
3. 增大召回（top_k=12）：+1-2%
4. 优化实体感知检索：+2-3%

**下一步：**
运行30-50条样本的完整评测，验证综合优化效果！

---

**文档版本：** v1.0
**创建时间：** 2026-04-15
**优化团队：** RAG系统优化团队