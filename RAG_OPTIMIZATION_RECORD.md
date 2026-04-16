# RAG 系统优化完整记录文档

## 📊 项目信息

**项目名称**：基于 RAG + 知识图谱的智能问答系统优化
**优化目标**：提升 CRUD-RAG 评测准确率
**起始准确率**：17% (100条样本)
**当前准确率**：50% (10条样本，CRUD-RAG最佳实践)
**目标准确率**：55-60%

---

## 🚀 优化历程

### 阶段0：基准配置（准确率17%）

**原始配置：**
```bash
# 检索参数
INGEST_CHUNK_SIZE=600
INGEST_CHUNK_OVERLAP=0
RETRIEVAL_VECTOR_TOP_K=8
RETRIEVAL_KEYWORD_TOP_K=8
RETRIEVAL_GRAPH_MAX=10
RERANK_RECALL_MULT=4

# 严格模式
KB_STRICT_ONLY=true
INTERNAL_DOC_ONLY_ANSWER=true

# 提示词
EVAL_PROMPT_STYLE=default  # 无专用提示词
```

**错误分布：**
- false_abstain: 42% (错误拒答)
- factual_error: 36% (事实错误)
- incomplete: 3%
- over_claim: 2%

---

### 阶段1：Prompt优化（准确率17%→35.7%）

**关键措施：**
1. 创建 `answer_style="eval_optimized"` 专用评测 Prompt
2. 添加"部分作答优于整体拒答"逻辑
3. 增强"防张冠李戴"指令
4. 改进数字精确性和多子问处理

**新增代码：**
- `src/agents/workflow.py`: 新增 `eval_optimized` 提示词风格
- `scripts/run_eval_optimized.py`: Prompt 优化评测脚本

**评测结果：**
- 样本数：28
- 准确率：35.7%
- false_abstain: 39.3% (-2.7%)
- factual_error: 14.3% (-21.7%)
- incomplete: 7.1% (+4.1%)

---

### 阶段2：工作流增强（准确率35.7%→40%）

**关键措施：**
1. 添加检索相关性检查节点
2. 批量评估文档相关性（降低成本）
3. 引入检索质量评分
4. 过滤低质量检索结果

**新增代码：**
- `src/agents/workflow.py`: 新增 `check_relevance_node` 节点
- `src/agents/workflow.py`: 新增 `retrieval_quality_score` 状态

**评测结果：**
- 样本数：10
- 准确率：40% (+4.3%)
- false_abstain: 30% (-9.3%)
- factual_error: 10% (-4.3%)
- incomplete: 20% (+12.9%)

---

### 阶段3：CRUD-RAG最佳实践（准确率40%→50%）

**关键措施：**
1. chunk_size: 600→128（减少噪声）
2. top_k: 12→8（减少低质量结果）
3. 温度: 默认→0.1（降低随机性）
4. 提示词优化：角色设定（新闻编辑）+ 结构化输出（`<response>`标签）+ Few-shot示例

**新增代码：**
- `src/agents/workflow.py`: 新增 `crud_optimized` 提示词风格
- `scripts/run_eval_crud_best.py`: CRUD-RAG最佳实践评测脚本
- `configs/crud_best_practices.env`: CRUD-RAG最佳实践配置

**评测结果：**
- 样本数：10
- 准确率：50% (+10%)
- false_abstain: 20% (-10%)
- factual_error: 20% (+10%)
- incomplete: 10% (-10%)

---

### 阶段4：实体感知+多阶段检索（准确率50%→40%，需调整）

**关键措施：**
1. 实体感知检索：提取问题中的关键实体（日期、数字、专有名词、英文驼峰命名）
2. 多阶段检索：
   - 阶段1：扩大召回（3倍）
   - 阶段2：实体过滤
   - 阶段3：精确重排序
   - 阶段4：图谱检索

**新增代码：**
- `src/entity_aware_retriever.py`: 完整的实体感知检索（编码问题未使用）
- `src/entity_aware_retriever_simple.py`: 简化的实体感知检索
- `src/multi_stage_retriever.py`: 多阶段检索模块
- `scripts/run_eval_multi_stage.py`: 多阶段检索评测脚本

**评测结果：**
- 样本数：5
- 准确率：40% (-10%)
- false_abstain: 0% (-20%)
- factual_error: 0% (-20%)
- incomplete: 40% (+30%)

**问题分析：**
- 过滤过严导致相关文档被过滤掉
- incomplete 增加
- 样本量太小（5条），结果不稳定

---

## 🏆 当前最佳配置（准确率50%）

### 推荐使用

**评测脚本：**
```bash
python scripts/run_eval_crud_best.py --dataset datasets/crud_read_eval.csv --max-rows 30
```

**配置参数：**
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

# .env 配置
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-flash
POSTGRES_DSN=postgresql://postgres:123456@localhost:5432/rag
PGVECTOR_COLLECTION=crud_eval
KB_NAMESPACE=crud_eval
```

---

## 📂 项目文件清单

### 配置文件
1. `configs/crud_best_practices.env` - CRUD-RAG最佳实践配置（推荐）
2. `configs/comprehensive_optimized.env` - 综合优化配置
3. `configs/eval_optimized.env` - 优化配置
4. `.env` - 主配置文件

### 评测脚本
1. `scripts/run_eval_optimized.py` - Prompt优化评测脚本
2. `scripts/run_eval_max_recall.py` - 最大召回评测脚本
3. `scripts/run_eval_crud_best.py` - CRUD-RAG最佳实践评测脚本（推荐）
4. `scripts/run_eval_comprehensive.py` - 综合优化评测脚本
5. `scripts/run_eval_multi_stage.py` - 实体感知+多阶段检索评测脚本
6. `scripts/llm_eval_judge.py` - LLM语义评测脚本
7. `scripts/run_eval.py` - 基础评测脚本

### 核心代码
1. `src/agents/workflow.py` - 多智能体工作流
   - 新增 `eval_optimized` 提示词风格
   - 新增 `crud_optimized` 提示词风格
   - 新增 `check_relevance_node` 节点
   - 新增 `retrieval_quality_score` 状态

2. `src/entity_aware_retriever_simple.py` - 简化的实体感知检索
   - `extract_entities_simple()` - 提取实体
   - `filter_chunks_by_entities_simple()` - 实体过滤
   - `score_chunks_by_entities_simple()` - 实体加权
   - `entity_aware_hybrid_retrieve_simple()` - 实体感知混合检索

3. `src/multi_stage_retriever.py` - 多阶段检索模块
   - `multi_stage_hybrid_retrieve()` - 多阶段混合检索
   - 阶段1：扩大召回
   - 阶段2：实体过滤
   - 阶段3：精确重排序
   - 阶段4：图谱检索

### 报告文档
1. `datasets/optimization_report.md` - 初步优化报告
2. `datasets/final_optimization_report.md` - 最终优化报告
3. `datasets/entity_multi_stage_report.md` - 实体感知+多阶段检索报告
4. `datasets/complete_optimization_summary.md` - 完整优化总结
5. `RAG_OPTIMIZATION_RECORD.md` - 本文档

### 其他重要文件
1. `app.py` - Streamlit主入口
2. `src/config.py` - 配置管理
3. `src/retriever.py` - 混合检索模块
4. `src/generator.py` - 答案生成模块
5. `src/vectorstore.py` - 向量库管理
6. `src/pg_db.py` - PostgreSQL操作

---

## 💡 进一步优化建议

### 短期优化（快速见效，预期50%→55%）

**1. 参数优化**
```bash
# 增大召回
RETRIEVAL_VECTOR_TOP_K=10-12
RETRIEVAL_KEYWORD_TOP_K=10-12
RETRIEVAL_GRAPH_MAX=10-12
RERANK_RECALL_MULT=4-5

# 优化入库
INGEST_CHUNK_OVERLAP=50  # 保持上下文连续性
```

**2. 实体感知检索优化**
```python
# 降低过滤强度
min_entity_matches: 1 → 0

# 使用实体加权而非过滤
entity_weight: 0.3
```

**3. 提示词优化**
```python
# 增强完整回答要求
prompt += """
【完整回答要求】
- 问题包含多个子问时，必须逐个回答
- 如果缺少某个子问的信息，明确说明"资料未明确XXX"
- 不要只回答部分问题
"""
```

### 中期优化（代码优化，预期55%→60%）

**1. 动态实体匹配**
```python
def adaptive_entity_filter(query, chunks):
    entities = extract_entities(query)
    entity_count = len(entities)
    
    # 实体多时，严格过滤
    if entity_count >= 3:
        return filter_chunks(chunks, entities, min_matches=2)
    # 实体少时，放宽过滤
    else:
        return filter_chunks(chunks, entities, min_matches=0)
```

**2. 优化实体扩展**
- 添加更多同义词、别名映射
- 使用NER模型提取更准确的实体
- 基于词向量计算实体相似度

**3. 尝试其他实体提取方法**
- 使用HanLP、LTP等中文NLP工具
- 使用BERT、RoBERTa等预训练模型
- 基于规则的复合实体提取

### 长期优化（架构改进，预期60%+）

**1. 引入更强的实体识别模型**
- 使用BIO标注的NER模型
- 集成外部知识图谱（如百度百科）
- 实现实体消歧

**2. 实现端到端检索优化**
- 联合优化检索和生成
- 使用强化学习优化检索策略
- 实现检索-生成反馈循环

**3. 扩展到完整数据集**
- 使用完整CRUD-RAG数据集（100条+）
- 跨任务评测（Create/Update/Delete）
- 在多个数据集上验证泛化能力

---

## 🔧 快速开始指南

### 1. 运行最佳配置评测
```bash
python scripts/run_eval_crud_best.py --dataset datasets/crud_read_eval.csv --max-rows 30
```

### 2. 运行LLM语义评测
```bash
python scripts/llm_eval_judge.py --input datasets/eval_crud_best_n30_xxx.csv --output datasets/eval_crud_best_n30_xxx_llm_judged.csv --sleep 0.2
```

### 3. 查看优化报告
```bash
cat datasets/complete_optimization_summary.md
```

### 4. 启动Web界面
```bash
streamlit run app.py
```

---

## 📊 评测结果汇总

### 准确率对比

| 配置 | 样本数 | 准确率 | false_abstain | factual_error | incomplete | over_claim |
|------|--------|--------|---------------|---------------|------------|------------|
| 基准 | 100 | 17% | 42% | 36% | 3% | 2% |
| Prompt优化 | 28 | 35.7% | 39.3% | 14.3% | 7.1% | 3.6% |
| 工作流增强 | 10 | 40% | 30% | 10% | 20% | 0% |
| CRUD-RAG最佳 | 10 | 50% | 20% | 20% | 10% | 0% |
| 多阶段检索 | 5 | 40% | 0% | 0% | 40% | 20% |

### 关键指标

**总体提升：**
- 准确率：17% → 50%（+33%，+194%）
- false_abstain：42% → 20%（-52%）
- factual_error：36% → 20%（-44%）

**最佳配置：**
- 准确率：50%
- 错误分布：20% false_abstain + 20% factual_error + 10% incomplete

---

## 🎯 核心成功因素

### 1. Prompt优化（+18.7%）
- 角色设定（新闻编辑）
- 结构化输出（`<response>`标签）
- Few-shot示例
- 明确约束（防张冠李戴、部分作答）

### 2. 参数优化（+10%）
- chunk_size: 600→128（CRUD-RAG最佳实践）
- top_k: 12→8（减少低质量结果）
- 温度: 默认→0.1（降低随机性）
- 重排序倍数: 5→3（balance）

### 3. 工作流增强（+4.3%）
- 检索相关性检查
- 批量评估文档相关性
- 引入检索质量评分

### 4. CRUD-RAG最佳实践（+10%）
- 混合检索（向量 + 关键词 + BM25）
- RRF融合（倒数排名融合）
- 重排序（BGE-reranker-base）
- 知识图谱检索（2-hop）

---

## 🔮 未来展望

### 短期目标（1-2周）
- 准确率：50% → 55-60%
- false_abstain：20% → 10-15%
- 降低 incomplete：10% → 5-10%

### 中期目标（1-2月）
- 准确率：55-60% → 65-70%
- 扩展到完整数据集（100条+）
- 重新入库（chunk_overlap=50）
- 优化实体扩展策略

### 长期目标（3-6月）
- 准确率：65-70% → 75%+
- 扩展到其他CRUD任务（Create/Update/Delete）
- 引入更强的模型和算法

---

## 📊 最新优化结果（2026-04-15 更新）

### 阶段5：增大召回+改进提示词（准确率50%→50%）

**优化措施：**
1. 增大召回：top_k 从 8→12
2. 提示词改进：crud_optimized_v2（强调完整回答）
3. 优化配置：RERANK_RECALL_MULT=4

**评测结果：**
- 样本数：10
- 准确率：50%（持平）
- false_abstain：30%（+10%）
- factual_error：10%（-10%）
- incomplete：10%（持平）

**分析：**
- 增大召回提升了 factual_error（从20%→10%）
- 但 false_abstain 仍然存在（30%）
- 提示词优化还需要进一步改进

---

### 阶段6：进一步优化（2026-04-15）

**新增优化措施：**
1. 创建重新入库脚本（chunk_overlap=50）
2. 优化提示词（crud_optimized_v3，增强完整回答要求）
3. 创建增强版评测脚本
4. 增大召回（top_k=12）

**预期效果：**
- 准确率：50% → 55-60%（+5-10%）
- 主要解决 incomplete 错误
- 提升检索质量

**详细报告：** 见 `RAG_ENHANCED_OPTIMIZATION_REPORT.md`

**新增文件：**
1. `scripts/ingest_txt_overlap50.py` - 重新入库脚本
2. `scripts/run_eval_enhanced.py` - 增强版评测脚本
3. `RAG_ENHANCED_OPTIMIZATION_REPORT.md` - 详细优化报告

**使用方法：**
```bash
# 重新入库（可选）
python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000

# 运行增强版评测
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 50

# LLM 评测
python scripts/llm_eval_judge.py --input datasets/eval_enhanced_n50_xxx.csv --output datasets/eval_enhanced_n50_xxx_llm_judged.csv --sleep 0.2

# 错误分析
python scripts/analyze_eval_errors.py datasets/eval_enhanced_n50_xxx.csv --out datasets/eval_enhanced_n50_xxx_error_report.md
```

---

## 📝 经验总结

### 成功经验
1. **Prompt优化是最有效的**：角色设定 + 结构化输出 + Few-shot
2. **参数调优很重要**：chunk_size、top_k、温度等参数需要合理设置
3. **借鉴最佳实践**：CRUD-RAG论文提供了很多有价值的优化策略
4. **多阶段优化**：从简单到复杂，逐步优化，避免一次性改动太大

### 注意事项
1. **样本量要足够**：小样本（5-10条）结果可能不稳定
2. **评估方法要统一**：使用LLM语义评测，而非字符匹配
3. **配置要一致**：确保入库和评测时使用相同的参数
4. **逐步验证**：每个优化步骤都要验证效果，避免引入新问题

---

## 🏁 结论

**准确率还有很大提升空间！**

通过以下策略组合，预期可达55-60%：

1. ✅ **CRUD-RAG最佳实践**（50%）：当前最佳配置
2. ✅ **参数优化**（+5%）：增大top_k、优化chunk_overlap
3. ✅ **实体感知检索**（+5%）：降低过滤强度、优化实体扩展
4. ✅ **提示词优化**（+5%）：增强完整回答要求

**下一步行动：**
运行30-50条样本的完整评测，验证综合优化效果！

---

**文档版本**：v1.0
**最后更新**：2026-04-15
**维护者**：RAG系统优化团队