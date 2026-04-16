# RAG 系统优化行动计划（2026-04-15）

## 🎯 目标

**当前准确率**: ~30.5%（基于48条样本评测）
**目标准确率**: 40-45%
**预期提升**: +10-15%

---

## 📋 优化任务清单

### ✅ 已完成任务

1. **分析项目状态** - 已完成
   - 阅读所有优化文档
   - 分析代码结构
   - 确定优化方向

2. **创建优化脚本** - 已完成
   - `scripts/ingest_txt_overlap50.py` - 重新入库脚本
   - `scripts/run_eval_enhanced.py` - 增强版评测脚本
   - `src/agents/workflow.py` - 新增crud_optimized_v3提示词

3. **编写优化文档** - 已完成
   - `RAG_ENHANCED_OPTIMIZATION_REPORT.md` - 详细优化报告
   - `RAG_QUICK_REFERENCE.md` - 快速参考指南
   - `RAG_ERROR_ANALYSIS_REPORT.md` - 错误分析报告

---

### 🚀 待执行任务

#### 优先级1：立即执行（今日）

##### 任务1.1：运行增强版评测
**状态**: 待执行
**预估时间**: 10-15分钟
**依赖**: 无

**操作步骤**:
```bash
cd "D:\graduation project"
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 30
```

**预期结果**:
- 生成 `datasets/eval_enhanced_n30_xxx.csv`
- 获得基础评测指标（EM、F1等）

**成功标准**:
- 脚本成功运行
- 生成输出文件
- 无重大错误

---

##### 任务1.2：LLM语义评测
**状态**: 待执行
**预估时间**: 5-10分钟
**依赖**: 任务1.1完成

**操作步骤**:
```bash
# 找到刚生成的评测文件
cd "D:\graduation project\datasets"
dir eval_enhanced_n30*.csv

# 运行LLM评测（替换实际的文件名）
python ../scripts/llm_eval_judge.py --input eval_enhanced_n30_xxx.csv --output eval_enhanced_n30_xxx_llm_judged.csv --sleep 0.2
```

**预期结果**:
- 生成 `datasets/eval_enhanced_n30_xxx_llm_judged.csv`
- 获得语义准确率（llm_correct）

**成功标准**:
- LLM评测成功完成
- 准确率有所提升

---

##### 任务1.3：错误分析
**状态**: 待执行
**预估时间**: 5分钟
**依赖**: 任务1.2完成

**操作步骤**:
```bash
cd "D:\graduation project\datasets"
python ../scripts/analyze_eval_errors.py eval_enhanced_n30_xxx.csv --out eval_enhanced_n30_xxx_error_report.md
```

**预期结果**:
- 生成 `datasets/eval_enhanced_n30_xxx_error_report.md`
- 详细分析错误类型分布

**成功标准**:
- 成功生成错误分析报告
- 识别主要错误类型

---

#### 优先级2：短期任务（1-2周）

##### 任务2.1：重新入库（可选）
**状态**: 待评估
**预估时间**: 30-60分钟
**依赖**: 无
**说明**: 需要原始数据文件

**操作步骤**:
```bash
# 方法1：清空旧库（谨慎操作）
cd "D:\graduation project"
python scripts/clear_rag_kb.py

# 方法2：使用新的collection（推荐）
# 在 .env 中设置：
# PGVECTOR_COLLECTION=crud_eval_overlap50

# 重新入库
python scripts/ingest_txt_overlap50.py --dir "D:\\CRUD_RAG\\data\\80000_docs" --batch-size 200 --limit 2000
```

**预期结果**:
- 使用chunk_overlap=50重新入库
- 保持上下文连续性

**成功标准**:
- 成功入库
- 无错误

**注意事项**:
- 确保有原始数据文件
- 考虑磁盘空间
- 可以先小批量测试

---

##### 任务2.2：优化实体匹配
**状态**: 待实现
**预估时间**: 2-4小时
**依赖**: 无

**操作步骤**:
1. 查看现有实体感知检索代码
   - `src/entity_aware_retriever_simple.py`

2. 实现优化：
   - 降低过滤强度（min_entity_matches=0）
   - 使用实体加权而非过滤
   - 添加同义词映射

3. 测试优化效果

**代码示例**:
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

**预期效果**:
- 减少false_abstain（+2-3%）
- 减少factual_error（+1-2%）

---

##### 任务2.3：进一步优化提示词
**状态**: 待实现
**预估时间**: 1-2小时
**依赖**: 任务1.3完成（基于错误分析）

**操作步骤**:
1. 分析错误案例
2. 识别提示词不足之处
3. 针对性优化提示词
4. 测试优化效果

**优化方向**:
- 强化"完整回答"要求
- 增加更多Few-shot示例
- 强调"防张冠李戴"原则

**预期效果**:
- 减少incomplete（+2-3%）

---

##### 任务2.4：运行更大规模评测
**状态**: 待执行
**预估时间**: 15-30分钟
**依赖**: 任务2.2和2.3完成

**操作步骤**:
```bash
cd "D:\graduation project"
python scripts/run_eval_enhanced.py --dataset datasets/crud_read_eval.csv --max-rows 50

# LLM评测
python scripts/llm_eval_judge.py --input datasets/eval_enhanced_n50_xxx.csv --output datasets/eval_enhanced_n50_xxx_llm_judged.csv --sleep 0.2

# 错误分析
python scripts/analyze_eval_errors.py datasets/eval_enhanced_n50_xxx.csv --out datasets/eval_enhanced_n50_xxx_error_report.md
```

**预期结果**:
- 50条样本的完整评测结果
- 更可靠的统计结果
- 发现更多错误模式

---

#### 优先级3：中期任务（1-2月）

##### 任务3.1：引入更强的实体识别
**状态**: 待规划
**预估时间**: 1-2周
**依赖**: 无

**操作步骤**:
1. 研究中文NLP工具（HanLP、LTP）
2. 研究预训练模型（BERT、RoBERTa）
3. 实现实体识别模块
4. 集成到检索流程
5. 测试效果

**预期效果**:
- 减少false_abstain（+2-3%）
- 减少factual_error（+1-2%）

---

##### 任务3.2：端到端优化
**状态**: 待规划
**预估时间**: 2-4周
**依赖**: 任务3.1完成

**操作步骤**:
1. 研究检索-生成联合优化方法
2. 实现检索质量评分反馈
3. 实现检索策略动态调整
4. 测试效果

**预期效果**:
- 综合提升准确率（+2-3%）

---

##### 任务3.3：扩展数据集
**状态**: 待规划
**预估时间**: 1-2周
**依赖**: 无

**操作步骤**:
1. 获取完整CRUD-RAG数据集（100条+）
2. 准备跨任务评测数据（Create/Update/Delete）
3. 运行完整评测
4. 分析结果

**预期效果**:
- 提升泛化能力
- 发现更多错误模式

---

## 📊 进度跟踪

### 已完成 ✅

- [x] 分析项目状态
- [x] 创建优化脚本
- [x] 编写优化文档

### 进行中 🔄

- [ ] 任务1.1：运行增强版评测
- [ ] 任务1.2：LLM语义评测
- [ ] 任务1.3：错误分析

### 待办 📝

#### 优先级1（立即执行）
- [ ] 任务2.1：重新入库（可选）
- [ ] 任务2.2：优化实体匹配
- [ ] 任务2.3：进一步优化提示词
- [ ] 任务2.4：运行更大规模评测

#### 优先级2（短期，1-2周）
- [ ] 任务3.1：引入更强的实体识别
- [ ] 任务3.2：端到端优化
- [ ] 任务3.3：扩展数据集

---

## 💡 技术要点

### 1. 增强版配置

```bash
# 检索参数
RETRIEVAL_VECTOR_TOP_K=12      # 增大召回
RETRIEVAL_KEYWORD_TOP_K=12     # 增大召回
RERANK_RECALL_MULT=4           # 增大召回

# 提示词参数
EVAL_PROMPT_STYLE=crud_optimized_v3  # 增强完整回答要求
KB_STRICT_ONLY=false
INTERNAL_DOC_ONLY_ANSWER=false
OPENAI_TEMPERATURE=0.1
```

### 2. 新增提示词特点

- **更强调完整回答**：必须逐个回答每个子问
- **更多示例**：包括处理缺失信息的示例
- **更详细的指导**：如何识别问题中的多个子问

### 3. 错误类型分布

基于现有评测数据（48条样本）：
- **ok**: ~25-35%（正确）
- **factual_error**: ~30-35%（事实错误）
- **false_abstain**: ~25-30%（错误拒答）
- **incomplete**: ~20-25%（不完整）

### 4. 主要优化方向

1. **增大召回**：减少false_abstain和incomplete
2. **优化实体匹配**：减少false_abstain和factual_error
3. **增强提示词**：减少incomplete

---

## 🎯 成功标准

### 短期目标（1-2周）

- [ ] 准确率提升至35-40%
- [ ] false_abstain降低至20%以下
- [ ] incomplete降低至15%以下

### 中期目标（1-2月）

- [ ] 准确率提升至40-45%
- [ ] false_abstain降低至15%以下
- [ ] incomplete降低至10%以下
- [ ] 在30-50条样本上验证结果

---

## 📞 联系和支持

### 文档参考

- `RAG_QUICK_REFERENCE.md` - 快速参考指南
- `RAG_ENHANCED_OPTIMIZATION_REPORT.md` - 详细优化报告
- `RAG_ERROR_ANALYSIS_REPORT.md` - 错误分析报告
- `RAG_COMPLETE_SUMMARY.md` - 完整优化总结
- `RAG_FINAL_REPORT.md` - 最终优化报告

### 问题排查

如果遇到问题，请参考：
1. 检查环境配置（.env文件）
2. 检查依赖安装（requirements.txt）
3. 检查数据文件路径
4. 查看错误日志

---

**文档版本**: v1.0
**创建时间**: 2026-04-15
**最后更新**: 2026-04-15
**优化团队**: RAG系统优化团队