# 多阶段检索评测汇总

## 配置
- vector_top_k: 8
- keyword_top_k: 8
- graph_max: 8
- 阶段1召回倍数: 3
- 最小实体匹配数: 1
- 温度: 0.1
- 提示词: crud_optimized

## 指标
- 样本数: 5
- EM 均值: 0.0000
- 字符 F1 均值: 0.1976
- 延迟 P50: 11.7633s
- 延迟 P95: 27.3974s

## 下一步
运行 LLM 语义评测：
```bash
python scripts/llm_eval_judge.py --input D:\graduation project\datasets\eval_multi_stage_n5_1776253380.csv --output D:\graduation project\datasets\eval_multi_stage_n5_1776253380_llm_judged.csv --sleep 0.2
```
