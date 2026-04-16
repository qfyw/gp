# 评测汇总

- 时间（UTC）: `2026-04-15T11:17:50.214398+00:00`
- 数据集: `D:\graduation project\datasets\crud_read_eval.csv`
- 逐条 CSV: `D:\graduation project\datasets\eval_max_recall_n10_1776251621.csv`

## 指标说明

- **mean_em**：预测与标准答案逐字完全一致的比例；长答案下通常很低。
- **mean_char_f1**：字符级 F1，主看检索+生成质量。
- **evidence_***：仅统计 CSV 里 `evidence` 非空的样本。

## 按检索模式

| mode | n | mean_em | mean_char_f1 | ev_strict | ev_any | n_ev | P50(s) | P95(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 10 | 0.0 | 0.573526 | — | — | 0 | 21.4564 | 28.8455 |
