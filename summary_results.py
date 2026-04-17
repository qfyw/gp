import json
from pathlib import Path

dirs = sorted([d for d in Path('datasets').iterdir() if d.is_dir()])
print(f"{'Experiment':<30s} {'F1 Score':>10s}  {'Recall':>10s}")
print("-" * 55)

for d in dirs:
    f = d / 'ragquesteval_results_1776.json'
    if f.exists():
        data = json.load(open(f, encoding='utf-8'))
        f1 = data['quest_avg_f1_mean'] * 100
        recall = data['quest_recall_mean'] * 100
        print(f'{d.name:<30s} {f1:>8.2f}%  {recall:>8.2f}%')