"""
论文表格生成器

将测试结果格式化为论文中使用的表格格式。
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any


class PaperTableGenerator:
    """论文表格生成器"""

    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self._load_results()

    def _load_results(self) -> List[Dict]:
        """加载测试结果"""
        if self.results_file.suffix == '.json':
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', [])
        elif self.results_file.suffix == '.csv':
            with open(self.results_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            raise ValueError(f"不支持的文件格式: {self.results_file.suffix}")

    def generate_retrieval_comparison_table(self) -> str:
        """生成检索策略对比表格（Table 1）"""
        table = """
Table 1: Comparison of Retrieval Strategies

| Method | Vector | Keyword | BM25 | Knowledge Graph | Quest Avg F1 | Quest Recall |
|--------|--------|---------|------|----------------|-------------|-------------|
"""
        for result in self.results:
            metrics = result.get('metrics', {})

            # 从配置中提取信息
            config = result.get('config', {})
            vector = '✓' if config.get('RETRIEVAL_VECTOR_TOP_K', 0) > 0 else '✗'
            keyword = '✓' if config.get('RETRIEVAL_KEYWORD_TOP_K', 0) > 0 else '✗'
            bm25 = '✓' if config.get('BM25_ENABLED', '0') == '1' else '✗'
            kg = '✓' if config.get('RETRIEVAL_GRAPH_MAX', 0) > 0 else '✗'

            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            name = result['name'].replace('_', ' ').title()
            table += f"| {name} | {vector} | {keyword} | {bm25} | {kg} | {f1} | {recall} |\n"

        return table

    def generate_ablation_table(self) -> str:
        """生成消融实验表格（Table 2）"""
        # 找出完整模型的结果
        full_model = None
        for result in self.results:
            if 'full' in result['name'].lower() or result['name'] == 'full_hybrid':
                full_model = result
                break

        if not full_model:
            full_model = self.results[0] if self.results else {}

        full_f1 = full_model.get('metrics', {}).get('quest_avg_f1', 0)
        full_recall = full_model.get('metrics', {}).get('quest_recall', 0)

        table = """
Table 2: Ablation Study

| Method | Description | Quest Avg F1 | Δ | Quest Recall | Δ |
|--------|-------------|-------------|---|-------------|---|
"""

        for result in self.results:
            metrics = result.get('metrics', {})
            f1 = metrics.get('quest_avg_f1', 0)
            recall = metrics.get('quest_recall', 0)

            if isinstance(f1, str):
                f1 = 0
            if isinstance(recall, str):
                recall = 0

            # 计算差异
            f1_delta = f1 - full_f1 if full_model else 0
            recall_delta = recall - full_recall if full_model else 0

            f1_delta_str = f"{f1_delta:+.4f}" if f1_delta != 0 else "-"
            recall_delta_str = f"{recall_delta:+.4f}" if recall_delta != 0 else "-"

            # 标记完整模型
            name = result['name'].replace('_', ' ').title()
            if result == full_model:
                name = f"**{name}**"

            description = result.get('description', name)

            table += f"| {name} | {description} | {f1:.4f} | {f1_delta_str} | {recall:.4f} | {recall_delta_str} |\n"

        return table

    def generate_baseline_comparison_table(self) -> str:
        """生成基线对比表格（Table 3）"""
        table = """
Table 3: Comparison with Baseline Methods

| Method | Quest Avg F1 | Quest Recall |
|--------|-------------|-------------|
"""

        for result in self.results:
            metrics = result.get('metrics', {})
            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            name = result['name'].replace('_', ' ').title()
            # 标记我们的方法
            if 'full' in result['name'].lower():
                name = f"**{name}**"

            table += f"| {name} | {f1} | {recall} |\n"

        return table

    def generate_latex_table(self, table_type: str = 'retrieval') -> str:
        """生成 LaTeX 表格"""
        if table_type == 'retrieval':
            return self._generate_latex_retrieval_table()
        elif table_type == 'ablation':
            return self._generate_latex_ablation_table()
        elif table_type == 'baseline':
            return self._generate_latex_baseline_table()
        else:
            raise ValueError(f"未知的表格类型: {table_type}")

    def _generate_latex_retrieval_table(self) -> str:
        """生成 LaTeX 检索策略对比表格"""
        table = """
\\begin{table}[h]
\\centering
\\caption{Comparison of Retrieval Strategies}
\\label{tab:retrieval_comparison}
\\begin{tabular}{lccccc}
\\toprule
Method & Vector & Keyword & BM25 & Knowledge Graph & Quest Avg F1 \\\\ 
& & & & & & Quest Recall \\\\
\\midrule
"""

        for result in self.results:
            config = result.get('config', {})
            vector = '$\\checkmark$' if config.get('RETRIEVAL_VECTOR_TOP_K', 0) > 0 else '$-$'
            keyword = '$\\checkmark$' if config.get('RETRIEVAL_KEYWORD_TOP_K', 0) > 0 else '$-$'
            bm25 = '$\\checkmark$' if config.get('BM25_ENABLED', '0') == '1' else '$-$'
            kg = '$\\checkmark$' if config.get('RETRIEVAL_GRAPH_MAX', 0) > 0 else '$-$'

            metrics = result.get('metrics', {})
            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            name = result['name'].replace('_', ' ').title()
            table += f"{name} & {vector} & {keyword} & {bm25} & {kg} & {f1} & {recall} \\\\\n"

        table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return table

    def _generate_latex_ablation_table(self) -> str:
        """生成 LaTeX 消融实验表格"""
        table = """
\\begin{table}[h]
\\centering
\\caption{Ablation Study}
\\label{tab:ablation}
\\begin{tabular}{lccc}
\\toprule
Method & Quest Avg F1 & $\\Delta$ & Quest Recall & $\\Delta$ \\\\
\\midrule
"""

        for result in self.results:
            metrics = result.get('metrics', {})
            f1 = metrics.get('quest_avg_f1', 0)
            recall = metrics.get('quest_recall', 0)

            if isinstance(f1, str):
                f1 = 0
            if isinstance(recall, str):
                recall = 0

            # 找出第一个结果作为基线
            baseline = self.results[0].get('metrics', {})
            baseline_f1 = baseline.get('quest_avg_f1', 0)
            baseline_recall = baseline.get('quest_recall', 0)

            f1_delta = f1 - baseline_f1 if isinstance(baseline_f1, float) else 0
            recall_delta = recall - baseline_recall if isinstance(baseline_recall, float) else 0

            f1_delta_str = f"{f1_delta:+.4f}" if f1_delta != 0 else "-"
            recall_delta_str = f"{recall_delta:+.4f}" if recall_delta != 0 else "-"

            name = result['name'].replace('_', ' ').title()
            table += f"{name} & {f1:.4f} & {f1_delta_str} & {recall:.4f} & {recall_delta_str} \\\\\n"

        table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return table

    def _generate_latex_baseline_table(self) -> str:
        """生成 LaTeX 基线对比表格"""
        table = """
\\begin{table}[h]
\\centering
\\caption{Comparison with Baseline Methods}
\\label{tab:baseline}
\\begin{tabular}{lcc}
\\toprule
Method & Quest Avg F1 & Quest Recall \\\\
\\midrule
"""

        for result in self.results:
            metrics = result.get('metrics', {})
            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            name = result['name'].replace('_', ' ').title()
            if 'full' in result['name'].lower():
                name = f"\\textbf{{{name}}}"

            table += f"{name} & {f1} & {recall} \\\\\n"

        table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return table

    def save_all_tables(self, output_dir: str):
        """保存所有表格"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Markdown 表格
        with open(output_path / "table1_retrieval_comparison.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_retrieval_comparison_table())

        with open(output_path / "table2_ablation_study.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_ablation_table())

        with open(output_path / "table3_baseline_comparison.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_baseline_comparison_table())

        # LaTeX 表格
        with open(output_path / "table1_retrieval_comparison.tex", 'w', encoding='utf-8') as f:
            f.write(self.generate_latex_table('retrieval'))

        with open(output_path / "table2_ablation_study.tex", 'w', encoding='utf-8') as f:
            f.write(self.generate_latex_table('ablation'))

        with open(output_path / "table3_baseline_comparison.tex", 'w', encoding='utf-8') as f:
            f.write(self.generate_latex_table('baseline'))

        print(f"所有表格已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成论文表格')
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='测试结果文件（JSON 或 CSV）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/paper_tables',
        help='输出目录'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['markdown', 'latex', 'all'],
        default='all',
        help='输出格式'
    )

    args = parser.parse_args()

    generator = PaperTableGenerator(args.results)

    if args.format in ['markdown', 'all']:
        print("\n生成 Markdown 表格...")
        print(generator.generate_retrieval_comparison_table())
        print(generator.generate_ablation_table())
        print(generator.generate_baseline_comparison_table())

    if args.format in ['latex', 'all']:
        print("\n生成 LaTeX 表格...")
        print(generator.generate_latex_table('retrieval'))
        print(generator.generate_latex_table('ablation'))
        print(generator.generate_latex_table('baseline'))

    if args.format == 'all':
        generator.save_all_tables(args.output_dir)


if __name__ == "__main__":
    main()