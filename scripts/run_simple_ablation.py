"""
简化的消融实验测试脚本

直接运行 RAGQuestEval 评估，支持不同配置场景的对比。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import csv

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 RAGQuestEval
from scripts.test_ragquesteval import RAGQuestEval


class SimpleAblationTest:
    """简化版消融实验"""

    def __init__(self, output_dir: str = "datasets/ablation_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_test(
        self,
        name: str,
        description: str,
        test_data_file: str,
        config: Dict[str, Any] = None
    ) -> Dict:
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"运行测试: {name}")
        print(f"描述: {description}")
        print(f"{'='*60}\n")

        result = {
            'name': name,
            'description': description,
            'config': config or {},
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'metrics': {}
        }

        try:
            start_time = time.time()

            # 创建评估器
            evaluator = RAGQuestEval()

            # 读取测试数据
            import csv as csv_module
            data_points = []

            with open(test_data_file, 'r', encoding='utf-8-sig') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    data_points.append({
                        "ID": row.get('ID', ''),
                        "ground_truth_text": row.get('ground_truth_text', ''),
                        "generated_text": row.get('generated_text', '')
                    })

            print(f"加载了 {len(data_points)} 条测试数据")

            # 运行评估
            print("开始评估...")
            eval_results = evaluator.evaluate_batch(data_points)

            elapsed_time = time.time() - start_time

            print(f"\n✓ 测试完成 ({elapsed_time:.1f}s)")
            print(f"Quest Avg F1:  {eval_results['quest_avg_f1_mean']:.4f} ± {eval_results['quest_avg_f1_std']:.4f}")
            print(f"Quest Recall:  {eval_results['quest_recall_mean']:.4f} ± {eval_results['quest_recall_std']:.4f}")

            result['metrics'] = {
                'quest_avg_f1': eval_results['quest_avg_f1_mean'],
                'quest_recall': eval_results['quest_recall_mean'],
                'quest_avg_f1_std': eval_results['quest_avg_f1_std'],
                'quest_recall_std': eval_results['quest_recall_std'],
                'num_samples': len(data_points)
            }
            result['success'] = True

            # 保存详细结果
            timestamp = int(time.time() * 1000)
            detail_file = self.output_dir / f"{name}_results_{timestamp}.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)

            print(f"详细结果已保存: {detail_file}")

        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            result['error'] = str(e)

        result['end_time'] = datetime.now().isoformat()
        return result

    def run_comparison_tests(self, test_data_file: str):
        """运行对比测试集"""
        print(f"\n{'='*80}")
        print("开始运行对比测试集")
        print(f"{'='*80}\n")

        # 定义测试场景
        scenarios = [
            {
                'name': 'test_1',
                'description': '基础测试（原始数据）',
                'config': {}
            },
            # 可以添加更多场景
            # {
            #     'name': 'test_2',
            #     'description': '优化版本1',
            #     'config': {'param1': 'value1'}
            # }
        ]

        print(f"共 {len(scenarios)} 个测试场景\n")

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] 运行场景: {scenario['name']}")

            result = self.run_test(
                name=scenario['name'],
                description=scenario['description'],
                test_data_file=test_data_file,
                config=scenario.get('config')
            )

            self.results.append(result)

        # 生成报告
        self._generate_report()

    def _generate_report(self):
        """生成测试报告"""
        print(f"\n{'='*80}")
        print("测试报告")
        print(f"{'='*80}\n")

        # 统计
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful

        print(f"总测试数: {total}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print()

        # 对比表格
        self._print_comparison_table()

        # 保存报告
        self._save_report()

    def _print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            return

        print("\n" + "="*80)
        print("结果对比表格")
        print("="*80)

        # 表头
        print(f"{'场景名称':<25} {'描述':<25} {'Quest Avg F1':>12} {'Quest Recall':>12} {'状态':>6}")
        print("-"*80)

        # 数据行
        for result in self.results:
            name = result['name'][:25]
            description = result['description'][:25]
            metrics = result.get('metrics', {})

            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')
            status = '✓' if result['success'] else '✗'

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            print(f"{name:<25} {description:<25} {str(f1):>12} {str(recall):>12} {status:>6}")

        print("="*80)

    def _save_report(self):
        """保存报告"""
        if not self.results:
            return

        # 保存 JSON
        json_file = self.output_dir / "ablation_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'successful': sum(1 for r in self.results if r['success']),
                'results': self.results
            }, f, ensure_ascii=False, indent=2)

        print(f"\nJSON 结果已保存: {json_file}")

        # 保存 CSV
        csv_file = self.output_dir / "comparison_table.csv"
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['场景名称', '描述', 'Quest Avg F1', 'Quest Recall', '样本数', '成功'])

            for result in self.results:
                metrics = result.get('metrics', {})
                f1 = metrics.get('quest_avg_f1', '')
                recall = metrics.get('quest_recall', '')
                num_samples = metrics.get('num_samples', '')

                writer.writerow([
                    result['name'],
                    result['description'],
                    f1 if f1 else 'N/A',
                    recall if recall else 'N/A',
                    num_samples,
                    '✓' if result['success'] else '✗'
                ])

        print(f"对比表格已保存: {csv_file}")

        # 生成 Markdown 报告
        self._generate_markdown_report()

    def _generate_markdown_report(self):
        """生成 Markdown 报告"""
        md_file = self.output_dir / "ablation_report.md"

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 统计
            total = len(self.results)
            successful = sum(1 for r in self.results if r['success'])

            f.write("## 测试统计\n\n")
            f.write(f"- **总测试数**: {total}\n")
            f.write(f"- **成功**: {successful}\n")
            f.write(f"- **失败**: {total - successful}\n\n")

            # 对比表格
            f.write("## 结果对比\n\n")
            f.write("| 场景名称 | 描述 | Quest Avg F1 | Quest Recall | 样本数 |\n")
            f.write("|---------|------|-------------|-------------|-------|\n")

            for result in self.results:
                name = result['name']
                description = result['description']
                metrics = result.get('metrics', {})

                f1 = metrics.get('quest_avg_f1', 'N/A')
                recall = metrics.get('quest_recall', 'N/A')
                num_samples = metrics.get('num_samples', '-')

                if isinstance(f1, float):
                    f1 = f"{f1:.4f}"
                if isinstance(recall, float):
                    recall = f"{recall:.4f}"

                success_mark = '' if result['success'] else ' ❌'
                f.write(f"| {name}{success_mark} | {description} | {f1} | {recall} | {num_samples} |\n")

            f.write("\n## 详细结果\n\n")

            for result in self.results:
                f.write(f"### {result['name']}: {result['description']}\n\n")

                if not result['success']:
                    f.write(f"**状态**: ❌ 失败\n\n")
                    f.write(f"**错误**: {result.get('error', 'Unknown')}\n\n")
                    continue

                f.write(f"**状态**: ✅ 成功\n\n")

                metrics = result.get('metrics', {})
                if metrics:
                    f.write(f"**指标**:\n\n")
                    f.write(f"- Quest Avg F1: {metrics.get('quest_avg_f1', 'N/A')}\n")
                    f.write(f"- Quest Recall: {metrics.get('quest_recall', 'N/A')}\n")
                    f.write(f"- Quest Avg F1 标准差: {metrics.get('quest_avg_f1_std', 'N/A')}\n")
                    f.write(f"- Quest Recall 标准差: {metrics.get('quest_recall_std', 'N/A')}\n")
                    f.write(f"- 样本数: {metrics.get('num_samples', 'N/A')}\n\n")

        print(f"Markdown 报告已保存: {md_file}")


def main():
    parser = argparse.ArgumentParser(description='运行简化版消融实验')
    parser.add_argument(
        '--test-data',
        type=str,
        default='datasets/ragquesteval_test_n20.csv',
        help='测试数据文件'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/ablation_tests',
        help='输出目录'
    )

    args = parser.parse_args()

    # 检查测试数据文件
    test_data_file = Path(args.test_data)
    if not test_data_file.exists():
        print(f"错误: 测试数据文件不存在: {test_data_file}")
        sys.exit(1)

    # 运行对比测试
    tester = SimpleAblationTest(args.output_dir)
    tester.run_comparison_tests(str(test_data_file))


if __name__ == "__main__":
    main()