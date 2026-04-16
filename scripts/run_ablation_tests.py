"""
自动化消融实验和对照测试脚本

支持多种测试场景的自动化运行和结果对比。
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import csv

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AblationTestRunner:
    """消融实验运行器"""

    def __init__(self, config_file: str, output_dir: str = "datasets/ablation_tests"):
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.config = self._load_config()
        self.results = []

    def _load_config(self) -> Dict:
        """加载测试配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_env_file(self, scenario_config: Dict) -> str:
        """创建临时环境配置文件"""
        env_file = self.output_dir / f"_{scenario_config['name']}.env"

        # 读取基础 .env
        base_env = PROJECT_ROOT / ".env"
        env_vars = {}

        if base_env.exists():
            with open(base_env, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value

        # 更新配置
        for key, value in scenario_config.get('config', {}).items():
            env_vars[key] = str(value)

        # 写入临时文件
        with open(env_file, 'w', encoding='utf-8') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        return str(env_file)

    def _run_scenario(self, scenario: Dict) -> Dict:
        """运行单个测试场景"""
        name = scenario['name']
        description = scenario.get('description', name)

        print(f"\n{'='*60}")
        print(f"运行测试: {name} - {description}")
        print(f"{'='*60}")

        results = {
            'name': name,
            'description': description,
            'config': scenario.get('config', {}),
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'metrics': {}
        }

        try:
            # 创建临时配置
            env_file = self._create_env_file(scenario)

            # 运行评估（这里需要根据实际的评估命令调整）
            # 假设使用 scripts/run_eval.py
            eval_command = [
                sys.executable,
                "scripts/run_eval.py",
                "--num-samples", "20",
                "--output-dir", str(self.output_dir),
                "--scenario", name
            ]

            print(f"执行命令: {' '.join(eval_command)}")

            # 设置环境变量
            env = os.environ.copy()
            env['SCENARIO_CONFIG'] = env_file

            # 运行命令
            start_time = time.time()
            process = subprocess.run(
                eval_command,
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            elapsed_time = time.time() - start_time

            if process.returncode == 0:
                print(f"✓ 测试完成 ({elapsed_time:.1f}s)")

                # 尝试读取结果
                result_file = self.output_dir / f"eval_{name}_results.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results['metrics'] = json.load(f)
                    results['success'] = True
                else:
                    print("⚠ 未找到结果文件")
                    results['error'] = "未找到结果文件"
            else:
                print(f"✗ 测试失败")
                print(f"错误输出: {process.stderr}")
                results['error'] = process.stderr

            # 清理临时文件
            if os.path.exists(env_file):
                os.remove(env_file)

        except subprocess.TimeoutExpired:
            print("✗ 测试超时")
            results['error'] = "测试超时"
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results['error'] = str(e)

        results['end_time'] = datetime.now().isoformat()
        return results

    def run_all(self, scenarios: List[str] = None):
        """运行所有测试场景"""
        all_scenarios = self.config.get('test_scenarios', [])

        if scenarios:
            all_scenarios = [s for s in all_scenarios if s['name'] in scenarios]

        print(f"开始运行 {len(all_scenarios)} 个测试场景...")
        print(f"输出目录: {self.output_dir}\n")

        for i, scenario in enumerate(all_scenarios, 1):
            print(f"\n[{i}/{len(all_scenarios)}] {scenario['description']}")

            result = self._run_scenario(scenario)
            self.results.append(result)

            # 保存中间结果
            self._save_results()

        # 生成最终报告
        self._generate_report()

    def _save_results(self):
        """保存测试结果"""
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'total_scenarios': len(self.results),
                'successful': sum(1 for r in self.results if r['success']),
                'results': self.results
            }, f, ensure_ascii=False, indent=2)

    def _generate_report(self):
        """生成测试报告"""
        print(f"\n{'='*60}")
        print("测试报告")
        print(f"{'='*60}\n")

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

        # 保存表格
        self._save_comparison_table()

        # 生成 Markdown 报告
        self._generate_markdown_report()

    def _print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            return

        print("\n" + "="*80)
        print("结果对比表格")
        print("="*80)

        # 表头
        print(f"{'场景名称':<25} {'描述':<20} {'Quest Avg F1':>12} {'Quest Recall':>12}")
        print("-"*80)

        # 数据行
        for result in self.results:
            name = result['name'][:25]
            description = result['description'][:20]
            metrics = result.get('metrics', {})

            f1 = metrics.get('quest_avg_f1', 'N/A')
            recall = metrics.get('quest_recall', 'N/A')

            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"

            print(f"{name:<25} {description:<20} {str(f1):>12} {str(recall):>12}")

        print("="*80)

    def _save_comparison_table(self):
        """保存对比表格到 CSV"""
        if not self.results:
            return

        csv_file = self.output_dir / "comparison_table.csv"

        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow(['场景名称', '描述', 'Quest Avg F1', 'Quest Recall', '成功'])

            # 数据行
            for result in self.results:
                metrics = result.get('metrics', {})
                f1 = metrics.get('quest_avg_f1', '')
                recall = metrics.get('quest_recall', '')
                success = result['success']

                writer.writerow([
                    result['name'],
                    result['description'],
                    f1 if f1 else 'N/A',
                    recall if recall else 'N/A',
                    '✓' if success else '✗'
                ])

        print(f"\n对比表格已保存: {csv_file}")

    def _generate_markdown_report(self):
        """生成 Markdown 报告"""
        if not self.results:
            return

        md_file = self.output_dir / "ablation_report.md"

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 统计
            total = len(self.results)
            successful = sum(1 for r in self.results if r['success'])
            failed = total - successful

            f.write("## 测试统计\n\n")
            f.write(f"- **总测试数**: {total}\n")
            f.write(f"- **成功**: {successful}\n")
            f.write(f"- **失败**: {failed}\n\n")

            # 对比表格
            f.write("## 结果对比\n\n")
            f.write("| 场景名称 | 描述 | Quest Avg F1 | Quest Recall |\n")
            f.write("|---------|------|-------------|-------------|\n")

            for result in self.results:
                name = result['name']
                description = result['description']
                metrics = result.get('metrics', {})

                f1 = metrics.get('quest_avg_f1', 'N/A')
                recall = metrics.get('quest_recall', 'N/A')

                if isinstance(f1, float):
                    f1 = f"{f1:.4f}"
                if isinstance(recall, float):
                    recall = f"{recall:.4f}"

                success_mark = '' if result['success'] else ' ❌'
                f.write(f"| {name}{success_mark} | {description} | {f1} | {recall} |\n")

            f.write("\n## 详细结果\n\n")

            for result in self.results:
                f.write(f"### {result['name']}: {result['description']}\n\n")

                if not result['success']:
                    f.write(f"**状态**: ❌ 失败\n\n")
                    f.write(f"**错误**: {result.get('error', 'Unknown')}\n\n")
                    continue

                f.write(f"**状态**: ✅ 成功\n\n")
                f.write(f"**配置**:\n```json\n")
                f.write(json.dumps(result.get('config', {}), indent=2, ensure_ascii=False))
                f.write("\n```\n\n")

                metrics = result.get('metrics', {})
                if metrics:
                    f.write(f"**指标**:\n\n")
                    f.write(f"- Quest Avg F1: {metrics.get('quest_avg_f1', 'N/A')}\n")
                    f.write(f"- Quest Recall: {metrics.get('quest_recall', 'N/A')}\n")
                    f.write(f"- Quest Avg F1 标准差: {metrics.get('quest_avg_f1_std', 'N/A')}\n")
                    f.write(f"- Quest Recall 标准差: {metrics.get('quest_recall_std', 'N/A')}\n\n")

        print(f"Markdown 报告已保存: {md_file}")


def main():
    parser = argparse.ArgumentParser(description='运行消融实验和对照测试')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/test_scenarios.json',
        help='测试场景配置文件'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/ablation_tests',
        help='输出目录'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        nargs='+',
        help='指定要运行的场景名称（默认运行所有）'
    )

    args = parser.parse_args()

    # 运行测试
    runner = AblationTestRunner(args.config, args.output_dir)
    runner.run_all(args.scenarios)


if __name__ == "__main__":
    main()