"""
一键运行完整的消融实验流程

自动化运行测试、生成表格和报告。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"{'='*60}\n")
    print(f"执行: {' '.join(cmd)}\n")

    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} 完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='一键运行完整测试流程')
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
    parser.add_argument(
        '--skip-tables',
        action='store_true',
        help='跳过表格生成'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("RAGQuestEval 消融实验 - 完整流程")
    print("="*80)

    # 检查测试数据
    test_data_file = Path(args.test_data)
    if not test_data_file.exists():
        print(f"\n✗ 错误: 测试数据文件不存在: {test_data_file}")
        sys.exit(1)

    print(f"\n配置:")
    print(f"  测试数据: {test_data_file}")
    print(f"  输出目录: {args.output_dir}")

    # 步骤 1: 运行消融实验
    success = run_command(
        [
            sys.executable,
            "scripts/run_simple_ablation.py",
            "--test-data", str(test_data_file),
            "--output-dir", args.output_dir
        ],
        "运行消融实验"
    )

    if not success:
        print("\n✗ 消融实验失败，终止流程")
        sys.exit(1)

    # 步骤 2: 生成论文表格
    if not args.skip_tables:
        results_file = Path(args.output_dir) / "ablation_results.json"

        if not results_file.exists():
            print(f"\n✗ 错误: 结果文件不存在: {results_file}")
            sys.exit(1)

        success = run_command(
            [
                sys.executable,
                "scripts/generate_paper_tables.py",
                "--results", str(results_file),
                "--output-dir", "datasets/paper_tables",
                "--format", "all"
            ],
            "生成论文表格"
        )

        if not success:
            print("\n⚠ 表格生成失败，但测试已完成")

    # 完成
    print("\n" + "="*80)
    print("✓ 完整流程执行完成")
    print("="*80)

    print("\n生成的文件:")
    print(f"  - {args.output_dir}/ablation_results.json")
    print(f"  - {args.output_dir}/comparison_table.csv")
    print(f"  - {args.output_dir}/ablation_report.md")

    if not args.skip_tables:
        print(f"\n  - datasets/paper_tables/table1_retrieval_comparison.md")
        print(f"  - datasets/paper_tables/table2_ablation_study.md")
        print(f"  - datasets/paper_tables/table3_baseline_comparison.md")
        print(f"  - datasets/paper_tables/table1_retrieval_comparison.tex")
        print(f"  - datasets/paper_tables/table2_ablation_study.tex")
        print(f"  - datasets/paper_tables/table3_baseline_comparison.tex")

    print("\n查看报告:")
    print(f"  cat {args.output_dir}/ablation_report.md")

    print("\n下一步:")
    print("  1. 查看对比表格，分析结果")
    print("  2. 将论文表格复制到论文文档中")
    print("  3. 根据结果优化系统参数")


if __name__ == "__main__":
    main()