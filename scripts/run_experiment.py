#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键运行检索策略对比实验 + RAGQuestEval 评估
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_retrieval_comparison(
    data_path: str,
    max_samples: int,
    output_dir: str
):
    """运行检索策略对比实验"""
    print(f"\n{'='*60}")
    print(f"步骤 1: 运行检索策略对比实验")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "scripts/run_retrieval_comparison.py",
        "--data-path", data_path,
        "--max-samples", str(max_samples),
        "--output-dir", output_dir
    ]

    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_ragquesteval_evaluation(
    results_file: str,
    output_dir: str
):
    """运行RAGQuestEval评估"""
    print(f"\n{'='*60}")
    print(f"步骤 2: 运行 RAGQuestEval 评估")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "scripts/evaluate_ragquesteval.py",
        "--results-file", results_file,
        "--output-dir", output_dir
    ]

    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='一键运行检索策略对比实验 + RAGQuestEval 评估'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='CRUD_RAG/data/crud_split/split_merged.json',
        help='CRUD-RAG数据集路径'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='最大样本数'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/experiment_results',
        help='输出目录'
    )
    parser.add_argument(
        '--skip-retrieval',
        action='store_true',
        help='跳过检索策略对比（只运行评估）'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"检索策略对比实验 + RAGQuestEval 评估")
    print(f"{'='*60}")
    print(f"数据路径: {args.data_path}")
    print(f"样本数: {args.max_samples}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 步骤1: 运行检索策略对比
    if not args.skip_retrieval:
        retrieval_output = os.path.join(args.output_dir, "retrieval_comparison")
        success = run_retrieval_comparison(
            args.data_path,
            args.max_samples,
            retrieval_output
        )

        if not success:
            print("\n✗ 检索策略对比失败")
            sys.exit(1)

        # 找到结果文件
        results_file = os.path.join(
            retrieval_output,
            f"comparison_results_{args.max_samples}.json"
        )

        if not os.path.exists(results_file):
            print(f"\n✗ 结果文件不存在: {results_file}")
            sys.exit(1)
    else:
        # 如果跳过检索，需要指定结果文件
        parser.error("--skip-retrieval 需要指定结果文件")

    # 步骤2: 运行RAGQuestEval评估
    eval_output = os.path.join(args.output_dir, "ragquesteval_evaluation")
    success = run_ragquesteval_evaluation(
        results_file,
        eval_output
    )

    if not success:
        print("\n✗ RAGQuestEval 评估失败")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[OK] 所有实验完成！")
    print(f"{'='*60}")
    print(f"\n结果保存在:")
    print(f"  检索结果: {results_file}")
    print(f"  评估结果: {eval_output}")
    print(f"\n查看评估结果:")
    print(f"  cat {eval_output}/*.json")


if __name__ == '__main__':
    main()