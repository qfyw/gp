#!/usr/bin/env python3
"""
快速对照测试脚本

用法：
    python run_comparison_quick.py --mode test    # 快速测试（10个样本）
    python run_comparison_quick.py --mode full    # 完整测试（100个样本）
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_test(mode: str):
    """运行对照测试"""

    # 基础命令
    cmd = [
        ".venv/Scripts/python.exe",
        "scripts/run_eval.py",
        "--dataset", "CRUD_RAG/data/crud_split/split_merged.json",
        "--pred-style", "short"
    ]

    # 根据模式设置参数
    if mode == "test":
        cmd.extend([
            "--output", "datasets/comparison_test.csv",
            "--modes", "vector", "vector_keyword", "full",
            "--max-rows", "10"
        ])
        print("🚀 开始快速测试（10个样本）...")
        print("预计时间：5-10分钟\n")

    elif mode == "full":
        cmd.extend([
            "--output", "datasets/comparison_full.csv",
            "--modes", "vector", "vector_keyword", "full",
            "--max-rows", "100"
        ])
        print("🚀 开始完整测试（100个样本）...")
        print("预计时间：1.5-2小时\n")

    else:
        print(f"❌ 未知模式: {mode}")
        print("可用模式: test, full")
        return 1

    # 检查虚拟环境
    venv_python = Path(".venv/Scripts/python.exe")
    if not venv_python.exists():
        print("❌ 虚拟环境不存在！")
        print("请先创建虚拟环境：")
        print("  python -m venv .venv")
        print("  .venv\\Scripts\\Activate.ps1")
        return 1

    # 检查数据文件
    data_file = Path("CRUD_RAG/data/crud_split/split_merged.json")
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return 1

    # 运行测试
    print("执行命令：")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 测试完成！")
        print(f"结果文件: datasets/comparison_{mode}.csv")
        print(f"汇总报告: datasets/comparison_{mode}_summary.md")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败，返回码: {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        return 130

def main():
    parser = argparse.ArgumentParser(description="快速对照测试脚本")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "full"],
        help="测试模式：test=快速测试（10样本），full=完整测试（100样本）"
    )

    args = parser.parse_args()

    # 检查虚拟环境
    if "VIRTUAL_ENV" not in dict(sys.environ.items()):
        print("⚠️  警告：虚拟环境未激活！")
        print("建议先激活虚拟环境：")
        print("  .venv\\Scripts\\Activate.ps1")
        print()
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return 1

    return run_test(args.mode)

if __name__ == "__main__":
    sys.exit(main())