"""
快速测试 RAGQuestEval 指标

使用示例数据快速测试 RAGQuestEval 指标的实现。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_ragquesteval import RAGQuestEval


def quick_test():
    """快速测试"""
    print("=" * 60)
    print("RAGQuestEval 快速测试")
    print("=" * 60)

    # 示例数据
    data_points = [
        {
            "ID": "1",
            "ground_truth_text": "2023年7月27日，42位来自9个国家的华裔青年在云南腾冲的普洱茶研学基地参观学习，体验了采茶、压茶、包茶饼等传统制茶工艺，感受到浓厚的普洱茶历史文化。活动由云南省人民政府侨务办公室主办。腾冲市以其悠久的种茶历史和丰富的茶文化，吸引了众多华裔青年的参与。",
            "generated_text": "42位华裔青年在云南腾冲体验了制茶工艺，活动由云南省侨办主办。"
        },
        {
            "ID": "2",
            "ground_truth_text": "2014年，全国新增并网光伏发电容量1060万千瓦，约占全球新增容量的四分之一。其中，全国新增光伏电站855万千瓦，分布式205万千瓦。据统计，2014年中国光伏发电量达到了250亿千瓦时，同比增长超过200%。",
            "generated_text": "中国2014年新增光伏装机1060万千瓦，占全球四分之一，其中电站855万千瓦，分布式205万千瓦。"
        },
        {
            "ID": "3",
            "ground_truth_text": "国家卫生健康委员会于2023年7月28日启动了名为\"启明行动\"的专项活动，旨在防控儿童青少年的近视问题。该活动所依据的指导性文件为《防控儿童青少年近视核心知识十条》。",
            "generated_text": "\"启明行动\"针对儿童青少年近视防控，依据《防控儿童青少年近视核心知识十条》。"
        }
    ]

    print(f"\n准备测试 {len(data_points)} 条数据...")
    print(f"数据预览:")
    for i, dp in enumerate(data_points[:2], 1):
        print(f"\n  样本 {i}:")
        print(f"    原文: {dp['ground_truth_text'][:50]}...")
        print(f"    生成: {dp['generated_text']}")

    # 创建评估器
    print("\n正在初始化 RAGQuestEval 评估器...")
    evaluator = RAGQuestEval()

    # 评估
    print("\n开始评估...")
    print("注意：这个过程可能需要一些时间，因为需要多次调用 LLM")
    print("-" * 60)

    results = evaluator.evaluate_batch(data_points)

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"\nQuest Avg F1:  {results['quest_avg_f1_mean']:.4f} ± {results['quest_avg_f1_std']:.4f}")
    print(f"Quest Recall:  {results['quest_recall_mean']:.4f} ± {results['quest_recall_std']:.4f}")
    print("=" * 60)

    # 显示详细结果
    print("\n详细结果:")
    for i, detail in enumerate(results['details'], 1):
        print(f"\n样本 {detail['id']}:")
        print(f"  Quest Avg F1:   {detail['quest_avg_f1']:.4f}")
        print(f"  Quest Recall:   {detail['quest_recall']:.4f}")
        print(f"  生成问题数:     {len(detail['detail']['questions_gt'])}")

        # 显示问题和答案（最多3个）
        for j, (q, a_gt, a_gm) in enumerate(zip(
            detail['detail']['questions_gt'][:3],
            detail['detail']['answers_gt4gt'][:3],
            detail['detail']['answers_gm4gt'][:3]
        ), 1):
            print(f"\n  问题 {j}: {q}")
            print(f"    参考答案: {a_gt}")
            print(f"    预测答案: {a_gm}")

        if len(detail['detail']['questions_gt']) > 3:
            print(f"  ... 还有 {len(detail['detail']['questions_gt']) - 3} 个问题")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 保存结果
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)

    import json
    import numpy as np

    timestamp = int(np.datetime64('now').astype('int64') / 1e6)
    output_file = output_dir / f'ragquesteval_quick_test_{timestamp}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保：")
        print("1. 已配置 OPENAI_API_KEY 和 OPENAI_API_BASE")
        print("2. 已安装必要的依赖（jieba, numpy, pandas）")
        print("3. 网络连接正常（需要调用 LLM API）")