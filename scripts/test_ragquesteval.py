"""
RAGQuestEval 指标测试脚本

基于 CRUD-RAG 论文的 RAGQuestEval 指标，通过问答方式评估生成质量。

使用方法:
    python scripts/test_ragquesteval.py --result-file datasets/eval_result.csv
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
import argparse
from typing import List, Dict, Tuple
import jieba
import numpy as np
from collections import Counter
import pandas as pd

from src.generator import get_llm


def compute_f1(a_gold: str, a_pred: str) -> float:
    """计算单个问答对的 F1 分数"""
    gold_toks = list(jieba.cut(a_gold))
    pred_toks = list(jieba.cut(a_pred))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def word_based_f1_score(a_gold_list: List[str], a_pred_list: List[str]) -> float:
    """计算所有问答对的平均 F1 分数"""
    f1_list = []
    for a_gold, a_pred in zip(a_gold_list, a_pred_list):
        f1_list.append(compute_f1(a_gold, a_pred))
    return np.mean(f1_list) if f1_list else 0.0


class RAGQuestEval:
    """RAGQuestEval 评估器"""

    def __init__(self, model_name: str = None):
        self.llm = get_llm()
        self.quest_gt_save = {}

        # 提示词模板
        self.gen_prompt_template = """你是一位新闻编辑，现在，你被提供了一篇新闻，请先从新闻中抽取出你认为重要的所有关键信息（通常为一个关键词，包含文章中的所有实体和名词性短语），然后，根据关键信息设计几个问题，考验大家能否正确回答问题。用json的形式返回答案。以下是个例子。

新闻：2014年，全国新增并网光伏发电容量1060万千瓦，约占全球新增容量的四分之一。其中，全国新增光伏电站855万千瓦，分布式205万千瓦。据统计，2014年中国光伏发电量达到了250亿千瓦时，同比增⻓超过 200%。

{json_response}

现在你需要为这篇新闻设计问题，尽量涵盖大多数关键信息，请尽量让答案可以用两三个词回答，答案不能太长，key_info包含文章中的所有实体和名词性短语，question与key_info一一对应，数量一致，输出用json的格式：

{news}"""

        self.json_response_example = '''{"key_info": ["新增并网光伏发电容量1060万千瓦", "四分之一", "全国新增光伏电站855万千瓦", "分布式光伏容量205万千瓦", "2014年中国光伏发电量250亿千瓦。", "同比增长超过200%"], "question": ["2014年中国新增并网光伏发电容量是多少？", "2014年中国新增并网光伏发电容量约占全球新增容量的几分之几？","全国新增光伏电站的容量是多少？", "分布式光伏容量是多少？", "2014年中国光伏发电量是多少？", "2014年中国光伏发电量相比前一年增长了多少？"]}'''

        self.answer_prompt_template = """你是一位新闻编辑，现在，你被提供了1篇新闻的摘要，和几个问题，请分别根据新闻的摘要回答这些问题。用list的格式输出答案。以下是个例子：

新闻摘要:

2023年7月27日，42位来自9个国家的华裔青年在云南腾冲的普洱茶研学基地参观学习，体验了采茶、压茶、包茶饼等传统制茶工艺，感受到浓厚的普洱茶历史文化。活动由云南省人民政府侨务办公室主办。腾冲市以其悠久的种茶历史和丰富的茶文化，吸引了众多华裔青年的参与。

问题：

["参观学习的华裔青年共有多少人？","这些华裔青年来自几个国家？","华裔青年们的领队是谁？","这次活动是由谁主办的？","腾冲市吸引海外华裔青年的特点是什么？"],

回答:

参观学习的华裔青年共有多少人？
<response>
42人
</response>

这些华裔青年来自几个国家？
<response>
9个
</response>

华裔青年们的领队是谁？
<response>
无法推断
</response>

这次活动是由谁主办的？
<response>
云南省人民政府侨务办公室
</response>

腾冲市吸引海外华裔青年的特点是什么？
<response>
悠久的种茶历史和丰富的茶文化
</response>


现在新闻的摘要是：

{context}

问题：

{questions}

请给出根据新闻的摘要的回答（回答的文本写在<response></response>之间。请注意，用一两个词或者非常简短的语句回答问题，不要添加多余的词。遇见无法回答的问题，请说："无法推断"）
"""

    def question_generation(self, text4gen: str) -> Dict:
        """从文本中生成问题"""
        query = self.gen_prompt_template.format(
            json_response=self.json_response_example,
            news=text4gen
        )

        try:
            resp = self.llm.invoke(query)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            question4eval = json.loads(text)
            return question4eval
        except Exception as e:
            print(f"问题生成失败: {e}")
            return {"key_info": [], "question": []}

    def question_answer(self, context: str, questions: List[str]) -> List[str]:
        """根据上下文回答问题"""
        query = self.answer_prompt_template.format(
            context=context,
            questions=json.dumps(questions, ensure_ascii=False)
        )

        try:
            resp = self.llm.invoke(query)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            pattern = r'<response>\s*(.*?)\s*</response>'
            real_answers = re.findall(pattern, text, re.DOTALL)
            return real_answers
        except Exception as e:
            print(f"回答生成失败: {e}")
            return ["无法推断"] * len(questions)

    def get_qa_pair(self, data_point: Dict) -> Tuple[List[str], List[str], List[str]]:
        """获取问答对"""
        ground_truth_text = data_point["ground_truth_text"]
        generated_text = data_point["generated_text"]

        # 检查缓存
        if data_point["ID"] in self.quest_gt_save.keys():
            questions_gt = self.quest_gt_save[data_point["ID"]]["question"]
            answers_gt4gt = self.quest_gt_save[data_point["ID"]]["answers"]
        else:
            # 生成问题
            keyinfo_and_questions = self.question_generation(ground_truth_text)
            questions_gt = keyinfo_and_questions["question"]

            # 用 ground truth 回答问题
            answers_gt4gt = self.question_answer(ground_truth_text, questions_gt)

            # 缓存
            keyinfo_and_questions["answers"] = answers_gt4gt
            self.quest_gt_save[data_point["ID"]] = keyinfo_and_questions

        # 用生成文本回答问题
        answers_gm4gt = self.question_answer(generated_text, questions_gt)

        return questions_gt, answers_gt4gt, answers_gm4gt

    def evaluate(self, data_point: Dict) -> Tuple[float, float, Dict]:
        """评估单个数据点"""
        try:
            questions_gt, answers_gt4gt, answers_gm4gt = self.get_qa_pair(data_point)

            quest_eval_save = {
                "questions_gt": questions_gt,
                "answers_gt4gt": answers_gt4gt,
                "answers_gm4gt": answers_gm4gt
            }

            # 去除 ground truth 无法推断的问题
            indices = [i for i, x in enumerate(answers_gt4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]

            if len(answers_gm4gt) == 0:
                return 0, 0, quest_eval_save

            # 计算 Quest Recall
            undetermined_ratio = answers_gm4gt.count("无法推断") / len(answers_gm4gt)
            quest_recall = 1 - undetermined_ratio

            # 去除无法推断的问题，计算 F1
            indices = [i for i, x in enumerate(answers_gm4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]

            if answers_gm4gt == []:
                return 0, 0, quest_eval_save

            # 计算 Quest Avg F1
            quest_avg_f1 = word_based_f1_score(answers_gt4gt, answers_gm4gt)

            return quest_avg_f1, quest_recall, quest_eval_save

        except Exception as e:
            print(f"评估失败: {e}")
            quest_eval_save = {
                "questions_gt": [],
                "answers_gt4gt": [],
                "answers_gm4gt": []
            }
            return 0, 0, quest_eval_save

    def evaluate_batch(self, data_points: List[Dict]) -> Dict:
        """批量评估"""
        f1_scores = []
        recall_scores = []
        details = []

        for i, data_point in enumerate(data_points):
            print(f"评估进度: {i+1}/{len(data_points)}")
            f1, recall, detail = self.evaluate(data_point)
            f1_scores.append(f1)
            recall_scores.append(recall)
            details.append({
                "id": data_point.get("ID", i),
                "quest_avg_f1": f1,
                "quest_recall": recall,
                "detail": detail
            })

        return {
            "quest_avg_f1_mean": np.mean(f1_scores),
            "quest_recall_mean": np.mean(recall_scores),
            "quest_avg_f1_std": np.std(f1_scores),
            "quest_recall_std": np.std(recall_scores),
            "details": details
        }


def main():
    parser = argparse.ArgumentParser(description='RAGQuestEval 指标测试')
    parser.add_argument('--result-file', type=str, required=True,
                        help='包含生成结果的 CSV 文件路径')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='输出目录')
    parser.add_argument('--save-quest-gt', action='store_true',
                        help='保存生成的问题和答案')

    args = parser.parse_args()

    # 读取结果文件
    result_file = Path(args.result_file)
    if not result_file.exists():
        print(f"错误: 文件不存在 {args.result_file}")
        return

    df = pd.read_csv(result_file)

    # 检查必要的列
    required_cols = ['ground_truth_text', 'generated_text']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: 缺少必要的列 {col}")
            return

    # 准备数据
    data_points = []
    for idx, row in df.iterrows():
        data_points.append({
            "ID": str(row.get('ID', idx)),
            "ground_truth_text": row['ground_truth_text'],
            "generated_text": row['generated_text']
        })

    print(f"开始评估 {len(data_points)} 条数据...")

    # 创建评估器
    evaluator = RAGQuestEval()

    # 评估
    results = evaluator.evaluate_batch(data_points)

    # 输出结果
    print("\n" + "="*50)
    print("RAGQuestEval 评估结果")
    print("="*50)
    print(f"Quest Avg F1: {results['quest_avg_f1_mean']:.4f} ± {results['quest_avg_f1_std']:.4f}")
    print(f"Quest Recall: {results['quest_recall_mean']:.4f} ± {results['quest_recall_std']:.4f}")
    print("="*50)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = int(np.datetime64('now').astype('int64') / 1e6)
    output_file = output_dir / f'ragquesteval_results_{timestamp}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 保存问题答案（如果需要）
    if args.save_quest_gt:
        quest_gt_file = output_dir / f'quest_gt_save_{timestamp}.json'
        with open(quest_gt_file, 'w', encoding='utf-8') as f:
            json.dump(evaluator.quest_gt_save, f, ensure_ascii=False, indent=2)
        print(f"问题和答案已保存到: {quest_gt_file}")


if __name__ == "__main__":
    main()