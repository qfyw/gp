#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGQuestEval 评估脚本
根据CRUD-RAG论文的方法，使用问答方式评估RAG系统生成质量
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGQuestEval:
    """RAGQuestEval 评估器"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )
        self.model = os.getenv('OPENAI_MODEL', 'qwen-flash')

    def generate_questions(self, text: str, max_retries: int = 3) -> List[Dict]:
        """从文本生成问题

        Args:
            text: 输入文本
            max_retries: 最大重试次数

        Returns:
            问题列表
        """
        prompt = f"""请从以下文本中提取关键信息，并生成3-5个问题来验证这些信息。

文本：
{text}

请以JSON格式返回，包含问题列表，格式如下：
{{
    "questions": [
        {{
            "question": "问题1",
            "answer": "答案1"
        }},
        {{
            "question": "问题2",
            "answer": "答案2"
        }}
    ]
}}

要求：
1. 问题应该覆盖文本中的主要信息
2. 答案应该基于文本，准确无误
3. 问题应该清晰具体
4. 只返回JSON，不要其他内容"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的问题生成器，擅长从文本中提取关键信息并生成问题。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

                content = response.choices[0].message.content.strip()

                # 提取JSON
                json_str = self.extract_json(content)

                if json_str:
                    data = json.loads(json_str)
                    questions = data.get('questions', [])

                    if questions:
                        return questions

            except Exception as e:
                print(f"  问题生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        return []

    def answer_questions(
        self,
        questions: List[Dict],
        context: str,
        max_retries: int = 3
    ) -> List[str]:
        """使用上下文回答问题

        Args:
            questions: 问题列表
            context: 上下文文本
            max_retries: 最大重试次数

        Returns:
            答案列表
        """
        if not questions:
            return []

        questions_text = "\n".join([
            f"{i+1}. {q['question']}" for i, q in enumerate(questions)
        ])

        prompt = f"""请根据以下上下文回答问题。

上下文：
{context}

问题：
{questions_text}

请按顺序回答每个问题，格式如下：
1. <response>答案1</response>
2. <response>答案2</response>
3. <response>答案3</response>

要求：
1. 答案必须基于上下文
2. 如果上下文中没有相关信息，回答"无法推断"
3. 保持答案简洁准确
4. 只返回答案，不要其他内容"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的问题回答者，擅长根据上下文回答问题。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

                content = response.choices[0].message.content.strip()

                # 提取答案
                answers = self.extract_answers(content, len(questions))

                if len(answers) == len(questions):
                    return answers

            except Exception as e:
                print(f"  回答生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        # 如果失败，返回无法推断
        return ["无法推断"] * len(questions)

    def extract_json(self, text: str) -> str:
        """从文本中提取JSON

        Args:
            text: 包含JSON的文本

        Returns:
            JSON字符串
        """
        # 尝试直接解析
        try:
            json.loads(text)
            return text
        except:
            pass

        # 尝试清理Markdown标记
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{(.*?)\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_str = '{' + match + '}' if pattern.endswith('}(') else match
                    json.loads(json_str)
                    return json_str
                except:
                    continue

        # 尝试括号匹配
        stack = []
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start >= 0:
                        json_str = text[start:i+1]
                        try:
                            json.loads(json_str)
                            return json_str
                        except:
                            pass

        return ""

    def extract_answers(self, text: str, num_questions: int) -> List[str]:
        """从文本中提取答案

        Args:
            text: 包含答案的文本
            num_questions: 问题数量

        Returns:
            答案列表
        """
        answers = []

        # 尝试提取 <response> 标签
        pattern = r'<response>(.*?)</response>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if len(matches) == num_questions:
            return [m.strip() for m in matches]

        # 如果没有匹配到，按行分割
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '问题', '答案')):
                answers.append(line)
            if len(answers) >= num_questions:
                break

        # 填充剩余的答案
        while len(answers) < num_questions:
            answers.append("无法推断")

        return answers[:num_questions]

    def calculate_f1(self, reference: str, prediction: str) -> float:
        """计算F1分数

        Args:
            reference: 参考答案
            prediction: 预测答案

        Returns:
            F1分数
        """
        if not reference or not prediction:
            return 0.0

        if reference.strip() == prediction.strip():
            return 1.0

        # 简单的字符级F1
        ref_chars = set(reference.strip())
        pred_chars = set(prediction.strip())

        if not ref_chars and not pred_chars:
            return 1.0

        if not ref_chars or not pred_chars:
            return 0.0

        intersection = ref_chars & pred_chars

        precision = len(intersection) / len(pred_chars)
        recall = len(intersection) / len(ref_chars)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def evaluate_sample(
        self,
        reference_text: str,
        generated_text: str
    ) -> Dict:
        """评估单个样本

        Args:
            reference_text: 参考文本（ground truth）
            generated_text: 生成文本

        Returns:
            评估结果
        """
        # 从参考文本生成问题
        questions = self.generate_questions(reference_text)

        if not questions:
            return {
                'success': False,
                'error': '问题生成失败',
                'f1_scores': [],
                'quest_avg_f1': 0.0,
                'quest_recall': 0.0
            }

        # 使用参考文本回答
        ref_answers = self.answer_questions(questions, reference_text)

        # 使用生成文本回答
        gen_answers = self.answer_questions(questions, generated_text)

        # 计算F1分数
        f1_scores = []
        for ref_ans, gen_ans in zip(ref_answers, gen_answers):
            f1 = self.calculate_f1(ref_ans, gen_ans)
            f1_scores.append(f1)

        # 计算平均值
        quest_avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # 计算Recall（无法推断的比例）
        cannot_infer = sum(1 for ans in gen_answers if '无法推断' in ans)
        quest_recall = 1.0 - (cannot_infer / len(gen_answers)) if gen_answers else 0.0

        return {
            'success': True,
            'num_questions': len(questions),
            'f1_scores': f1_scores,
            'quest_avg_f1': quest_avg_f1,
            'quest_recall': quest_recall,
            'questions': questions,
            'reference_answers': ref_answers,
            'generated_answers': gen_answers
        }

    def evaluate_results(
        self,
        results_file: str,
        output_dir: str = "datasets/ragquesteval_results"
    ) -> Dict:
        """评估测试结果

        Args:
            results_file: 测试结果文件
            output_dir: 输出目录

        Returns:
            评估结果
        """
        print(f"\n{'='*60}")
        print(f"RAGQuestEval 评估")
        print(f"{'='*60}")
        print(f"输入文件: {results_file}")

        # 加载测试结果
        with open(results_file, 'r', encoding='utf-8') as f:
            test_results = json.load(f)

        strategies = test_results.get('strategies', [])

        evaluation_summary = {
            'input_file': results_file,
            'strategies': [],
            'timestamp': datetime.now().isoformat()
        }

        for strategy in strategies:
            strategy_name = strategy['strategy']
            print(f"\n评估策略: {strategy_name}")

            sample_results = strategy.get('results', [])

            strategy_eval = {
                'strategy': strategy_name,
                'description': strategy.get('config', {}),
                'samples': [],
                'quest_avg_f1_scores': [],
                'quest_recall_scores': []
            }

            for i, sample in enumerate(sample_results, 1):
                print(f"  [{i}/{len(sample_results)}] 评估样本...")

                try:
                    eval_result = self.evaluate_sample(
                        sample['reference_answer'],
                        sample['generated_answer']
                    )

                    if eval_result['success']:
                        strategy_eval['samples'].append({
                            'question': sample['question'],
                            'quest_avg_f1': eval_result['quest_avg_f1'],
                            'quest_recall': eval_result['quest_recall'],
                            'num_questions': eval_result['num_questions']
                        })

                        strategy_eval['quest_avg_f1_scores'].append(
                            eval_result['quest_avg_f1']
                        )
                        strategy_eval['quest_recall_scores'].append(
                            eval_result['quest_recall']
                        )

                        print(f"    F1: {eval_result['quest_avg_f1']:.4f}, "
                              f"Recall: {eval_result['quest_recall']:.4f}")
                    else:
                        print(f"    失败: {eval_result['error']}")

                except Exception as e:
                    print(f"    错误: {e}")

            # 计算汇总指标
            if strategy_eval['quest_avg_f1_scores']:
                strategy_eval['avg_quest_avg_f1'] = sum(
                    strategy_eval['quest_avg_f1_scores']
                ) / len(strategy_eval['quest_avg_f1_scores'])

                strategy_eval['avg_quest_recall'] = sum(
                    strategy_eval['quest_recall_scores']
                ) / len(strategy_eval['quest_recall_scores'])

                # 计算标准差
                import numpy as np
                strategy_eval['std_quest_avg_f1'] = np.std(
                    strategy_eval['quest_avg_f1_scores']
                )
                strategy_eval['std_quest_recall'] = np.std(
                    strategy_eval['quest_recall_scores']
                )

            evaluation_summary['strategies'].append(strategy_eval)

        # 保存评估结果
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"ragquesteval_{os.path.basename(results_file)}"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)

        # 打印汇总
        print(f"\n{'='*60}")
        print(f"评估汇总")
        print(f"{'='*60}")

        for strategy_eval in evaluation_summary['strategies']:
            print(f"\n策略: {strategy_eval['strategy']}")
            if 'avg_quest_avg_f1' in strategy_eval:
                print(f"  Quest Avg F1: {strategy_eval['avg_quest_avg_f1']:.4f} "
                      f"± {strategy_eval['std_quest_avg_f1']:.4f}")
                print(f"  Quest Recall: {strategy_eval['avg_quest_recall']:.4f} "
                      f"± {strategy_eval['std_quest_recall']:.4f}")
                print(f"  样本数: {len(strategy_eval['samples'])}")

        print(f"\n结果已保存到: {output_file}")

        return evaluation_summary


def main():
    parser = argparse.ArgumentParser(description='RAGQuestEval 评估')
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='测试结果文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/ragquesteval_results',
        help='输出目录'
    )

    args = parser.parse_args()

    # 创建评估器
    evaluator = RAGQuestEval()

    # 运行评估
    evaluator.evaluate_results(
        results_file=args.results_file,
        output_dir=args.output_dir
    )

    print("\n[OK] 评估完成！")


if __name__ == '__main__':
    main()