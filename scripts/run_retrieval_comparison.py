#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检索策略对比实验
根据CRUD-RAG论文的RAGQuestEval指标，对比不同检索策略的效果
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RETRIEVAL_VECTOR_TOP_K,
    RETRIEVAL_KEYWORD_TOP_K,
    RETRIEVAL_GRAPH_MAX,
)
from src.eval_runner import retrieve_by_mode
from src.agents.workflow import run_advanced_workflow


class RetrievalStrategyTester:
    """检索策略测试器"""

    def __init__(self):
        # 初始化向量存储和知识图谱
        self.vectorstore = self._init_vectorstore()
        self.graph = self._init_graph()

    def _init_vectorstore(self):
        """初始化向量存储"""
        try:
            from langchain_postgres import PGVector
            from src.config import POSTGRES_DSN, PGVECTOR_COLLECTION, EMBEDDING_MODEL

            if not POSTGRES_DSN:
                print("警告: 未配置 POSTGRES_DSN，向量检索将不可用")
                return None

            # 创建 embeddings
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = PGVector(
                connection=POSTGRES_DSN,
                collection_name=PGVECTOR_COLLECTION,
                embeddings=embeddings,
            )
            return vectorstore
        except Exception as e:
            print(f"警告: 初始化向量存储失败: {e}")
            return None

    def _init_graph(self):
        """初始化知识图谱"""
        try:
            import pickle
            from src.config import KG_PERSIST_PATH

            if KG_PERSIST_PATH.exists():
                with open(KG_PERSIST_PATH, 'rb') as f:
                    graph = pickle.load(f)
                return graph
            else:
                print("警告: 知识图谱文件不存在")
                return None
        except Exception as e:
            print(f"警告: 初始化知识图谱失败: {e}")
            return None

    def load_crud_data(self, data_path: str, max_samples: int = 20) -> List[Dict]:
        """加载CRUD-RAG数据集

        Args:
            data_path: CRUD-RAG数据集路径
            max_samples: 最大样本数

        Returns:
            数据样本列表
        """
        print(f"加载数据: {data_path}")

        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否是 CRUD-RAG 格式
            if isinstance(data, dict):
                # 使用 questanswer_1doc 数据
                data = data.get('questanswer_1doc', data.get('questanswer', []))

                if isinstance(data, dict):
                    # 如果还是字典，可能是按分组的
                    data = list(data.values())

        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError("不支持的数据格式")

        samples = data[:max_samples]

        print(f"加载了 {len(samples)} 个样本")
        return samples

    def test_strategy(
        self,
        samples: List[Dict],
        strategy_name: str,
        strategy_mode: str
    ) -> Dict:
        """测试单个检索策略

        Args:
            samples: 测试样本
            strategy_name: 策略名称
            strategy_mode: 策略模式

        Returns:
            测试结果
        """
        print(f"\n{'='*60}")
        print(f"测试策略: {strategy_name}")
        print(f"模式: {strategy_mode}")
        print(f"{'='*60}")

        results = []

        start_time = time.time()

        for i, sample in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] 处理样本...")

            try:
                # 获取问题（CRUD-RAG 格式使用 'questions' 字段）
                question = sample.get('questions', sample.get('question', sample.get('question_text', '')))

                if not question:
                    print(f"跳过样本 {i}: 没有问题")
                    continue

                print(f"  问题: {question[:50]}...")

                # 检索
                print(f"  检索中...")
                retrieved_docs = retrieve_by_mode(
                    query=question,
                    vectorstore=self.vectorstore,
                    graph=self.graph,
                    mode=strategy_mode,
                    vector_top_k=RETRIEVAL_VECTOR_TOP_K,
                    keyword_top_k=RETRIEVAL_KEYWORD_TOP_K,
                    graph_max=RETRIEVAL_GRAPH_MAX,
                )
                print(f"  检索到 {len(retrieved_docs)} 个文档")

                # 生成答案
                print(f"  生成答案中...")
                state = run_advanced_workflow(
                    question,
                    retrieved_docs,
                    graph=self.graph
                )
                generated_answer = state.get('final_answer', '')

                # 获取参考答案（CRUD-RAG 格式使用 'answers' 字段）
                reference_answer = sample.get('answers', sample.get('answer', sample.get('reference', '')))

                # 保存结果
                results.append({
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'retrieved_docs_count': len(retrieved_docs),
                })

                print(f"  [OK] 完成")

            except Exception as e:
                print(f"  ✗ 错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed_time = time.time() - start_time

        result = {
            'strategy': strategy_name,
            'mode': strategy_mode,
            'results': results,
            'total_samples': len(samples),
            'successful_samples': len(results),
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n策略 {strategy_name} 测试完成")
        print(f"  成功: {result['successful_samples']}/{result['total_samples']}")
        print(f"  耗时: {elapsed_time:.2f}秒")

        return result

    def run_comparison(
        self,
        data_path: str,
        strategies: List[Dict],
        max_samples: int = 20,
        output_dir: str = "datasets/retrieval_comparison"
    ) -> Dict:
        """运行检索策略对比

        Args:
            data_path: 数据路径
            strategies: 策略列表
            max_samples: 最大样本数
            output_dir: 输出目录

        Returns:
            对比结果
        """
        print(f"\n{'='*60}")
        print(f"检索策略对比实验")
        print(f"{'='*60}")
        print(f"数据: {data_path}")
        print(f"样本数: {max_samples}")
        print(f"策略数: {len(strategies)}")
        print(f"输出目录: {output_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        samples = self.load_crud_data(data_path, max_samples)

        # 测试每个策略
        all_results = []

        for strategy in strategies:
            result = self.test_strategy(
                samples,
                strategy['name'],
                strategy['mode']
            )
            all_results.append(result)

        # 保存结果
        results_summary = {
            'data_path': data_path,
            'max_samples': max_samples,
            'strategies': all_results,
            'timestamp': datetime.now().isoformat()
        }

        output_file = os.path.join(output_dir, f"comparison_results_{max_samples}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"所有策略测试完成")
        print(f"结果已保存到: {output_file}")
        print(f"{'='*60}")

        return results_summary


def get_default_strategies() -> List[Dict]:
    """获取默认的检索策略配置

    Returns:
        策略列表
    """
    return [
        {
            'name': 'vector_only',
            'description': '仅向量检索',
            'mode': 'vector'
        },
        {
            'name': 'vector_keyword',
            'description': '向量 + 关键词检索',
            'mode': 'vector_keyword'
        },
        {
            'name': 'full_hybrid',
            'description': '完整混合检索',
            'mode': 'full'
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='检索策略对比实验')
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
        default='datasets/retrieval_comparison',
        help='输出目录'
    )
    parser.add_argument(
        '--custom-config',
        type=str,
        default=None,
        help='自定义策略配置文件（JSON）'
    )

    args = parser.parse_args()

    # 创建测试器
    tester = RetrievalStrategyTester()

    # 获取策略配置
    if args.custom_config:
        with open(args.custom_config, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
    else:
        strategies = get_default_strategies()

    print(f"\n将测试 {len(strategies)} 个策略:")
    for strategy in strategies:
        print(f"  - {strategy['name']}: {strategy['description']}")

    # 运行对比实验
    results = tester.run_comparison(
        data_path=args.data_path,
        strategies=strategies,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )

    print("\n[OK] 实验完成！")


if __name__ == '__main__':
    main()