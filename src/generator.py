# 答案生成与溯源模块：基于 Context 调用 LLM，并标注来源
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .retriever import RetrievedChunk
from .config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class GenerationResult:
    """生成结果：答案文本 + 溯源列表。"""
    answer: str
    sources: List[dict]  # [{"source": "文档A.pdf 第3页", "content": "片段..."}, ...]


def build_context_with_sources(chunks: List[RetrievedChunk]) -> str:
    """将检索块格式化为带来源标注的上下文文本。"""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] [来源: {c.source}]\n{c.content}")
    return "\n\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    """构造要求基于上下文回答并标注来源的 Prompt。"""
    return f"""你是一个基于给定资料回答问题的助手。请仅根据以下「参考资料」回答问题，并在答案中标注引用来源（格式如 [来源: 文档名 第X页] 或 [来源: 知识图谱节点: XXX]）。

## 参考资料
{context}

## 用户问题
{question}

## 要求
1. 仅根据上述参考资料回答，不要编造。
2. 答案中必须标注所引用内容的来源。
3. 若参考资料无法回答，请说明「根据现有资料无法回答」并简要说明缺少哪类信息。"""


def get_llm(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
    """获取 LangChain 兼容的 LLM（OpenAI 兼容 API，可接 DeepSeek/Qwen/Kimi）。"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    api_key = api_key or OPENAI_API_KEY
    base_url = base_url or OPENAI_API_BASE
    model = model or OPENAI_MODEL

    if not api_key:
        return None

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url if base_url else None,
        temperature=0.3,
    )


def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
    llm=None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> GenerationResult:
    """
    根据检索到的 chunks 生成答案，并返回溯源列表。
    若未传入 llm 且无环境配置，则返回占位答案与溯源。
    """
    context = build_context_with_sources(chunks)
    prompt = build_prompt(question, context)

    if llm is None:
        llm = get_llm(api_key=api_key, base_url=base_url, model=model)

    if llm is None:
        return GenerationResult(
            answer="请配置 LLM API（如 OPENAI_API_KEY、OPENAI_API_BASE、OPENAI_MODEL 或使用 .env），或传入 llm 实例后再进行回答。当前仅展示检索到的参考资料。",
            sources=[{"source": c.source, "content": c.content} for c in chunks],
        )

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        answer = f"调用 LLM 时出错: {e}\n\n以下是检索到的参考资料，可供参考：\n\n{context}"
    sources = [{"source": c.source, "content": c.content} for c in chunks]
    return GenerationResult(answer=answer, sources=sources)
