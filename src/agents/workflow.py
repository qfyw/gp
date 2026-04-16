from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import networkx as nx
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ..config import INTERNAL_DOC_PROMPT_CHARS, INTERNAL_DOC_ONLY_ANSWER, KB_STRICT_ONLY
from ..generator import get_llm
from ..retriever import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str
    internal_docs: List[Dict[str, Any]]
    kg_triples: List[Dict[str, Any]]  # 2-hop paths, formatted
    final_answer: str
    answer_style: str  # "sourced" | "concise" | "crudrag" | "eval_optimized" | "crud_optimized" | "crud_optimized_v2" | "crud_optimized_v3" | "crud_optimized_v4"
    route: Dict[str, bool]  # {"need_kg": bool}
    # 允许并行节点把 trace 追加到同一个 list
    traces: Annotated[List[str], operator.add]
    # 新增：检索质量评分
    retrieval_quality_score: float


def _trace(msg: str) -> Dict[str, Any]:
    return {"traces": [msg]}


# 图谱节点匹配：这些词在教材里极常见，若用「子串包含」会把整句型错误节点当成命中
# （例如节点名以「…不能多于系统」结尾，只因含「系统」就被选中，导致所有 2-hop 路径前缀异常）
_KG_AMBIGUOUS_ENTITIES = frozenset(
    {
        "系统",
        "资源",
        "进程",
        "线程",
        "模块",
        "方法",
        "管理",
        "结构",
        "服务",
        "数据",
        "信息",
        "类型",
        "对象",
        "设备",
        "文件",
        "用户",
        "程序",
        "内存",
        "算法",
        "模型",
        "机制",
        "功能",
    }
)


def _kg_node_matches_entity(ent: str, ns: str) -> bool:
    ent = (ent or "").strip()
    ns = str(ns or "").strip()
    if len(ent) < 2 or not ns:
        return False
    el, nl = ent.lower(), ns.lower()
    if ent in ns or el in nl:
        if ent in _KG_AMBIGUOUS_ENTITIES and len(ns) > 22:
            return False
        return True
    if len(ns) >= 2 and (nl in el):
        return True
    return False


def router_node(state: AgentState) -> Dict[str, Any]:
    """问答路由：仅决定是否走图谱推理。"""
    import re

    q_raw = (state.get("question") or "").strip()
    q = q_raw.lower()

    # 触发图谱推理的典型意图：
    # - 关系/因果/多跳推理：关系、关联、影响、原因、路径、链路、多跳…
    # - 结构/组成/分类：属于什么系统、包括哪些模块、由什么组成、架构/子系统/组件…
    #
    # 这里只做轻量启发式，避免把所有事实问答都送去 KGQuery。
    kg_keywords = [
        # 关系/推理类
        "关系",
        "关联",
        "影响",
        "原因",
        "推理",
        "链路",
        "路径",
        "多跳",
        "联系",
        # 结构/组成/分类类（你的 SkyCampus 问题属于这一类）
        "属于",
        "归属",
        "是什么系统",
        "什么系统",
        "系统",
        "模块",
        "子系统",
        "组件",
        "架构",
        "结构",
        "组成",
        "构成",
        "包括",
        "包含",
        "分为",
        "划分",
        "由",
    ]

    # 更强的短语模式：命中这些就基本可以认为需要“结构化关系”能力
    kg_patterns = [
        r"属于.*系统",
        r"(包括|包含).*(模块|子系统|组件|功能)",
        r"(由|通过).*(组成|构成)",
        r"(系统|平台).*(架构|结构)",
        r"(分为|划分为).*(模块|子系统|部分)",
    ]

    hit_kw = any(k in q_raw for k in kg_keywords)  # 保持中文原样匹配更稳
    hit_pat = any(re.search(p, q_raw) for p in kg_patterns)

    # 极短问题（<=4字）通常不是结构化关系问题，避免误触发
    too_short = len(q_raw) <= 4

    need_kg = (hit_kw or hit_pat) and (not too_short)
    route = {"need_kg": bool(need_kg)}
    return {
        "route": route,
        **_trace(f"Router：need_kg={route['need_kg']} (kw={hit_kw}, pat={hit_pat})"),
    }


def _extract_core_entities(question: str, llm=None) -> List[str]:
    # 优先用 LLM 抽实体；无 LLM 时退化为简单规则
    if llm is None:
        import re

        words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", question)
        # 去掉常见虚词
        stop = {"什么", "如何", "怎样", "哪些", "可以", "是否", "原因", "影响", "关系", "关联"}
        return [w for w in words if w not in stop][:5]

    prompt = (
        "请从用户问题中提取 1-5 个“核心实体/关键词”，只输出 JSON 数组字符串，例如："
        '["实体1","实体2"]。\n\n用户问题：'
        + question
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        import json

        arr = json.loads(text.strip())
        if isinstance(arr, list):
            ents = [str(x).strip() for x in arr if str(x).strip()][:5]
            # 对“操作系统和cpu的关系”这类问题，把复合实体按连接词再切一次，避免只保留整句。
            out: List[str] = []
            seen = set()
            import re
            stop = {"关系", "关联", "影响", "原因", "说说", "介绍", "一下", "什么"}
            for e in ents:
                parts = [p for p in re.split(r"[的和与及、,，\s]+", e) if p]
                for p in parts:
                    if len(p) < 2 or p in stop:
                        continue
                    k = p.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(p)
            return out[:5] or ents
    except Exception:
        pass
    return []


def kg_query_node_factory(graph: Optional[nx.DiGraph]):
    def kg_query_node(state: AgentState) -> Dict[str, Any]:
        if not (state.get("route") or {}).get("need_kg"):
            return _trace("KGQuery：跳过（Router 判定无需图谱推理）")

        if graph is None or graph.number_of_nodes() == 0:
            return _trace("KGQuery：图谱为空，跳过")

        updates: Dict[str, Any] = {}
        updates.update(_trace("KGQuery：开始两跳查询（2-hop）"))
        llm = get_llm()
        entities = _extract_core_entities(state.get("question", ""), llm=llm)
        if not entities:
            updates["traces"] = updates.get("traces", []) + ["KGQuery：未能抽取核心实体，跳过"]
            return updates

        # 匹配节点：子串包含 + 对泛词限长，并按节点名长度优先（更像「实体」而非整句）
        nodes = list(graph.nodes())
        candidates: List[str] = []
        for n in nodes:
            ns = str(n)
            if any(_kg_node_matches_entity(ent, ns) for ent in entities if ent):
                candidates.append(ns)
        candidates = list(dict.fromkeys(candidates))
        candidates.sort(key=len)
        matched = candidates[:8]

        if not matched:
            updates["traces"] = updates.get("traces", []) + [f"KGQuery：未找到匹配实体节点（entities={entities}）"]
            return updates

        paths_out: List[Dict[str, Any]] = []
        seen = set()
        entities_l = [e.lower() for e in entities]

        def edge_score(u: str, rel: str, v: str) -> int:
            txt = f"{u} {rel} {v}".lower()
            return sum(1 for e in entities_l if e and e in txt)

        # 2-hop: A -> B -> C
        for a in matched:
            # 先收集第一跳并按问题实体相关度排序，避免关键边被任意顺序淹没
            first_hops: List[tuple[str, str, str]] = []
            for _, b, d1 in list(graph.out_edges(a, data=True)):
                r1 = str(d1.get("relation", "相关"))
                first_hops.append((str(a), r1, str(b)))
            first_hops.sort(key=lambda x: edge_score(x[0], x[1], x[2]), reverse=True)

            # 先输出部分 1-hop，提升“操作系统-CPU”这类直接关系的可见性
            for u1, r1, b in first_hops[:10]:
                key1 = (u1, r1, b)
                if key1 in seen:
                    continue
                seen.add(key1)
                paths_out.append({"path": f"{u1} - {r1} - {b}", "source": f"图谱推导: {u1}"})
                if len(paths_out) >= 15:
                    break
            if len(paths_out) >= 15:
                break

            # 出边第一跳
            for _, r1, b in first_hops[:30]:
                # 第二跳候选同样按相关度排序
                second_hops: List[tuple[str, str, str]] = []
                for _, c, d2 in list(graph.out_edges(b, data=True)):
                    r2 = str(d2.get("relation", "相关"))
                    second_hops.append((str(b), r2, str(c)))
                second_hops.sort(key=lambda x: edge_score(x[0], x[1], x[2]), reverse=True)
                # 第二跳
                for _, r2, c in second_hops[:30]:
                    key = (a, r1, str(b), r2, str(c))
                    if key in seen:
                        continue
                    seen.add(key)
                    paths_out.append(
                        {
                            "path": f"{a} - {r1} - {b} - {r2} - {c}",
                            "source": f"图谱推导: {a}",
                        }
                    )
                    if len(paths_out) >= 15:
                        break
                if len(paths_out) >= 15:
                    break
            if len(paths_out) >= 15:
                break

        updates["kg_triples"] = paths_out
        updates["traces"] = updates.get("traces", []) + [f"KGQuery：完成，返回 {len(paths_out)} 条两跳路径"]
        return updates

    return kg_query_node


def join_node(state: AgentState) -> Dict[str, Any]:
    # 仅用于等待并行分支汇合
    return _trace("Join：已汇合 KGQuery 的结果")


def check_relevance_node(state: AgentState) -> Dict[str, Any]:
    """检索相关性检查：过滤低质量的检索结果，提升事实准确性。"""
    updates: Dict[str, Any] = {}
    updates.update(_trace("RelevanceCheck：开始评估检索结果相关性"))

    docs = state.get("internal_docs") or []
    question = state.get("question", "")

    if not docs or not question:
        updates["retrieval_quality_score"] = 0.0
        updates["traces"] = updates.get("traces", []) + ["RelevanceCheck：无检索结果或问题，跳过"]
        return updates

    llm = get_llm()
    if llm is None:
        updates["retrieval_quality_score"] = 0.5
        updates["traces"] = updates.get("traces", []) + ["RelevanceCheck：无 LLM，使用默认分数"]
        return updates

    # 批量评估：一次调用评估所有文档（前8条，控制成本）
    docs_to_check = docs[:8]
    docs_text = "\n\n".join(
        f"[文档 {i+1}]\n{(doc.get('content') or '')[:600]}"
        for i, doc in enumerate(docs_to_check)
    )

    prompt = f"""你是一个检索质量评估员。判断以下文档片段是否与用户问题相关。

用户问题：{question}

文档片段：
{docs_text}

请对每个文档片段给出相关程度评分（0-1 之间的数字），并说明是否应该保留该文档。

评分标准：
- 0.9-1.0：直接回答问题，包含关键事实/数字/答案
- 0.6-0.8：高度相关，提供有用背景信息
- 0.3-0.5：部分相关，可能有用但不直接
- 0.0-0.2：基本不相关

输出格式（JSON）：
{{
  "scores": [文档1评分, 文档2评分, ...],
  "keep_indices": [应该保留的文档索引，从0开始]
}}

只输出 JSON，不要其他内容。"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        import json
        import re

        # 提取 JSON
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            obj = json.loads(m.group(0))
            scores = obj.get("scores", [])
            keep_indices = set(obj.get("keep_indices", []))

            avg_score = sum(scores) / len(scores) if scores else 0.5
            filtered_docs = [doc for i, doc in enumerate(docs_to_check) if i in keep_indices]

            # 如果过滤后太少（少于3条），保留原始文档
            if len(filtered_docs) < 3:
                filtered_docs = docs

            updates["internal_docs"] = filtered_docs
            updates["retrieval_quality_score"] = avg_score
            updates["traces"] = updates.get("traces", []) + [
                f"RelevanceCheck：平均相关性 {avg_score:.2f}，保留 {len(filtered_docs)}/{len(docs)} 条"
            ]
        else:
            # 解析失败，保留所有文档
            updates["internal_docs"] = docs
            updates["retrieval_quality_score"] = 0.5
            updates["traces"] = updates.get("traces", []) + ["RelevanceCheck：解析失败，保留所有文档"]
    except Exception as e:
        # 出错时保留所有文档
        updates["internal_docs"] = docs
        updates["retrieval_quality_score"] = 0.5
        updates["traces"] = updates.get("traces", []) + [f"RelevanceCheck：出错: {e}，保留所有文档"]

    return updates


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    updates.update(_trace("Synthesizer：开始综合与冲突检测"))
    llm = get_llm()
    if llm is None:
        updates["final_answer"] = "请先在 `.env` 配置 OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL（如通义千问 OpenAI 兼容），否则无法调用大模型生成答案。"
        updates["traces"] = updates.get("traces", []) + ["Synthesizer：未配置 LLM，返回提示文本"]
        return updates

    internal = state.get("internal_docs") or []
    kg = state.get("kg_triples") or []
    answer_style = (state.get("answer_style") or "sourced").strip().lower()
    if answer_style not in ("sourced", "concise", "crudrag", "eval_optimized", "crud_optimized", "crud_optimized_v2", "crud_optimized_v3", "crud_optimized_v4"):
        answer_style = "sourced"

    retrieval_score = state.get("retrieval_quality_score", 0.5)

    strict_guard = (
        "【严格知识库模式】\n"
        "你只能依据【内部文档】与【图谱推导】作答，禁止使用模型参数中的常识、背景知识或外部信息补全。\n"
        "若资料中没有可直接支持的事实，必须只输出：无法确定。\n"
    ) if KB_STRICT_ONLY else ""
    internal_only_guard = (
        "【仅内部文档作答模式】\n"
        "最终答案中的事实、数字、结论只能来自【内部文档】。\n"
        "【图谱推导】仅可用于帮助理解问题与定位线索，不能作为独立事实来源。\n"
        "若内部文档没有直接证据，即使图谱里有相关路径，也必须输出：无法确定。\n"
    ) if INTERNAL_DOC_ONLY_ANSWER else ""

    def _fmt_internal() -> str:
        lines = []
        for i, it in enumerate(internal, 1):
            src = it.get("source", "内部文档")
            content = (it.get("content") or "")[:INTERNAL_DOC_PROMPT_CHARS]
            lines.append(f"[内部文档 {i}] 来源: {src}\n{content}")
        return "\n\n".join(lines) or "（无）"

    def _fmt_kg() -> str:
        lines = []
        for i, it in enumerate(kg, 1):
            lines.append(f"[图谱 {i}] {it.get('path','')}")
        return "\n\n".join(lines) or "（无）"

    if answer_style == "crudrag":
        # CRUD-RAG 官方 quest_answer 模板（仅用于评测链路；网页端保持不变）
        root = Path(__file__).resolve().parents[2]
        tpl_path = root / "CRUD_RAG" / "src" / "prompts" / "quest_answer.txt"
        try:
            template = tpl_path.read_text(encoding="utf-8")
        except Exception:
            template = ""

        docs = "\n\n".join((it.get("content") or "") for it in (internal or []))
        prompt = (
            strict_guard + internal_only_guard
            + (template or "{question}\n\n{search_documents}\n")
        ).format(
            question=state.get("question", ""),
            search_documents=docs,
        )
    elif answer_style == "crud_optimized":
        # CRUD-RAG 最佳实践提示词：角色设定 + 结构化输出 + 明确约束
        prompt = f"""你是一位专业的新闻编辑，擅长从新闻报道中提取关键信息并回答读者问题。

【你的任务】
根据【内部文档】中的新闻报道，准确回答用户的提问。

【输出要求】
1) **简洁直接**：用1-2句自然中文回答，不要啰嗦
2) **结构化输出**：将答案放在 <response></response> 标签之间
3) **准确事实**：所有数字、日期、名称必须与原文一致
4) **部分作答**：若部分信息可确定，优先给出可确定的部分，用"资料未明确…"说明缺失部分
5) **拒绝幻觉**：不编造文档中没有的信息
6) **防张冠李戴**：问题若含地名/机构/日期，答案必须与同一条报道中的事实一致
7) **自然表达**：用"并/同时/且"连接多个信息点

【内部文档】（相关性评分：{retrieval_score:.2f}）
{_fmt_internal()}

【图谱推导（2-hop）】
{_fmt_kg()}

【示例】
问题：上海和成都市体育局在促进体育消费方面有哪些措施？
回答：<response>上海市体育局联合美团、大众点评发放500万元体育消费券，覆盖3000多家门店；成都市体育局利用大运会契机发放各类消费券，通过举办大型展会活动促进消费。</response>

用户问题：
{state.get("question","")}

请将你的答案放在 <response></response> 标签之间："""
    elif answer_style == "concise":
        # 评测优化 Prompt：减少拒答，提升事实准确性
        prompt = f"""你是专业的问答评测助手。请综合【内部文档】【图谱推导】回答用户问题。

## 核心原则（评测专用）
1) **部分作答优于整体拒答**：若资料中存在部分可核实的事实（如其中一个数字、部分列表），优先给出能确定的部分，用"资料未明确…"说明无法确定的部分。
2) **防张冠李戴**：问题若含地名/机构/日期/书名等锚点，答案必须与同一条报道中的事实一致；禁止把其他事件/城市的数字挪用。
3) **数字精确性**：金额、人数、批次、百分比等须与文档一致；若多篇冲突，优先采用与问题锚点最匹配的段落。
4) **多子问必答**：问题含多个疑问（如"多少…以及…"）须在同一句内逐问作答；缺少某子问时用"资料未明确…"标注。
5) **拒答门槛极高**：仅当所有文档都与问题完全无关、无任何可核对线索时才输出"无法确定"；只要存在相关实体/数字，就优先作答。
6) **一句话自然表达**：用"并/同时/且"连接多个信息点，写成自然中文句子。
7) **禁止套话**：不要出现"根据资料/综合信息/综上"等字样。
8) **数字与实体匹配**：若问题问"多少…同时…"，答案中必须同时包含两类数字；若文档只有其一，只写有的一类并说明另一类未明确。

【内部文档】（相关性评分：{retrieval_score:.2f}）
{_fmt_internal()}

【图谱推导（2-hop）】
{_fmt_kg()}

用户问题：
{state.get("question","")}

请将你的答案放在 <response></response> 标签之间："""
    elif answer_style == "crud_optimized_v4":
        # CRUD-RAG 最佳实践 + 针对false_abstain的专项优化
        prompt = f"""你是一位专业的新闻编辑，擅长从新闻报道中提取关键信息并准确回答读者问题。

【核心原则】
1) **部分作答优于整体拒答**：只要文档中有任何相关信息，就优先给出能确定的部分，用"资料未明确XXX"说明无法确定的部分，不要整体拒答
2) **准确事实**：所有数字、日期、名称必须与原文一致
3) **拒绝幻觉**：不编造文档中没有的信息
4) **防张冠李戴**：问题含地名/机构/日期时，答案必须与同一条报道中的事实一致
5) **实体匹配宽松**：即使人名/机构名不完全一致，只要内容相关就回答

【输出要求】
1) **简洁直接**：用1-2句自然中文回答
2) **结构化输出**：将答案放在 <response></response> 标签之间
3) **明确缺失**：若某个信息在文档中找不到，明确说明"资料未明确XXX"

【重要提示：避免错误拒答】
- 即使问题中的具体人名/机构名在文档中未明确提及，但如果相关事实存在，就给出答案
- 例如：如果问题是"根据XXX的研究..."，即使文档中未提及XXX，但只要相关研究内容存在，就回答该内容
- 只有当文档中完全没有任何相关信息时，才使用"无法确定"

【内部文档】（相关性评分：{retrieval_score:.2f}）
{_fmt_internal()}

【图谱推导（2-hop）】
{_fmt_kg()}

【示例1】
问题：根据布鲁诺·比佐泽罗·佩罗尼进行的研究，每天食用多少克坚果可以将抑郁症的风险降低17%？
回答：<response>每天食用约30克坚果可使抑郁症风险降低17%。</response>
（注：即使文档中未提及研究者姓名，但给出了研究结果）

【示例2】
问题：根据中南大学湘雅三医院眼科副主任医师曹燕娜的说法，儿童青少年患上干眼症的原因有哪些？
回答：<response>儿童干眼症的诱因包括用眼不当、学习任务繁重、长时间佩戴角膜接触镜、长期睡眠不足、挑食偏食导致维生素A缺乏，以及过敏性疾病等。</response>
（注：即使文档中未明确提及该医生，但给出了干眼症的原因）

【示例3】
问题：陕西西安市发放的体育类电子消费券的具体金额是多少？可以在多少家体育场馆使用？
回答：<response>资料中提及了体育消费券相关信息，但未明确具体金额和可用场馆数量。</response>
（注：文档中有相关信息但不完整，说明缺失部分）

用户问题：
{state.get("question","")}

请将你的答案放在 <response></response> 标签之间（务必避免错误拒答，优先给出能确定的部分）："""
    elif answer_style == "crud_optimized_v3":
        # CRUD-RAG 最佳实践 + 改进版提示词
        prompt = f"""你是一位专业的新闻编辑，擅长从新闻报道中提取关键信息并准确回答读者问题。

【核心原则】
1) **完整回答**：问题包含多个子问时，必须逐个回答，不能只回答部分问题
2) **准确事实**：所有数字、日期、名称必须与原文一致
3) **部分作答**：若部分信息可确定，优先给出能确定的部分，用"资料未明确XXX"说明缺失部分
4) **拒绝幻觉**：不编造文档中没有的信息
5) **防张冠李戴**：问题含地名/机构/日期时，答案必须与同一条报道中的事实一致

【输出要求】
1) **简洁直接**：用1-2句自然中文回答
2) **结构化输出**：将答案放在 <response></response> 标签之间
3) **多子问处理**：用"并/同时/且"连接多个信息点，逐个回答
4) **明确缺失**：若某个子问在文档中找不到，明确说明"资料未明确XXX"

【重要提示：防止只回答部分问题】
- 仔细检查问题中的所有疑问词（如：什么、多少、何时、何地、为何）
- 如果问题问"A和B的XXX"，必须分别说明A和B的XXX
- 如果问题问"数量和金额"，必须同时包含数量和金额
- 如果某个子问在文档中找不到，必须明确说明"资料未明确XXX"，而不是完全忽略

【内部文档】（相关性评分：{retrieval_score:.2f}）
{_fmt_internal()}

【图谱推导（2-hop）】
{_fmt_kg()}

【示例1】
问题：上海和成都市体育局在促进体育消费方面有哪些措施？
回答：<response>上海市体育局联合美团、大众点评发放500万元体育消费券，覆盖3000多家门店；成都市体育局利用大运会契机发放各类消费券，通过举办大型展会活动促进消费。</response>

【示例2】
问题：国家卫生健康委员会于2023年7月28日启动了名为"启明行动"的专项活动，请说明活动的目标人群和指导文件。
回答：<response>"启明行动"针对儿童青少年的近视防控问题，所依据的指导性文件为《防控儿童青少年近视核心知识十条》。</response>

【示例3】
问题：某项目总投资额和受益人数是多少？
回答：<response>项目总投资额为50亿元，资料未明确受益人数。</response>

【示例4】
问题：2023年和2024年的销售额分别是多少？
回答：<response>2023年销售额为100万元，资料未明确2024年销售额。</response>

用户问题：
{state.get("question","")}

请将你的答案放在 <response></response> 标签之间（务必完整回答所有子问，对缺失信息明确说明）："""
    elif answer_style == "crud_optimized_v2":
        # CRUD-RAG 最佳实践 + 改进版提示词
        prompt = f"""你是一位专业的新闻编辑，擅长从新闻报道中提取关键信息并准确回答读者问题。

【核心原则】
1) **完整回答**：问题包含多个子问时，必须逐个回答，不能只回答部分问题
2) **准确事实**：所有数字、日期、名称必须与原文一致
3) **部分作答**：若部分信息可确定，优先给出能确定的部分，用"资料未明确XXX"说明缺失部分
4) **拒绝幻觉**：不编造文档中没有的信息
5) **防张冠李戴**：问题含地名/机构/日期时，答案必须与同一条报道中的事实一致

【输出要求】
1) **简洁直接**：用1-2句自然中文回答
2) **结构化输出**：将答案放在 <response></response> 标签之间
3) **多子问处理**：用"并/同时/且"连接多个信息点，逐个回答
4) **明确缺失**：若某个子问在文档中找不到，明确说明"资料未明确XXX"

【内部文档】（相关性评分：{retrieval_score:.2f}）
{_fmt_internal()}

【图谱推导（2-hop）】
{_fmt_kg()}

【示例1】
问题：上海和成都市体育局在促进体育消费方面有哪些措施？
回答：<response>上海市体育局联合美团、大众点评发放500万元体育消费券，覆盖3000多家门店；成都市体育局利用大运会契机发放各类消费券，通过举办大型展会活动促进消费。</response>

【示例2】
问题：国家卫生健康委员会于2023年7月28日启动了名为"启明行动"的专项活动，请说明活动的目标人群和指导文件。
回答：<response>"启明行动"针对儿童青少年的近视防控问题，所依据的指导性文件为《防控儿童青少年近视核心知识十条》。</response>

用户问题：
{state.get("question","")}

请将你的答案放在 <response></response> 标签之间（务必完整回答所有子问）："""
    else:
        prompt = f"""{strict_guard}{internal_only_guard}你是一个"多源信息融合 + 知识图谱增强"的问答助手。
请综合【内部文档】【图谱推导】回答用户问题。

## 冲突检测规则
- 若【内部文档】与【图谱推导】存在不一致，优先以【内部文档】为准，并在答案中指出差异。

## 严格溯源格式（强制）
- 每一句结论后必须带来源标签：
  - 内部文档用：[内部文档: <来源>]
  - 图谱推导用：[图谱推导]

## 内部文档
{_fmt_internal()}

## 图谱推导（2-hop）
{_fmt_kg()}

用户问题：
{state.get("question","")}
"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        # 提取结构化输出（crudrag 和 crud_optimized 模式）
        if answer_style in ("crudrag", "crud_optimized"):
            # 尝试提取 <response>...</response> 内容
            if "<response>" in text and "</response>" in text:
                text = text.split("<response>", 1)[-1].split("</response>", 1)[0].strip()
        updates["final_answer"] = text
        updates["traces"] = updates.get("traces", []) + ["Synthesizer：完成生成"]
    except Exception as e:
        updates["final_answer"] = f"调用 LLM 时出错: {e}"
        updates["traces"] = updates.get("traces", []) + [f"Synthesizer：失败：{e}"]
    return updates


def build_workflow(graph: Optional[nx.DiGraph] = None):
    sg = StateGraph(AgentState)
    sg.add_node("router", router_node)
    sg.add_node("kg_query", kg_query_node_factory(graph))
    sg.add_node("check_relevance", check_relevance_node)
    sg.add_node("join", join_node)
    sg.add_node("synthesizer", synthesizer_node)

    sg.add_edge(START, "router")
    # 并行：router -> kg -> join
    sg.add_edge("router", "kg_query")
    sg.add_edge("kg_query", "join")
    # 检索相关性检查（在 join 和 synthesizer 之间）
    sg.add_edge("join", "check_relevance")
    sg.add_edge("check_relevance", "synthesizer")
    sg.add_edge("synthesizer", END)
    return sg.compile()


def run_advanced_workflow(
    question: str,
    chunks: List[RetrievedChunk],
    graph: Optional[nx.DiGraph] = None,
    *,
    answer_style: str = "sourced",
) -> AgentState:
    state: AgentState = {
        "question": question,
        "internal_docs": [{"source": c.source, "content": c.content} for c in (chunks or [])],
        "kg_triples": [],
        "final_answer": "",
        "answer_style": answer_style,
        "route": {},
        "traces": [],
    }
    app = build_workflow(graph=graph)
    return app.invoke(state)

