#!/usr/bin/env python3
"""
实体感知检索模块：提取问题中的关键实体，只检索包含这些实体的文档。

优化策略：
1. 实体提取：人名、地名、机构名、专有名词、数字、日期
2. 实体扩展：同义词、别名、简称
3. 实体过滤：只检索包含关键实体的文档
4. 实体加权：包含更多实体的文档得分更高

用法：
  from src.entity_aware_retriever import entity_aware_hybrid_retrieve
  chunks = entity_aware_hybrid_retrieve(query, vectorstore, graph, top_k=10)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .config import (
    BM25_TOP_K,
    BM25_ENABLED,
    RERANK_DOC_CAP,
    RERANK_ENABLED,
    RERANK_RECALL_MULT,
    RRF_K,
)
from .pg_db import bm25_search as pg_bm25_search
from .pg_db import keyword_search as pg_keyword_search
from .reranker import rerank_doc_chunks
from .retriever import (
    RetrievedChunk,
    bm25_search,
    boost_chunks_by_query_anchors,
    expand_query_for_hybrid,
    fuse_doc_chunks_rrf,
    graph_search,
    merge_vector_keyword_chunks,
    normalize_query_for_search,
    query_anchor_phrases,
    vector_search,
)

try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    PGVector = Any  # type: ignore


@dataclass
class EntityMatch:
    """实体匹配信息"""
    entity: str
    match_type: str  # "exact" | "fuzzy" | "alias"
    positions: List[int]  # 在问题中的位置


def extract_entities_from_question(question: str) -> Dict[str, List[EntityMatch]]:
    """
    从问题中提取关键实体。

    实体类型：
    - person: 人名（2-4字中文，或英文人名）
    - location: 地名（城市、省份、国家）
    - organization: 机构名（政府、企业、学校）
    - proper_noun: 专有名词（产品名、活动名、政策名）
    - date: 日期（YYYY年MM月DD日等格式）
    - number: 数字（金额、数量、百分比）
    """
    q = question or ""
    entities: Dict[str, List[EntityMatch]] = {
        "person": [],
        "location": [],
        "organization": [],
        "proper_noun": [],
        "date": [],
        "number": [],
    }

    # 1. 提取日期
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}[日号]",
        r"\d{4}年\d{1,2}月",
        r"\d{1,2}月\d{1,2}[日号]",
    ]
    for pattern in date_patterns:
        for m in re.finditer(pattern, q):
            entities["date"].append(
                EntityMatch(entity=m.group(0), match_type="exact", positions=[m.start()])
            )

    # 2. 提取数字（金额、数量、百分比）
    number_patterns = [
        r"\d+万元?",
        r"\d+亿元?",
        r"\d+百万元?",
        r"\d+个?",
        r"\d+家?",
        r"\d+名?",
        r"\d+项?",
        r"\d+[多条|个|种|类]",
        r"\d+\.\d+%",
        r"\d+%",
    ]
    for pattern in number_patterns:
        for m in re.finditer(pattern, q):
            entities["number"].append(
                EntityMatch(entity=m.group(0), match_type="exact", positions=[m.start()])
            )

    # 3. 提取地名（中国城市、省份）
    # 常见省份/直辖市
    provinces = ["北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
                 "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
                 "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
                 "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门"]
    for prov in provinces:
        if prov in q:
            positions = [m.start() for m in re.finditer(re.escape(prov), q)]
            entities["location"].append(
                EntityMatch(entity=prov, match_type="exact", positions=positions)
            )

    # 城市（常见城市名，2-3字）
    cities = ["石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德",
              "沧州", "廊坊", "衡水", "太原", "大同", "阳泉", "长治", "晋城",
              "朔州", "晋中", "运城", "忻州", "临汾", "吕梁", "呼和浩特", "包头",
              "乌海", "赤峰", "通辽", "鄂尔多斯", "呼伦贝尔", "巴彦淖尔", "乌兰察布",
              "兴安盟", "锡林郭勒", "阿拉善", "沈阳", "大连", "鞍山", "抚顺", "本溪",
              "丹东", "锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭", "朝阳", "葫芦岛",
              "长春", "吉林", "四平", "辽源", "通化", "白山", "松原", "白城", "延边",
              "哈尔滨", "齐齐哈尔", "鸡西", "鹤岗", "双鸭山", "大庆", "伊春", "佳木斯",
              "七台河", "牡丹江", "黑河", "绥化", "大兴安岭", "南京", "无锡", "徐州",
              "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州",
              "宿迁", "杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州",
              "舟山", "台州", "丽水", "合肥", "芜湖", "蚌埠", "淮南", "马鞍山", "淮北",
              "铜陵", "安庆", "黄山", "滁州", "阜阳", "宿州", "六安", "亳州", "池州",
              "宣城", "福州", "厦门", "莆田", "三明", "泉州", "漳州", "南平", "龙岩",
              "宁德", "南昌", "景德镇", "萍乡", "九江", "新余", "鹰潭", "赣州", "吉安",
              "宜春", "抚州", "上饶", "济南", "青岛", "淄博", "枣庄", "东营", "烟台",
              "潍坊", "济宁", "泰安", "威海", "日照", "莱芜", "临沂", "德州", "聊城",
              "滨州", "菏泽", "郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡",
              "焦作", "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口",
              "驻马店", "武汉", "黄石", "十堰", "宜昌", "襄阳", "鄂州", "荆门", "孝感",
              "荆州", "黄冈", "咸宁", "随州", "恩施", "仙桃", "潜江", "天门", "神农架",
              "长沙", "株洲", "湘潭", "衡阳", "邵阳", "岳阳", "常德", "张家界", "益阳",
              "郴州", "永州", "怀化", "娄底", "湘西", "广州", "韶关", "深圳", "珠海",
              "汕头", "佛山", "江门", "湛江", "茂名", "肇庆", "惠州", "梅州", "汕尾",
              "河源", "阳江", "清远", "东莞", "中山", "潮州", "揭阳", "云浮", "南宁",
              "柳州", "桂林", "梧州", "北海", "防城港", "钦州", "贵港", "玉林", "百色",
              "贺州", "河池", "来宾", "崇左", "海口", "三亚", "三沙", "儋州", "成都",
              "自贡", "攀枝花", "泸州", "德阳", "绵阳", "广元", "遂宁", "内江", "乐山",
              "南充", "眉山", "宜宾", "广安", "达州", "雅安", "巴中", "资阳", "阿坝",
              "甘孜", "凉山", "贵阳", "六盘水", "遵义", "安顺", "毕节", "铜仁", "黔西南",
              "黔东南", "黔南", "昆明", "曲靖", "玉溪", "保山", "昭通", "丽江", "普洱",
              "临沧", "楚雄", "红河", "文山", "西双版纳", "大理", "德宏", "怒江", "迪庆",
              "拉萨", "日喀则", "昌都", "林芝", "山南", "那曲", "阿里", "西安", "铜川",
              "宝鸡", "咸阳", "渭南", "延安", "汉中", "榆林", "安康", "商洛", "兰州",
              "嘉峪关", "金昌", "白银", "天水", "武威", "张掖", "平凉", "酒泉", "庆阳",
              "定西", "陇南", "临夏", "甘南", "西宁", "海东", "海北", "黄南", "海南",
              "果洛", "玉树", "海西", "银川", "石嘴山", "吴忠", "固原", "中卫", "乌鲁木齐",
              "克拉玛依", "吐鲁番", "哈密", "昌吉", "博尔塔拉", "巴音郭楞", "阿克苏",
              "克孜勒苏", "喀什", "和田", "伊犁", "塔城", "阿勒泰", "台北", "高雄",
              "基隆", "新竹", "台中", "台南", "嘉义", "新北", "桃园", "苗栗", "彰化",
              "南投", "云林", "嘉义", "屏东", "宜兰", "花莲", "台东", "澎湖", "金门",
              "连江"]
    for city in cities:
        if city in q:
            positions = [m.start() for m in re.finditer(re.escape(city), q)]
            entities["location"].append(
                EntityMatch(entity=city, match_type="exact", positions=positions)
            )

    # 4. 提取机构名（政府机构、高校、企业）
    org_patterns = [
        r"[省市县][区市局委办]+",
        r"[大学学院学校]+",
        r"[有限公司集团]+",
        r"[医院诊所]+",
        r"[委员会办公厅]+",
    ]
    for pattern in org_patterns:
        for m in re.finditer(pattern, q):
            entities["organization"].append(
                EntityMatch(entity=m.group(0), match_type="exact", positions=[m.start()])
            )

    # 5. 提取专有名词（英文驼峰命名、中文引号内容）
    # 英文驼峰命名（如 SkyCampus、OpenAI）
    camel_case_pattern = r"(?<![A-Za-z0-9])([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"
    for m in re.finditer(camel_case_pattern, q):
        entities["proper_noun"].append(
            EntityMatch(entity=m.group(1), match_type="exact", positions=[m.start()])
        )

    # 中文引号内容（如"启明行动"、"狮城人才"）
    quote_pattern = r"「([^」]{2,10})」|『([^』]{2,10})』|\"([^\"]{2,10})\""
    for m in re.finditer(quote_pattern, q):
        entity = next((g for g in m.groups() if g), "")
        if entity:
            entities["proper_noun"].append(
                EntityMatch(entity=entity, match_type="exact", positions=[m.start()])
            )

    # 6. 提取人名（2-4字中文，通常在"姓+名"结构中）
    # 常见姓氏
    surnames = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵"]
    for i in range(len(q) - 2):
        if q[i] in surnames:
            name = q[i:i+3] if i+3 <= len(q) else q[i:i+2]
            if len(name) >= 2 and name not in cities and name not in provinces:
                entities["person"].append(
                    EntityMatch(entity=name, match_type="fuzzy", positions=[i])
                )

    return entities


def expand_entities(entities: Dict[str, List[EntityMatch]]) -> Dict[str, Set[str]]:
    """
    扩展实体：同义词、别名、简称。
    """
    expanded: Dict[str, Set[str]] = {
        "person": set(),
        "location": set(),
        "organization": set(),
        "proper_noun": set(),
        "date": set(),
        "number": set(),
    }

    # 地点别名映射
    location_aliases = {
        "北京": ["北京市", "京"],
        "上海": ["上海市", "沪"],
        "天津": ["天津市", "津"],
        "重庆": ["重庆市", "渝"],
        "广州": ["广州市", "穗"],
        "深圳": ["深圳市"],
        "南京": ["南京市"],
        "杭州": ["杭州市"],
        "武汉": ["武汉市"],
        "成都": ["成都市"],
        "西安": ["西安市"],
        "长沙": ["长沙市"],
        "沈阳": ["沈阳市"],
        "石家庄": ["石家庄市"],
        "保定": ["保定市"],
        "沧州": ["沧州市"],
        "邯郸": ["邯郸市"],
        "邢台": ["邢台市"],
    }

    # 专有名词别名映射
    proper_noun_aliases = {
        "启明行动": ["启明行动专项活动", "启明行动计划"],
        "狮城人才": ["狮城人才计划", "狮城人才工程"],
        "双盲评审": ["招标投标双盲评审", "双盲评审措施"],
        "大运会": ["世界大学生夏季运动会"],
        "女足世界杯": ["国际足联女足世界杯", "FIFA女足世界杯"],
    }

    for entity_type, matches in entities.items():
        for match in matches:
            entity = match.entity
            expanded[entity_type].add(entity)

            # 应用别名映射
            if entity_type == "location" and entity in location_aliases:
                expanded[entity_type].update(location_aliases[entity])
            elif entity_type == "proper_noun" and entity in proper_noun_aliases:
                expanded[entity_type].update(proper_noun_aliases[entity])

            # 数字扩展（如 500万 → 500万元）
            if entity_type == "number":
                if "万" in entity and "元" not in entity:
                    expanded[entity_type].add(entity.replace("万", "万元"))
                elif "亿" in entity and "元" not in entity:
                    expanded[entity_type].add(entity.replace("亿", "亿元"))

    return expanded


def filter_chunks_by_entities(
    chunks: List[RetrievedChunk],
    expanded_entities: Dict[str, Set[str]],
    min_entity_matches: int = 1,
) -> List[RetrievedChunk]:
    """
    根据实体过滤文档：只保留包含关键实体的文档。
    """
    if not expanded_entities:
        return chunks

    # 合并所有实体
    all_entities = set()
    for entities in expanded_entities.values():
        all_entities.update(entities)

    if not all_entities:
        return chunks

    filtered_chunks: List[RetrievedChunk] = []
    for chunk in chunks:
        content = chunk.content.lower()
        source = chunk.source.lower()
        combined = content + "\n" + source

        # 计算匹配的实体数量
        entity_matches = 0
        matched_entities: Set[str] = set()
        for entity in all_entities:
            if entity.lower() in combined:
                entity_matches += 1
                matched_entities.add(entity.lower())

        # 至少匹配 min_entity_matches 个实体
        if entity_matches >= min_entity_matches:
            filtered_chunks.append(chunk)

    # 如果过滤后为空，保留原始文档（避免无结果）
    return filtered_chunks or chunks


def score_chunks_by_entities(
    chunks: List[RetrievedChunk],
    expanded_entities: Dict[str, Set[str]],
) -> List[RetrievedChunk]:
    """
    根据实体匹配数量对文档进行加权评分。
    """
    if not expanded_entities:
        return chunks

    # 合并所有实体
    all_entities = set()
    for entities in expanded_entities.values():
        all_entities.update(entities)

    if not all_entities:
        return chunks

    # 为每个文档计算实体匹配分数
    scored_chunks: List[tuple[float, RetrievedChunk]] = []
    for i, chunk in enumerate(chunks):
        content = chunk.content.lower()
        source = chunk.source.lower()
        combined = content + "\n" + source

        # 计算匹配的实体数量
        entity_matches = 0
        matched_entities: Set[str] = set()
        for entity in all_entities:
            if entity.lower() in combined:
                entity_matches += 1
                matched_entities.add(entity.lower())

        # 分数 = 实体匹配数量（可以加权重）
        # 更高的实体匹配数 = 更高的分数
        score = entity_matches

        scored_chunks.append((score, chunk))

    # 按分数降序排序，保持同分数的相对顺序
    scored_chunks.sort(key=lambda x: (-x[0], x[1]))

    return [chunk for _, chunk in scored_chunks]


def entity_aware_hybrid_retrieve(
    query: str,
    vectorstore: Optional[PGVector] = None,
    graph: Optional[nx.DiGraph] = None,
    vector_top_k: int = 8,
    keyword_top_k: int = 8,
    graph_max: int = 8,
    min_entity_matches: int = 1,
) -> List[RetrievedChunk]:
    """
    实体感知的混合检索：
    1. 提取问题中的关键实体
    2. 扩展实体（同义词、别名）
    3. 执行混合检索（扩大召回）
    4. 实体过滤（只保留包含实体的文档）
    5. 实体加权（按实体匹配数重排）
    6. 融合图谱结果
    """
    import networkx as nx

    # 1. 提取实体
    entities = extract_entities_from_question(query)

    # 2. 扩展实体
    expanded_entities = expand_entities(entities)

    # 3. 执行混合检索（扩大召回）
    mult = 4  # 扩大4倍召回
    vk = max(1, vector_top_k * mult)
    kk = max(1, keyword_top_k * mult)

    q_ret = expand_query_for_hybrid(query)

    vector_chunks: List[RetrievedChunk] = []
    if vectorstore is not None:
        vector_chunks = vector_search(vectorstore, q_ret, top_k=vk)

    keyword_chunks = keyword_search(q_ret, top_k=kk)
    bm25_chunks: List[RetrievedChunk] = []
    if BM25_ENABLED:
        bm_k = max(1, BM25_TOP_K * mult)
        bm25_chunks = bm25_search(q_ret, top_k=bm_k)

    # RRF 融合
    doc_chunks = fuse_doc_chunks_rrf(vector_chunks, keyword_chunks, bm25_chunks, k=RRF_K)

    # 4. 实体过滤
    doc_chunks = filter_chunks_by_entities(doc_chunks, expanded_entities, min_entity_matches)

    # 5. 实体加权
    doc_chunks = score_chunks_by_entities(doc_chunks, expanded_entities)

    # 6. 锚点加权
    doc_chunks = boost_chunks_by_query_anchors(doc_chunks, query)

    # 7. 重排序（如果启用）
    if RERANK_ENABLED and doc_chunks:
        from .reranker import rerank_doc_chunks
        cap_default = max(1, vector_top_k + keyword_top_k + (BM25_TOP_K if BM25_ENABLED else 0))
        cap = min(cap_default, RERANK_DOC_CAP) if (RERANK_DOC_CAP and RERANK_DOC_CAP > 0) else cap_default
        doc_chunks = rerank_doc_chunks(query, doc_chunks, top_n=cap)

    # 8. 截断到目标数量
    doc_chunks = doc_chunks[:vector_top_k + keyword_top_k]

    # 9. 图谱检索
    graph_chunks: List[RetrievedChunk] = []
    if graph is not None:
        graph_chunks = graph_search(graph, query, max_neighbors=graph_max)

    # 10. 合并去重
    seen_content: set[str] = {c.content for c in doc_chunks}
    merged: List[RetrievedChunk] = list(doc_chunks)
    for c in graph_chunks:
        if c.content not in seen_content:
            seen_content.add(c.content)
            merged.append(c)

    return merged


# 类型注解
from typing import Any