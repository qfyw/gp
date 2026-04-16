# 基于 RAG + 知识图谱的智能问答系统 MVP - Streamlit 入口
from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

# 国内网络：在任何 HuggingFace 相关库加载前设置镜像，否则会连 huggingface.co 超时/被拒
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 保证项目根在路径中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx

from src.data_loader import process_uploaded_files, process_web_urls, build_or_load_graph, save_graph
from src.vectorstore import load_vectorstore
from src.retriever import hybrid_retrieve
from src.generator import GenerationResult
from src.agents.workflow import run_advanced_workflow
from src.doc_store import load_docs, remove_doc_by_name
from src.pg_db import get_conn
from src.config import (
    DOC_ONLY_MODE,
    KB_STRICT_ONLY,
    KB_NAMESPACE,
    PGVECTOR_COLLECTION,
    RETRIEVAL_GRAPH_MAX,
    RETRIEVAL_KEYWORD_TOP_K,
    RETRIEVAL_VECTOR_TOP_K,
)

# 页面配置
st.set_page_config(page_title="RAG+KG 智能问答", page_icon="📚", layout="wide")

# 会话状态：打开页面后即连接数据库并加载图谱与嵌入（见下方 bootstrap_kb_if_needed）
if "messages" not in st.session_state:
    st.session_state.messages = []

_KB_GRAPH_READY = "_kb_graph_ready"
_KB_VS_READY = "_kb_vs_ready"


def is_uncertain_answer(text: str) -> bool:
    """判断是否为拒答/不确定回答（容忍标点与前后缀差异）。"""
    t = (text or "").strip()
    if not t:
        return False
    compact = "".join(ch for ch in t if ch not in " \t\r\n`'\"。.!！?？：:")
    return ("无法确定" in compact) or ("不确定" in compact and "无法" in compact)


def ensure_graph() -> None:
    """仅加载本地图谱 pickle，不触发嵌入模型；用于预览与入库合并。"""
    if st.session_state.get(_KB_GRAPH_READY):
        return
    try:
        st.session_state.graph = build_or_load_graph()
    except Exception as e:
        st.session_state.graph = nx.DiGraph()
        st.session_state["_kb_graph_load_error"] = str(e)
    st.session_state[_KB_GRAPH_READY] = True


def ensure_vectorstore(*, show_spinner: bool = True) -> None:
    """连接 PostgreSQL / pgvector 并加载 HuggingFace 嵌入（首次最慢，可能下载模型）。"""
    if st.session_state.get(_KB_VS_READY):
        return
    spin = (
        st.spinner("正在连接数据库并加载嵌入模型（首次启动可能较慢，请稍候）…")
        if show_spinner
        else nullcontext()
    )
    with spin:
        try:
            st.session_state.vectorstore = load_vectorstore()
            st.session_state.pop("_kb_vs_load_error", None)
        except Exception as e:
            st.session_state.vectorstore = None
            st.session_state["_kb_vs_load_error"] = str(e)
    st.session_state[_KB_VS_READY] = True


def ensure_stores_for_rag() -> None:
    """上传、检索、删除等需要向量库 + 图谱在内存中的一致视图。"""
    ensure_graph()
    ensure_vectorstore()


def bootstrap_kb_if_needed() -> None:
    """每个会话首次进入页面时：立即加载本地图谱 + 连接数据库并加载嵌入。"""
    if st.session_state.get(_KB_GRAPH_READY) and st.session_state.get(_KB_VS_READY):
        return
    with st.spinner("正在连接数据库并加载知识图谱与嵌入模型（首次可能较慢）…"):
        ensure_graph()
        ensure_vectorstore(show_spinner=False)


bootstrap_kb_if_needed()


def render_kg_preview() -> None:
    """右侧栏：知识图谱可视化预览（抽样部分节点与边）。"""
    ensure_graph()
    graph = st.session_state.graph
    if graph is None or graph.number_of_nodes() == 0:
        st.info("当前暂无知识图谱数据。请先上传文档或网页。")
        return
    try:
        from pyvis.network import Network

        # 仅抽取最多 30 个节点，避免图太大
        nodes = list(graph.nodes())[:30]
        sub_g = graph.subgraph(nodes)

        net = Network(height="400px", width="100%", directed=True, notebook=False, bgcolor="#ffffff")
        for n, data in sub_g.nodes(data=True):
            label = str(data.get("label", n))
            net.add_node(str(n), label=label)
        for u, v, data in sub_g.edges(data=True):
            if u in sub_g and v in sub_g:
                rel = data.get("relation", "")
                net.add_edge(str(u), str(v), label=rel)

        html = net.generate_html(notebook=False)
        components.html(html, height=420, scrolling=True)
    except Exception as e:
        st.warning(f"知识图谱可视化失败：{e}")


def on_upload(files):
    """处理上传：解析、向量化、写入图谱。"""
    if not files:
        return
    ensure_stores_for_rag()
    file_list = []
    for f in files:
        file_list.append((f.read(), f.name))
    with st.spinner("正在解析文档、切片、向量化并构建知识图谱..."):
        try:
            vs, g, ingest = process_uploaded_files(
                file_list,
                existing_vectorstore=st.session_state.vectorstore,
                existing_graph=st.session_state.graph,
            )
            st.session_state.vectorstore = vs
            st.session_state.graph = g
            if ingest.chunk_count == 0:
                st.warning(
                    "未能从上传文件中提取到任何可索引文本，因此向量库与关键词检索中不会有内容。\n\n"
                    "**常见原因**：PDF 为**扫描版/图片版**（整页是图，没有文字层）。"
                    "当前解析仅支持可选中复制的文字 PDF，**不会对图片做 OCR**。\n\n"
                    "**可行做法**：用 Adobe Acrobat、ABBYY、WPS 等对 PDF **OCR**，"
                    "或导出为 **Word / 可复制文字的 PDF** 后再上传；或自备该书的电子版文本。"
                )
            else:
                st.success(
                    f"已处理 {len(files)} 个文件，共写入 {ingest.chunk_count} 条文本块，可开始提问。"
                )
            if ingest.files_no_extractable_text:
                st.warning(
                    "以下文件未解析出文字（若整批都无内容，多为扫描版 PDF）："
                    + "、".join(ingest.files_no_extractable_text)
                )
        except Exception as e:
            st.error(f"处理失败: {e}")


def delete_docs_from_stores(names: list[str]) -> tuple[int, list[str]]:
    """
    从向量库、关键词表与 docs_index 中删除指定文档/URL。
    返回 (成功删除条数, 错误信息列表)。
    """
    ensure_stores_for_rag()
    vs = st.session_state.vectorstore
    g = st.session_state.graph
    ok = 0
    errors: list[str] = []
    for selected in names:
        if not (selected or "").strip():
            continue
        try:
            if vs is not None:
                try:
                    if hasattr(vs, "delete"):
                        vs.delete(filter={"filename": selected})
                        vs.delete(filter={"url": selected})
                except Exception:
                    pass
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # 1) 向量库“硬删除”：直接按 collection + metadata(filename/url) 删 embedding 行
                    try:
                        col = (PGVECTOR_COLLECTION or "").strip()
                        if col:
                            cur.execute(
                                "SELECT uuid FROM langchain_pg_collection WHERE name = %s;",
                                (col,),
                            )
                            row = cur.fetchone()
                            if row:
                                cid = row[0]
                                cur.execute(
                                    """
                                    DELETE FROM langchain_pg_embedding
                                    WHERE collection_id = %s
                                      AND (
                                        cmetadata->>'filename' = %s
                                        OR cmetadata->>'url' = %s
                                      );
                                    """,
                                    (cid, selected, selected),
                                )
                    except Exception:
                        # 不阻断：仍继续清理关键词/索引/图谱
                        pass

                    # 2) 关键词表删除：按 namespace 隔离
                    cur.execute(
                        "DELETE FROM rag_keyword_chunks WHERE namespace = %s AND (filename = %s OR url = %s);",
                        (KB_NAMESPACE, selected, selected),
                    )
            # 从知识图谱中删除该文档/URL 产生的边，并清理孤立节点
            try:
                if g is not None and g.number_of_nodes() > 0:
                    to_drop = []
                    for u, v, data in g.edges(data=True):
                        srcs = data.get("sources")
                        hit = False
                        if isinstance(srcs, (list, set, tuple)):
                            hit = selected in srcs
                        elif isinstance(srcs, str):
                            hit = (srcs == selected)
                        if hit:
                            to_drop.append((u, v))
                    if to_drop:
                        g.remove_edges_from(to_drop)
                        # 清理孤立节点
                        isolated = [n for n in list(g.nodes()) if g.degree(n) == 0]
                        if isolated:
                            g.remove_nodes_from(isolated)
                        save_graph(g)
                        st.session_state.graph = g
            except Exception:
                # 图谱删除为尽力而为，不阻断整体删除流程
                pass
            remove_doc_by_name(selected)
            ok += 1
        except Exception as e:
            errors.append(f"{selected}: {e}")
    return ok, errors


# ---------- 侧边栏：文件上传 ----------
with st.sidebar:
    st.header("知识库管理")
    st.caption("上传 PDF 或 Word 文档，系统将自动解析、向量化并抽取知识图谱。")
    uploaded = st.file_uploader(
        "选择 PDF / Word 文档",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
    )
    if st.button("上传并入库", type="primary") and uploaded:
        on_upload(uploaded)

    st.divider()
    st.caption("或添加网页 URL（每行一个）")
    url_text = st.text_area("网页 URL 列表", placeholder="https://example.com/article\nhttps://example.com/page")
    if st.button("抓取网页并入库"):
        urls = [u.strip() for u in (url_text or "").splitlines() if u.strip()]
        if not urls:
            st.warning("请先输入至少一个有效 URL。")
        else:
            ensure_stores_for_rag()
            with st.spinner("正在抓取网页、抽取文本、切片、向量化并构建知识图谱..."):
                try:
                    vs, g2 = process_web_urls(
                        urls,
                        existing_vectorstore=st.session_state.vectorstore,
                        existing_graph=st.session_state.graph,
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.graph = g2
                    st.success(f"已处理 {len(urls)} 个 URL，可开始提问。")
                except Exception as e:
                    st.error(f"处理失败: {e}")

    st.divider()
    st.caption("当前状态：")
    if st.session_state.get(_KB_VS_READY, False):
        has_vs = st.session_state.vectorstore is not None
        st.write("- 向量库: " + ("已加载" if has_vs else "未配置或连接失败"))
        err_vs = st.session_state.get("_kb_vs_load_error")
        if err_vs:
            st.caption(f"向量库异常：{err_vs}")
    else:
        st.write("- 向量库: 未完成初始化（请刷新页面重试）")
    if st.session_state.get(_KB_GRAPH_READY, False):
        gr = st.session_state.graph
        has_nodes = gr is not None and gr.number_of_nodes() > 0
        st.write("- 知识图谱: " + ("有数据" if has_nodes else "已加载（当前无节点）"))
        err_g = st.session_state.get("_kb_graph_load_error")
        if err_g:
            st.caption(f"图谱加载异常：{err_g}")
    else:
        st.write("- 知识图谱: 未完成初始化（请刷新页面重试）")

    st.divider()
    st.subheader("内部知识库")
    docs = load_docs()
    if not docs:
        st.caption("尚未有入库文档或 URL。")
    else:
        names = [d.name for d in docs]
        st.caption("可多选后批量删除；向量库删除为尽力而为，关键词表与索引会同步清理。")
        ms_key = "kb_batch_delete_names"
        reset_flag_key = "kb_batch_delete_names__reset"
        # 若上次操作要求重置选择，需要在 multiselect 实例化之前处理
        if st.session_state.get(reset_flag_key):
            st.session_state.pop(ms_key, None)
            st.session_state[reset_flag_key] = False
        if ms_key not in st.session_state:
            st.session_state[ms_key] = []

        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            if st.button("全选", use_container_width=True):
                st.session_state[ms_key] = list(names)
                st.rerun()
        with c_sel2:
            if st.button("清空选择", use_container_width=True):
                st.session_state[ms_key] = []
                st.rerun()

        to_delete = st.multiselect(
            "选择要删除的文档/URL（可多选）",
            options=names,
            key=ms_key,
        )

        if st.button("批量删除选中项", type="primary", disabled=not to_delete):
            if st.session_state.vectorstore is None:
                st.warning("当前尚未加载向量库，仍将尝试清理关键词表与文档索引。")
            with st.spinner(f"正在删除 {len(to_delete)} 项…"):
                n_ok, errs = delete_docs_from_stores(to_delete)
            if errs:
                st.error("部分删除失败：\n" + "\n".join(errs))
            if n_ok:
                st.success(f"已成功删除 {n_ok} 项。")
                # 删除后强制刷新：重新加载向量库句柄与图谱文件，并清理旧消息中的溯源缓存
                st.session_state.vectorstore = load_vectorstore()
                st.session_state.graph = build_or_load_graph()
                st.session_state.messages = []
                # 不能在同一轮渲染里直接修改已实例化的 multiselect key，改为设置重置标志并 rerun
                st.session_state[reset_flag_key] = True
                st.rerun()


# ---------- 主界面：三栏布局（中间聊天 + 右侧信息栏） ----------
st.title("RAG + 知识图谱 智能问答")
st.caption("基于多源信息融合与知识图谱增强的问答原型")

main_col, right_col = st.columns([2.2, 1.0])

with main_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            show_sources = bool(msg.get("sources"))
            if KB_STRICT_ONLY and is_uncertain_answer(msg.get("content", "")):
                show_sources = False
            if show_sources:
                with st.expander("溯源信息"):
                    for s in msg["sources"]:
                        st.caption(f"**{s['source']}**")
                        st.text(s.get("content", "")[:300] + ("..." if len(s.get("content", "")) > 300 else ""))

    if prompt := st.chat_input("输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            ensure_stores_for_rag()
            vs = st.session_state.vectorstore
            g = st.session_state.graph
            graph_for_qa = None if DOC_ONLY_MODE else g
            if vs is None and (g is None or g.number_of_nodes() == 0):
                reply = "请先在侧边栏上传 PDF 或 Word 文档 / 网页 URL 并入库，再提问。"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
            else:
                # st.status 需包住耗时步骤，否则长时间停在「空白」；检索常比工作流更慢（尤其首次加载 CrossEncoder）。
                advanced_state = None
                with st.status("🧭 处理中…", expanded=True) as status:
                    status.write("① 混合检索：向量 + 关键词 + 图谱（若开启重排，首次会加载 CrossEncoder，可能较慢）…")
                    chunks = hybrid_retrieve(
                        prompt,
                        vectorstore=vs,
                        graph=graph_for_qa,
                        vector_top_k=RETRIEVAL_VECTOR_TOP_K,
                        keyword_top_k=RETRIEVAL_KEYWORD_TOP_K,
                        graph_max=RETRIEVAL_GRAPH_MAX,
                    )
                    if not chunks:
                        status.update(label="未检索到片段", state="error")
                        reply = "未检索到相关参考资料，请确认已上传文档或换一种问法。"
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
                    else:
                        status.write(f"② 检索完成，共 {len(chunks)} 条片段；启动多智能体工作流…")
                        # traces 在工作流同步结束后才返回，中间无法逐条刷新（除非改用 LangGraph stream）
                        advanced_state = run_advanced_workflow(prompt, chunks, graph=graph_for_qa)
                        for t in (advanced_state.get("traces") or []):
                            status.write(t)
                        status.update(label="✅ 已完成", state="complete")

                if advanced_state is not None:
                    final_answer = (advanced_state.get("final_answer", "") or "").strip()
                    internal_docs = list(advanced_state.get("internal_docs") or [])
                    # 严格知识库模式下，若判定“无法确定”，不展示候选召回片段，避免误导为“有效溯源”。
                    if KB_STRICT_ONLY and is_uncertain_answer(final_answer):
                        internal_docs = []

                    def _stream_text(txt: str):
                        for part in txt.splitlines(True):
                            yield part

                    st.write_stream(_stream_text(final_answer))

                    with st.expander("溯源信息"):
                        if internal_docs:
                            st.subheader("内部文档")
                            for s in internal_docs[:10]:
                                st.caption(f"**{s.get('source','内部文档')}**")
                                st.text((s.get("content", "") or "")[:400] + ("..." if len((s.get("content", "") or "")) > 400 else ""))
                        if advanced_state.get("kg_triples"):
                            st.subheader("图谱推导（2-hop）")
                            for p in (advanced_state.get("kg_triples") or [])[:10]:
                                st.text(p.get("path", ""))

                    result = GenerationResult(
                        answer=final_answer,
                        sources=[
                            {"source": d.get("source", ""), "content": d.get("content", "")}
                            for d in internal_docs
                        ],
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )

with right_col:
    st.subheader("知识图谱预览")
    render_kg_preview()
