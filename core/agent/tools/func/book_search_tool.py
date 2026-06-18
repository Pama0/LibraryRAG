from llama_index.core.tools import FunctionTool

from core.agent.tools import register_tool, ToolContext


def _unwrap(n):
    """NodeWithScore → 内层 node；裸 node 原样返回。"""
    return getattr(n, "node", n)


def _node_id(n) -> str:
    """稳定标识：node_id / id_ / 退化到对象 id（fake/无 id 节点不会误并）。"""
    node = _unwrap(n)
    return getattr(node, "node_id", None) or getattr(node, "id_", None) or str(id(n))


def _node_text(n) -> str:
    return n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")


def _source_prefix(n) -> str:
    """据 metadata 拼出处前缀：【《书名》· 章节 · p.x-y】，缺字段则省略对应段。"""
    meta = getattr(_unwrap(n), "metadata", None) or {}
    parts = [f"《{meta.get('book_title') or '未知来源'}》"]
    chapter = meta.get("chapter")
    if chapter:
        parts.append(str(chapter))
    ps, pe = meta.get("page_start"), meta.get("page_end")
    if ps and pe:
        parts.append(f"p.{ps}" if ps == pe else f"p.{ps}-{pe}")
    elif meta.get("page"):
        parts.append(f"p.{meta['page']}")
    return "【" + " · ".join(parts) + "】"


@register_tool
class BookSearchTool:
    """书籍知识库检索：经 ctx 的可插拔 retriever/reranker 取片段并把命中 nodes 收进 ctx.sources。

    检索走 ctx.retriever（默认向量基线，agent 可注入 hybrid 等）；ctx.reranker 非空时先
    过召回 rerank_candidate_k 个候选再重排截断到 similarity_top_k——与 qa_capability 同套。
    """

    name = "book_search"
    description = "书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。"
    prompt_usage = "book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def __call__(self, query: str) -> str:
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return "请提供要检索的问题。"
        if self.ctx.index_manager.get_index() is None:
            return "知识库为空，请先上传 PDF。"
        reranker = self.ctx.reranker
        fetch_k = self.ctx.rerank_candidate_k if reranker else self.ctx.similarity_top_k
        nodes = await self.ctx.retriever.retrieve(
            query, index_manager=self.ctx.index_manager,
            book_titles=self.ctx.scope, top_k=fetch_k,
        )
        if reranker:
            nodes = await reranker.rerank(query, nodes, self.ctx.similarity_top_k)
        if not nodes:
            return "（未检索到相关内容）"
        # 收集去重（按 node_id 保序保首次）：多轮检索常重叠命中，避免回传/计数虚高。
        seen = {_node_id(s) for s in self.ctx.sources}
        for n in nodes:
            nid = _node_id(n)
            if nid not in seen:
                seen.add(nid)
                self.ctx.sources.append(n)
        # 带出处前缀、不再按字符截断（chunk 入库时已限好粒度），便于 grounding 引用。
        return "\n---\n".join(
            f"{_source_prefix(n)}\n{_node_text(n)}" for n in nodes
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )