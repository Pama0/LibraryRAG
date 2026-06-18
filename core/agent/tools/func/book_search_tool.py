from llama_index.core.tools import FunctionTool

from core.agent.tools import register_tool, ToolContext


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
        self.ctx.sources.extend(nodes)
        return "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:500]
            for n in nodes
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )