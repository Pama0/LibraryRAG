from llama_index.core.tools import FunctionTool

from core.agent.tools import register_tool, ToolContext
from core.retrieval.retrieve import build_book_filters


@register_tool
class BookSearchTool:
    """书籍知识库检索：按 query 取 top-k 原文片段并把命中 nodes 收进 ctx.sources。"""

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
        index = self.ctx.index_manager.get_index()
        if index is None:
            return "知识库为空，请先上传 PDF。"
        retriever = index.as_retriever(
            similarity_top_k=self.ctx.similarity_top_k,
            filters=build_book_filters(self.ctx.scope),
        )
        nodes = await retriever.aretrieve(query)
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