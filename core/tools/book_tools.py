"""书籍知识库 Agent 工具

工具职责：
- book_search: 走 BookRagWorkflow（规范化/判定/澄清）后检索合成
- simple_book_search: 最简流程，仅向量检索 + 合成，无任何 query 预处理
- list_books: 返回当前知识库已入库书籍清单

book_search / simple_book_search 的检索范围都仅由用户手选的 scope 硬约束，否则全库。

工具内通过 api.source_context 把检索到的 source nodes 写入请求级容器，
请求结束时由 chat handler 统一取出回传给前端。
"""
from llama_index.core import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from core.agent.source_context import (
    add_sources, node_to_source_ref, can_clarify, consume_clarify, get_scope,
)
from core.workflow.book_rag import BookRagWorkflow, ClarifyResult


def _scope_filters(book_titles):
    """把用户选定范围转成 metadata 过滤器；空范围返回 None（全库）。"""
    if not book_titles:
        return None
    return MetadataFilters(filters=[
        MetadataFilter(
            key="book_title",
            operator=FilterOperator.IN,
            value=list(book_titles),
        ),
    ])


def create_book_search_tool(
    index_manager,
    llm: LLM,
    similarity_top_k: int = 5,
) -> FunctionTool:
    """书籍内容检索工具

    持有 index_manager 引用而非 index 对象本身——这样书籍入库后能立刻
    被工具检索到（不需要重启 Agent）。
    """

    async def book_search(query: str) -> str:
        """从书籍知识库中检索内容并合成答案。

        检索范围由用户在前端手选的 scope 硬约束决定，未选则跨全部书检索；
        Agent 无需也无法指定书名（按书名猜书会误伤正确书籍，交给向量检索）。

        Args:
            query: 用户问题，必须是字符串。

        Returns:
            基于检索片段合成的答案文本。
        """
        # 防御：LLM 偶尔返回 dict
        if not isinstance(query, str):
            query = (query.get("title") or query.get("text") or str(query)) if isinstance(query, dict) else str(query)
        query = query.strip()
        if not query:
            return "请提供要查询的问题"

        index = index_manager.get_index()
        if index is None:
            return "知识库为空，请先在「文档管理」上传 PDF。"

        workflow = BookRagWorkflow(
            index_manager=index_manager,
            llm=llm,
            similarity_top_k=similarity_top_k,
        )
        # 查询范围：仅由用户手动选定的 scope 决定（硬约束）；未选则全库检索。
        effective_books = get_scope() or None

        result = await workflow.run(
            query=query, book_titles=effective_books,
            allow_clarify=can_clarify(),  # 预算耗尽时 workflow 会降级为检索，不再澄清
        )
        if isinstance(result, ClarifyResult):
            consume_clarify()
            return (
                f"[需要澄清] 无法直接检索：{result.clarify_reason}。"
                f"请先根据对话上下文判断用户真实意图、补全问题后重新调用本工具检索；"
                f"若上下文不足以判断，请向用户提问澄清，不要凭空假设。"
            )

        if not result.source_nodes:
            scope = f"《{'》《'.join(effective_books)}》中" if effective_books else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。"
        add_sources([node_to_source_ref(n) for n in result.source_nodes])
        return str(result)

    return FunctionTool.from_defaults(
        fn=book_search,
        name="book_search",
        description=(
            "书籍知识库检索：根据用户问题在已入库的技术书籍中查找相关内容。"
            "检索范围由用户选定，无需指定书名。"
            "回答用户对书籍内容的具体技术问题时调用此工具。"
        ),
    )


def create_simple_book_search(
    index_manager,
    llm: LLM,
    similarity_top_k: int = 5,
) -> FunctionTool:
    """最简书籍检索工具：仅做向量检索 + 答案合成。

    与 create_book_search_tool 的区别：不经过 BookRagWorkflow 的规范化/降噪/
    明确性判定/澄清/路由，直接用原始 query 检索再合成。适合做对照基线，
    或对 query 预处理无需求的场景。检索范围与 book_search 一致，仅由 scope 决定。
    """

    async def simple_book_search(query: str) -> str:
        """从书籍知识库中检索内容并合成答案（最简流程，无 query 预处理）。

        检索范围由用户在前端手选的 scope 硬约束决定，未选则跨全部书检索。

        Args:
            query: 用户问题，必须是字符串。

        Returns:
            基于检索片段合成的答案文本。
        """
        # 防御：LLM 偶尔返回 dict
        if not isinstance(query, str):
            query = (query.get("title") or query.get("text") or str(query)) if isinstance(query, dict) else str(query)
        query = query.strip()
        if not query:
            return "请提供要查询的问题"

        index = index_manager.get_index()
        if index is None:
            return "知识库为空，请先在「文档管理」上传 PDF。"

        effective_books = get_scope() or None

        retriever = index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=_scope_filters(effective_books),
        )
        nodes = await retriever.aretrieve(query)
        if not nodes:
            scope = f"《{'》《'.join(effective_books)}》中" if effective_books else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。"

        synthesizer = get_response_synthesizer(llm=llm)
        response = await synthesizer.asynthesize(query=query, nodes=nodes)
        add_sources([node_to_source_ref(n) for n in nodes])
        return str(response)

    return FunctionTool.from_defaults(
        fn=simple_book_search,
        name="simple_book_search",
        description=(
            "书籍知识库检索（最简）：根据用户问题在已入库的技术书籍中直接检索并合成答案。"
            "检索范围由用户选定，无需指定书名。"
            "回答用户对书籍内容的具体技术问题时调用此工具。"
        ),
    )


def create_list_books_tool(index_manager) -> FunctionTool:
    """列出已入库书籍工具"""

    def list_books() -> str:
        """列出当前知识库中已入库的所有书籍名称。

        Returns:
            书名清单字符串（每本一行，附块数）。
        """
        all_data = index_manager.chroma_collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in all_data.get("metadatas", []) or []:
            title = (meta or {}).get("book_title")
            if not title:
                continue
            counts[title] = counts.get(title, 0) + 1

        if not counts:
            return "知识库当前为空，请先在「文档管理」上传 PDF。"

        lines = [f"- 《{t}》（{c} 个向量块）" for t, c in sorted(counts.items())]
        return "已入库书籍：\n" + "\n".join(lines)

    return FunctionTool.from_defaults(
        fn=list_books,
        name="list_books",
        description=(
            "查询当前知识库中已入库的书籍清单。"
            "用户问'你有哪些书'、'知识库里有什么'，或问题模糊需要先了解可用书籍时调用。"
        ),
    )
