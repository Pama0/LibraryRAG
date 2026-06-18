"""book_tools 单测：两个工具类的方法 + 注册表工厂 + ctx 状态收集。"""
import pytest

from core.agent.tools import BookSearchTool, ListBooksTool
from core.agent.tools.book_tools import (
    ToolContext,
    ToolSpec,
    _TOOL_REGISTRY,
    assemble_tools,
    build_book_tools,
    register_tool,
)


class FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    async def aretrieve(self, query):
        return self._nodes


class FakeIndex:
    def __init__(self, nodes):
        self._nodes = nodes
        self.last_kw = None

    def as_retriever(self, **kw):
        self.last_kw = kw
        return FakeRetriever(self._nodes)


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class FakeIndexManager:
    def __init__(self, nodes, metas=None):
        self._index = FakeIndex(nodes)
        self.chroma_collection = _FakeCollection(metas or [])

    def get_index(self):
        return self._index


class _Node:
    def __init__(self, content):
        self._c = content

    def get_content(self):
        return self._c


def _ctx(nodes=(), metas=None, top_k=3):
    im = FakeIndexManager(nodes=list(nodes), metas=metas)
    return ToolContext(index_manager=im, similarity_top_k=top_k)


async def test_book_search_joins_passages_and_collects_sources():
    ctx = _ctx(nodes=[_Node("片段A"), _Node("片段B")])
    out = await BookSearchTool(ctx)("分布式事务")
    assert "片段A" in out and "片段B" in out
    assert len(ctx.sources) == 2


async def test_book_search_empty_returns_placeholder_no_collect():
    ctx = _ctx(nodes=[])
    out = await BookSearchTool(ctx)("不存在")
    assert out == "（未检索到相关内容）"
    assert ctx.sources == []


async def test_book_search_blank_query_prompts():
    ctx = _ctx(nodes=[_Node("x")])
    assert await BookSearchTool(ctx)("   ") == "请提供要检索的问题。"


async def test_book_search_passes_scope_and_top_k_to_retriever():
    ctx = _ctx(nodes=[_Node("x")], top_k=3)
    ctx.scope = ["书A"]
    await BookSearchTool(ctx)("q")
    kw = ctx.index_manager.get_index().last_kw
    assert kw["similarity_top_k"] == 3
    assert kw["filters"] is not None  # build_book_filters(["书A"]) 非空


def test_list_books_counts_titles():
    ctx = _ctx(metas=[{"book_title": "甲"}, {"book_title": "甲"}, {"book_title": "乙"}])
    out = ListBooksTool(ctx)()
    assert "《甲》（2 块）" in out
    assert "《乙》（1 块）" in out


def test_list_books_empty():
    ctx = _ctx(metas=[])
    assert ListBooksTool(ctx)() == "知识库当前为空。"


def test_build_book_tools_default_returns_both():
    tools = build_book_tools(_ctx())
    names = sorted(t.metadata.name for t in tools)
    assert names == ["book_search", "list_books"]


def test_build_book_tools_unknown_name_raises():
    with pytest.raises(ValueError):
        build_book_tools(_ctx(), ["nope"])


class _RecordingRetriever:
    """记录 retrieve 入参的假 Retriever，返回预置候选池。"""

    def __init__(self, nodes):
        self._nodes = nodes
        self.calls = []

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.calls.append(dict(query=query, book_titles=book_titles, top_k=top_k))
        return list(self._nodes)


class _RecordingReranker:
    """记录 rerank 入参的假 Reranker，截前 top_n。"""

    def __init__(self):
        self.calls = []

    async def rerank(self, query, nodes, top_n):
        self.calls.append(dict(query=query, top_n=top_n, n_in=len(nodes)))
        return nodes[:top_n]


async def test_book_search_overfetches_then_reranks_to_top_k():
    pool = [_Node(f"片段{i}") for i in range(20)]
    retr, rer = _RecordingRetriever(pool), _RecordingReranker()
    ctx = _ctx(nodes=[])  # index_manager 只需 get_index() 非 None
    ctx.retriever = retr
    ctx.reranker = rer
    ctx.similarity_top_k = 5
    ctx.rerank_candidate_k = 20
    out = await BookSearchTool(ctx)("q")
    assert retr.calls[0]["top_k"] == 20   # 有 reranker → 过召回到候选池
    assert rer.calls[0]["top_n"] == 5     # 重排截断到最终 top_k
    assert len(ctx.sources) == 5
    assert "片段0" in out


async def test_book_search_no_reranker_fetches_top_k_only():
    pool = [_Node("a"), _Node("b")]
    retr = _RecordingRetriever(pool)
    ctx = _ctx(nodes=[])
    ctx.retriever = retr
    ctx.reranker = None
    ctx.similarity_top_k = 3
    await BookSearchTool(ctx)("q")
    assert retr.calls[0]["top_k"] == 3    # 无 reranker → 不过召回
    assert len(ctx.sources) == 2


def test_assemble_tools_default_returns_both_and_numbered_prompt():
    tools, prompt = assemble_tools(_ctx())
    assert sorted(t.metadata.name for t in tools) == ["book_search", "list_books"]
    assert "1. " in prompt and "2. " in prompt
    assert "book_search(query)" in prompt
    assert "list_books()" in prompt


def test_assemble_tools_subset_only_selected():
    tools, prompt = assemble_tools(_ctx(), ["book_search"])
    assert [t.metadata.name for t in tools] == ["book_search"]
    assert "book_search(query)" in prompt
    assert "list_books" not in prompt


def test_assemble_tools_usage_override_replaces_default():
    _, prompt = assemble_tools(_ctx(), [ToolSpec("book_search", usage="自定义X")])
    assert "自定义X" in prompt
    assert "book_search(query)" not in prompt


def test_assemble_tools_unknown_name_raises():
    with pytest.raises(ValueError):
        assemble_tools(_ctx(), ["nope"])


def test_assemble_tools_falls_back_to_description_when_no_prompt_usage():
    @register_tool
    class _TmpTool:
        name = "_tmp_tool"
        description = "临时工具描述"

        def __init__(self, ctx):
            self.ctx = ctx

        def __call__(self) -> str:
            return ""

        def to_function_tool(self):
            from llama_index.core.tools import FunctionTool
            return FunctionTool.from_defaults(
                fn=self.__call__, name=self.name, description=self.description
            )

    try:
        _, prompt = assemble_tools(_ctx(), ["_tmp_tool"])
        assert "临时工具描述" in prompt
    finally:
        _TOOL_REGISTRY.pop("_tmp_tool", None)
