"""book_tools 单测：两个工具类的方法 + 注册表工厂 + ctx 状态收集。"""
import pytest

from core.agent.tools.book_tools import (
    BookSearchTool,
    ListBooksTool,
    ToolContext,
    build_book_tools,
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
        build_book_tools(_ctx(), names=["nope"])
