"""QaAgent 单测：检索工具（_search）+ run 流式桥接。

FunctionAgent 是真实组件，不在单测范围——用 MockLLM 让其构造通过，run 测试
用 fake agent 替身（产真 ToolCall/ToolCallResult 事件 + 可 await 的 final）。
"""
from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult
from llama_index.core.llms import MockLLM
from llama_index.core.tools import ToolOutput

from core.agent.qa_agent import QaAgent


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


class FakeCtx:
    def __init__(self):
        self.events = []

    def write_event_to_stream(self, ev):
        self.events.append(ev)


class _FakeHandler:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def stream_events(self):
        for e in self._events:
            yield e

    def __await__(self):
        async def _f():
            return self._final
        return _f().__await__()


class FakeAgent:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.last_kwargs = None

    def run(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeHandler(self._events, self._final)


def _agent(index_manager=None):
    return QaAgent(index_manager, MockLLM(), similarity_top_k=3, max_iterations=6)


async def test_search_returns_joined_passages_and_collects_nodes():
    qa = _agent(FakeIndexManager(nodes=[_Node("片段A"), _Node("片段B")]))
    qa._run_scope = None
    qa._run_sources = []
    out = await qa._search("分布式事务")
    assert "片段A" in out and "片段B" in out
    assert len(qa._run_sources) == 2


async def test_search_empty_returns_placeholder_and_collects_nothing():
    qa = _agent(FakeIndexManager(nodes=[]))
    qa._run_scope = None
    qa._run_sources = []
    out = await qa._search("不存在")
    assert out == "（未检索到相关内容）"
    assert qa._run_sources == []


async def test_run_bridges_tool_events_and_emits_final_delta():
    qa = _agent(FakeIndexManager(nodes=[]))
    events = [
        ToolCall(tool_name="book_search", tool_kwargs={"query": "子问题1"}, tool_id="1"),
        ToolCallResult(
            tool_name="book_search",
            tool_kwargs={"query": "子问题1"},
            tool_id="1",
            tool_output=ToolOutput(
                content="片段", tool_name="book_search", raw_input={}, raw_output="片段"
            ),
            return_direct=False,
        ),
    ]
    qa.agent = FakeAgent(events, final="综合答案")
    ctx = FakeCtx()

    answer, nodes = await qa.run(ctx, "openclaw 的整体架构与权衡", None)

    assert answer == "综合答案"
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalStartEvent") == 1
    assert names.count("RetrievalDoneEvent") == 1
    starts = [e for e in ctx.events if e.__class__.__name__ == "RetrievalStartEvent"]
    assert starts[0].query == "子问题1"
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert "综合答案" in deltas


async def test_run_resets_sources_each_call_and_passes_max_iterations():
    qa = _agent(FakeIndexManager(nodes=[]))
    qa._run_sources = ["stale"]
    qa.agent = FakeAgent([], final="答案")
    ctx = FakeCtx()

    answer, nodes = await qa.run(ctx, "q", ["书A"])
    assert nodes == []
    assert qa.agent.last_kwargs.get("max_iterations") == 6
    assert qa.agent.last_kwargs.get("user_msg") == "q"
