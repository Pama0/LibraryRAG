import pytest

from core.workflow.book_rag import BookRagWorkflow


class _Resp:
    """模拟 LLM 返回对象,str(resp) 即文本。"""
    def __init__(self, text: str):
        self._t = text
    def __str__(self) -> str:
        return self._t


class FakeLLM:
    """按队列依次返回预设文本;acomplete 是 workflow judge 唯一用到的方法。"""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
    async def acomplete(self, prompt, **kw):
        self.calls += 1
        return _Resp(self._responses.pop(0))


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


class FakeIndexManager:
    def __init__(self, nodes):
        self._index = FakeIndex(nodes)
    def get_index(self):
        return self._index


def _make_wf(llm, index_manager=None):
    return BookRagWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3)


async def test_judge_query_clear_returns_original():
    llm = FakeLLM(['{"clear": true, "rewritten_query": "B+树的索引结构"}'])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("B+树的索引结构")
    assert clear is True
    assert q == "B+树的索引结构"


async def test_judge_query_unclear_returns_rewrite():
    llm = FakeLLM(['{"clear": false, "rewritten_query": "数据库索引的实现原理"}'])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("讲讲数据库")
    assert clear is False
    assert q == "数据库索引的实现原理"


async def test_judge_query_malformed_falls_back_to_clear():
    llm = FakeLLM(["这不是JSON"])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("讲讲数据库")
    assert clear is True          # 解析失败 → 当作明确,不阻塞
    assert q == "讲讲数据库"       # 用原 query


async def test_decide_clear_routes_to_retrieve():
    llm = FakeLLM(['{"clear": true, "rewritten_query": "B+树"}'])
    wf = _make_wf(llm)
    action, q = await wf._decide("B+树", round=0)
    assert action == "retrieve"
    assert q == "B+树"


async def test_decide_clear_uses_corrected_query():
    # 带错别字但语义明确：判定 clear，但 query 被纠错，应带纠错后的进检索
    llm = FakeLLM(['{"clear": true, "rewritten_query": "Python的装饰器"}'])
    wf = _make_wf(llm)
    action, q = await wf._decide("Python的装饰起", round=0)
    assert action == "retrieve"
    assert q == "Python的装饰器"   # 用纠错后的，而非原始带错字的


async def test_decide_unclear_routes_to_rewrite():
    llm = FakeLLM(['{"clear": false, "rewritten_query": "数据库索引原理"}'])
    wf = _make_wf(llm)
    action, q = await wf._decide("讲讲数据库", round=0)
    assert action == "rewrite"
    assert q == "数据库索引原理"


async def test_decide_caps_at_max_rounds_without_calling_llm():
    llm = FakeLLM([])  # 队列为空：若被调用会 IndexError
    wf = _make_wf(llm)
    action, q = await wf._decide("还是很泛", round=2)
    assert action == "retrieve"   # 达上限直接检索
    assert q == "还是很泛"
    assert llm.calls == 0         # 未再调用 LLM


async def test_retrieve_nodes_passes_top_k_and_returns_nodes():
    llm = FakeLLM([])
    im = FakeIndexManager(nodes=["n1", "n2"])
    wf = _make_wf(llm, index_manager=im)
    nodes = await wf._retrieve_nodes("B+树", book_title=None)
    assert nodes == ["n1", "n2"]
    assert im._index.last_kw["similarity_top_k"] == 3
    assert im._index.last_kw["filters"] is None


async def test_retrieve_nodes_builds_book_title_filter():
    llm = FakeLLM([])
    im = FakeIndexManager(nodes=["n1"])
    wf = _make_wf(llm, index_manager=im)
    await wf._retrieve_nodes("B+树", book_title="MySQL是怎样运行的")
    assert im._index.last_kw["filters"] is not None
