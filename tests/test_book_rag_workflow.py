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
