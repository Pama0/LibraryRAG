"""QuerySplitter 单测：mock LLM 控返回，验证拆分解析 / 单问题透传 / 降级。

拆分质量（多主体 vs 比较 vs 多跳的边界判断）依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
from core.workflow.query_splitter import QuerySplitter


class _Resp:
    def __init__(self, text): self._t = text
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_splits_independent_multi_subject():
    llm = FakeLLM(['{"sub_queries":["MySQL的锁有哪些","Redis的持久化机制"]}'])
    subs = await QuerySplitter(llm).run("讲讲MySQL的锁和Redis的持久化")
    assert subs == ["MySQL的锁有哪些", "Redis的持久化机制"]


async def test_single_question_returns_one_element():
    llm = FakeLLM(['{"sub_queries":["什么是聚簇索引"]}'])
    subs = await QuerySplitter(llm).run("什么是聚簇索引啊")
    assert subs == ["什么是聚簇索引"]


async def test_empty_sub_queries_degrades_to_original():
    llm = FakeLLM(['{"sub_queries":[]}'])
    subs = await QuerySplitter(llm).run("讲讲MySQL")
    assert subs == ["讲讲MySQL"]


async def test_parse_failure_degrades_to_original(caplog):
    import logging
    llm = FakeLLM(["这不是JSON"])
    with caplog.at_level(logging.WARNING):
        subs = await QuerySplitter(llm).run("讲讲MySQL")
    assert subs == ["讲讲MySQL"]
    assert any("splitter" in r.getMessage().lower() for r in caplog.records)


async def test_query_in_prompt():
    llm = FakeLLM(['{"sub_queries":["x"]}'])
    await QuerySplitter(llm).run("讲讲MySQL和Redis的区别")
    assert "讲讲MySQL和Redis的区别" in llm.prompts[0]
