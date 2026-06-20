"""AnswerOutliner（据宽召回列概念骨架）单测。mock LLM 控返回，验解析/降级。"""
from core.workflow.answer_outliner import AnswerOutliner


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_outline_multi_node():
    llm = FakeLLM(['{"sub_queries":["MySQL索引基础","MySQL事务基础","MySQL锁基础"]}'])
    subs = await AnswerOutliner(llm).run("MySQL基础知识", ["片段1", "片段2"])
    assert subs == ["MySQL索引基础", "MySQL事务基础", "MySQL锁基础"]


async def test_outline_atomic_single_node():
    llm = FakeLLM(['{"sub_queries":["脏读的定义与例子"]}'])  # 原子概念 1 节
    subs = await AnswerOutliner(llm).run("什么是脏读", ["片段"])
    assert subs == ["脏读的定义与例子"]


async def test_outline_passages_passed_to_prompt():
    llm = FakeLLM(['{"sub_queries":["x"]}'])
    await AnswerOutliner(llm).run("讲讲X", ["关键片段ABC"])
    assert "关键片段ABC" in llm.prompts[0]


async def test_outline_respects_max_items():
    llm = FakeLLM(['{"sub_queries":["a","b","c","d"]}'])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"], max_items=2)
    assert subs == ["a", "b"]


async def test_outline_empty_on_parse_failure():
    llm = FakeLLM(["这不是JSON"])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert subs == []          # 空 → explain 将落 agent 兜底


async def test_outline_empty_on_empty_list():
    llm = FakeLLM(['{"sub_queries":[]}'])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert subs == []
