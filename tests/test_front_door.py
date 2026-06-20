"""FrontDoorAgent（Layer 1 对话准入节点）单测。

mock LLM 控返回，验证：4 出口解析 / 净化输入透传 / 失败降级。
对话/意图判断质量依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md。
"""
from core.workflow.front_door import FrontDoorAgent, FrontDoorDecision, format_history


class _Resp:
    def __init__(self, text: str):
        self._t = text

    def __str__(self) -> str:
        return self._t


class FakeLLM:
    """按队列依次返回预设文本，并记录收到的 prompt（断言历史/scope 拼接）。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeMemory:
    def __init__(self, msgs=None):
        self._msgs = list(msgs or [])

    def get(self):
        return self._msgs


def _agent(llm):
    return FrontDoorAgent(llm)


async def test_dispatch_qa_carries_clean_query():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"什么是聚簇索引","reply":""}'])
    d = await _agent(llm).run("什么是聚簇索引啊")
    assert isinstance(d, FrontDoorDecision)
    assert d.action == "dispatch_qa"
    assert d.clean_query == "什么是聚簇索引"


async def test_dispatch_qa_resolves_coreference():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL索引的应用场景"}'])
    d = await _agent(llm).run("它的应用场景是什么")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "MySQL索引的应用场景"


async def test_dispatch_study_plan():
    llm = FakeLLM(['{"action":"dispatch_study_plan","clean_query":"为《Redis设计与实现》制定学习计划"}'])
    d = await _agent(llm).run("给我做份学Redis的计划")
    assert d.action == "dispatch_study_plan"
    assert d.clean_query == "为《Redis设计与实现》制定学习计划"


async def test_converse_carries_reply():
    llm = FakeLLM(['{"action":"converse","reply":"你好！我是文档知识库助手～"}'])
    d = await _agent(llm).run("你好")
    assert d.action == "converse"
    assert "知识库助手" in d.reply


async def test_clarify_carries_reply():
    llm = FakeLLM(['{"action":"clarify","reply":"你说的「那个」是指前面的聚簇索引还是锁？"}'])
    d = await _agent(llm).run("那个再讲讲")
    assert d.action == "clarify"
    assert "聚簇索引" in d.reply


async def test_history_passed_to_prompt():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL索引的应用场景"}'])
    memory = FakeMemory([_Msg("user", "MySQL索引有哪些"), _Msg("assistant", "B+树索引……")])
    await _agent(llm).run("它的应用场景是什么", memory)
    assert "MySQL索引有哪些" in llm.prompts[0]


async def test_selected_books_injected_into_prompt():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"《openclaw》讲了什么"}'])
    await _agent(llm).run("这本书讲了什么", None, book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]


async def test_parse_failure_degrades_to_dispatch_qa_original(caplog):
    import logging
    llm = FakeLLM(["这不是JSON"])
    with caplog.at_level(logging.WARNING):
        d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"
    assert any("front_door 解析失败" in r.getMessage() for r in caplog.records)


async def test_empty_content_degrades_to_dispatch_qa():
    llm = FakeLLM([""])
    d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"


async def test_invalid_action_degrades_to_dispatch_qa():
    llm = FakeLLM(['{"action":"do_magic","clean_query":"x"}'])
    d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"


async def test_dispatch_qa_empty_clean_query_uses_original():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":""}'])
    d = await _agent(llm).run("什么是B+树")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "什么是B+树"


async def test_converse_empty_reply_gets_fallback():
    llm = FakeLLM(['{"action":"converse","reply":""}'])
    d = await _agent(llm).run("你好")
    assert d.action == "converse"
    assert d.reply  # 非空兜底


def test_format_history_keeps_summary_head_beyond_window():
    from core.workflow.summarizer import SUMMARY_MARKER
    msgs = [_Msg("user", f"{SUMMARY_MARKER}\n远期摘要内容")]
    msgs += [_Msg("user", f"q{i}") for i in range(10)]
    out = format_history(FakeMemory(msgs), max_msgs=3)
    assert "远期摘要内容" in out
    assert "q9" in out
    assert "q0" not in out


def test_format_history_without_summary_just_tail():
    msgs = [_Msg("user", f"q{i}") for i in range(10)]
    out = format_history(FakeMemory(msgs), max_msgs=3)
    assert "q9" in out and "q0" not in out
