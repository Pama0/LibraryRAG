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


# ── converse + list_books 元工具路径（Task 3）──────────────────────────


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _FakeIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _FakeCollection(metas)


def _agent_with_lib(llm, metas):
    """带 index_manager 的 FrontDoorAgent（元工具路径需要）。"""
    return FrontDoorAgent(llm, index_manager=_FakeIndexManager(metas))


async def test_converse_list_books_full_invokes_tool_and_composes_reply():
    # "库里都有什么" → 1st 判 converse+tool=list_books → 查库 → 2nd 组回复
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":"","reason":"元查询"}',
        '已入库的有《高性能MySQL》和《Redis》两本。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("现在库里都有什么书")
    assert d.action == "converse"
    assert "高性能MySQL" in d.reply       # 2nd 组的回复含书名
    assert llm.calls == 2                  # 1st 决策 + 2nd 组回复
    # 2nd prompt 含工具结果 + 原 query
    assert "已入库书籍" in llm.prompts[1] or "《高性能MySQL》" in llm.prompts[1]
    assert "现在库里都有什么书" in llm.prompts[1]


async def test_converse_list_books_filter_passes_filter_to_tool():
    # "有 MySQL 的书吗" → tool_filter="mysql" → 工具结果只含 MySQL 书
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_filter":"mysql","reply":""}',
        '有 MySQL 相关的书：《高性能MySQL》。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("有 MySQL 的书吗")
    assert d.action == "converse"
    assert "高性能MySQL" in d.reply
    # 2nd prompt 的工具结果不含 Redis（被 filter 过滤）
    assert "Redis" not in llm.prompts[1]
    assert "匹配「mysql」" in llm.prompts[1]


async def test_converse_list_books_count_only_returns_count():
    # "多少本" → tool_count_only=true → 工具结果只回计数
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_count_only":true,"reply":""}',
        '目前库里一共有 2 本书。',
    ])
    metas = [{"book_title": "甲"}, {"book_title": "乙"}]
    d = await _agent_with_lib(llm, metas).run("现在有多少本书")
    assert d.action == "converse"
    assert "2" in d.reply
    assert "已入库 2 本" in llm.prompts[1]   # 工具结果是计数，不是列表


async def test_converse_no_tool_uses_reply_directly_no_2nd_call():
    # 纯寒暄 → tool="" → reply 直接用，不调 2nd
    llm = FakeLLM(['{"action":"converse","tool":"","reply":"你好！我是文档知识库助手～"}'])
    d = await _agent_with_lib(llm, []).run("你好")
    assert d.action == "converse"
    assert "知识库助手" in d.reply
    assert llm.calls == 1                    # 只 1st，无 2nd


async def test_converse_tool_with_filter_and_count_only():
    # "有 mysql 吗，几本" → filter + count_only 同时带
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_filter":"mysql","tool_count_only":true,"reply":""}',
        '有 1 本匹配 MySQL 的书。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("有 mysql 吗，几本")
    assert d.action == "converse"
    assert "匹配「mysql」的书有 1 本" in llm.prompts[1]


async def test_converse_tool_compose_failure_degrades_to_raw_tool_result():
    # 2nd LLM 抛错 → 降级裸 tool_result 当 reply
    class _BoomLLM:
        def __init__(self):
            self.calls = 0
            self.prompts = []
        async def acomplete(self, prompt, **kw):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                return _Resp('{"action":"converse","tool":"list_books","reply":""}')
            raise RuntimeError("2nd 炸了")
    llm = _BoomLLM()
    metas = [{"book_title": "甲"}]
    d = await _agent_with_lib(llm, metas).run("库里有什么")
    assert d.action == "converse"
    assert "已入库书籍" in d.reply and "《甲》" in d.reply   # 裸 tool_result


async def test_converse_tool_compose_empty_degrades_to_raw_tool_result():
    # 2nd LLM 返回空 → 降级裸 tool_result
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":""}',
        "",
    ])
    metas = [{"book_title": "甲"}]
    d = await _agent_with_lib(llm, metas).run("库里有什么")
    assert d.action == "converse"
    assert "已入库书籍" in d.reply and "《甲》" in d.reply


async def test_converse_tool_list_books_failure_degrades_to_placeholder():
    # list_books_text 抛错 → 占位文本进 2nd LLM
    class _BrokenCollection:
        def get(self, include=None):
            raise RuntimeError("chroma 挂了")
    class _BrokenIM:
        chroma_collection = _BrokenCollection()
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":""}',
        '抱歉，我没能读取库藏清单。',
    ])
    agent = FrontDoorAgent(llm, index_manager=_BrokenIM())
    d = await agent.run("库里有什么")
    assert d.action == "converse"
    assert "未能读取库藏清单" in llm.prompts[1]   # 占位文本进了 2nd prompt
    assert "抱歉" in d.reply


async def test_dispatch_qa_ignores_tool_field():
    # dispatch_qa 即使 LLM 误填 tool，也不走工具路径
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL锁","tool":"list_books"}'])
    d = await _agent_with_lib(llm, []).run("MySQL有哪些锁")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "MySQL锁"
    assert llm.calls == 1                    # 不调 2nd


async def test_front_door_prompt_guards_proper_nouns():
    """规范化不得把库内专名当形近错字改写（openclaw → OpenCL 回归防护）。

    专名保护是 prompt 层的铁律——LLM 行为本身需真 LLM 验，单测只断言铁律进了 prompt。
    """
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲讲openclaw"}'])
    await _agent(llm).run("讲讲openclaw")
    p = llm.prompts[0]
    assert "专名" in p              # 专名保护铁律进 prompt
    assert "OpenCL" in p            # openclaw→OpenCL 形近误改反例写进 prompt


async def test_front_door_sets_disable_scope_on_all_books_request():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway","disable_scope":true}'])
    d = await _agent(llm).run("在所有书里讲一下gateway")
    assert d.action == "dispatch_qa"
    assert d.disable_scope is True


async def test_front_door_disable_scope_defaults_false():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    d = await _agent(llm).run("讲一下gateway")
    assert d.disable_scope is False


async def test_front_door_prompt_has_tool_definition_and_redline():
    llm = FakeLLM(['{"action":"converse","tool":"","reply":"hi"}'])
    await _agent_with_lib(llm, []).run("你好")
    p = llm.prompts[0]
    assert "list_books" in p                 # 工具定义进 prompt
    assert "tool_filter" in p
    assert "tool_count_only" in p
    # 红线：内容问题一律 dispatch_qa
    assert "dispatch_qa" in p
