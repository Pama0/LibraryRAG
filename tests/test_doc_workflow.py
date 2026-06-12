"""DocQueryWorkflow 顶层编排接线测试：门口 Router → 按 intent dispatch → 委托 QA。

聚焦【编排】，不测 QA 检索/合成实质（那在 test_qa_capability.py）：
- Router 在门口跑，clean_query 是 QA 预处理真正消费的输入（不被二次消指代）。
- intent=study_plan → 占位分支短路，不进 QA 预处理 / 检索。
- intent=qa → QA 分类 → dispatch 到对应分支，分支委托 wf.qa.* 并收成 FinalizeEvent。
- missing_info → 反问，不检索；finalize 写回会话记忆。

QA 实质（检索+合成）整体 stub 掉（替换 wf.qa.retrieve），只验证编排把
clean/降噪后的 query + scope 正确喂进去、最终结果正确回流。
"""
from core.workflow.doc_workflow import DocQueryWorkflow


class _Resp:
    def __init__(self, text: str):
        self._t = text

    def __str__(self) -> str:
        return self._t


class FakeLLM:
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

    def put(self, m):
        self._msgs.append(m)


def _wf(llm, index_manager=None):
    return DocQueryWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3, timeout=10)


# ── 全链路 dispatch 接线 ──────────────────────────────────────────────
async def test_study_plan_intent_short_circuits_without_qa_preprocess():
    llm = FakeLLM(['{"intent": "study_plan", "clean_query": "为Redis制定学习计划"}'])
    wf = _wf(llm)
    result = await wf.run(query="给我做份学Redis的计划", memory=FakeMemory())
    assert llm.calls == 1                       # 只有 Router 这一次
    assert "学习计划" in str(result.response)


async def test_qa_intent_feeds_clean_query_and_scope_to_answer():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "MySQL索引有哪些"}',
        '{"category": "retrievable", "rewritten_query": "MySQL索引有哪些"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        captured["book_titles"] = book_titles
        return "答案", ["n1"]

    wf.qa.retrieve = fake_retrieve

    mem = FakeMemory([_Msg("user", "MySQL索引"), _Msg("assistant", "B+树……")])
    result = await wf.run(query="它有哪些", memory=mem, book_titles=["高性能MySQL"])

    assert captured["query"] == "MySQL索引有哪些"     # clean/降噪后，不是原始"它有哪些"
    assert captured["book_titles"] == ["高性能MySQL"]  # scope 透传到检索
    assert str(result.response) == "答案"
    assert result.source_nodes == ["n1"]


async def test_route_passes_selected_books_to_router():
    # 用户选中的书 scope 要喂给门口 Router，用于把"这本书"补全
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "《openclaw》讲了什么"}',
        '{"category": "retrievable", "rewritten_query": "《openclaw》讲了什么"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="这本书讲了什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]   # Router prompt 带上了选中的书


async def test_qa_preprocess_consumes_clean_query_not_original():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "MySQL索引有哪些"}',
        '{"category": "retrievable", "rewritten_query": "MySQL索引有哪些"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="它有哪些", memory=FakeMemory())
    assert "MySQL索引有哪些" in llm.prompts[1]
    assert "它有哪些" not in llm.prompts[1]


async def test_router_parse_failure_defaults_to_qa_path():
    llm = FakeLLM([
        "这不是JSON",
        '{"category": "retrievable", "rewritten_query": "B+树索引"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        return "答案", []

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="B+树索引", memory=FakeMemory())
    assert llm.calls == 2
    assert captured["query"] == "B+树索引"


async def test_missing_info_clarifies_without_retrieval():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明"}',
    ])
    wf = _wf(llm)

    called = {"retrieve": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert called["retrieve"] is False            # 反问，不检索
    assert "指代不明" in str(result.response)


# ── missing_info：自然反问 / 预算耗尽降级声明假设 ──────────────────────
async def test_missing_info_uses_natural_clarify_question():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明", "clarify_question": "你说的「这个索引」指哪一个？B+树还是全文索引？"}',
    ])
    wf = _wf(llm)
    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert "你说的「这个索引」指哪一个" in str(result.response)


async def test_other_category_answers_via_dedicated_branch():
    # other 不再与 retrievable/解析失败混走 fallback，而是独立分支（v1 暂仍单轮检索）
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "设计一个支持千万级并发的发号器"}',
        '{"category": "other", "rewritten_query": "设计一个支持千万级并发的发号器", "reason": "开放设计题"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        return "复杂问题答案", ["n1"]

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="设计一个支持千万级并发的发号器", memory=FakeMemory())
    assert captured["query"] == "设计一个支持千万级并发的发号器"
    assert str(result.response) == "复杂问题答案"
    assert result.source_nodes == ["n1"]


async def test_missing_info_budget_exhausted_assumes_and_answers():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        captured["preamble"] = preamble
        return preamble + "尽力答", ["n1"]

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(
        query="这个索引的应用场景", memory=FakeMemory(), allow_clarify=False
    )
    assert "按最可能的解读作答" in captured["preamble"]   # 声明假设
    assert "尽力答" in str(result.response)
    assert result.source_nodes == ["n1"]                   # 确实检索了（未反问）


async def test_other_dispatches_to_bounded_agent():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "对比 openclaw 的两种架构取舍"}',
        '{"category": "other", "rewritten_query": "对比 openclaw 的两种架构取舍", "reason": "开放权衡"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_agent_run(ctx, query, book_titles):
        captured["query"] = query
        captured["book_titles"] = book_titles
        return "agent 综合答案", ["n1", "n2"]

    wf.qa_agent.run = fake_agent_run

    result = await wf.run(query="对比 openclaw 的两种架构取舍", memory=FakeMemory(), book_titles=["openclaw"])
    assert captured["query"] == "对比 openclaw 的两种架构取舍"
    assert captured["book_titles"] == ["openclaw"]
    assert str(result.response) == "agent 综合答案"
    assert result.source_nodes == ["n1", "n2"]


async def test_other_falls_back_to_single_retrieve_when_agent_raises(caplog):
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "设计题"}',
        '{"category": "other", "rewritten_query": "设计题", "reason": "开放设计"}',
    ])
    wf = _wf(llm)

    async def boom(ctx, query, book_titles):
        raise RuntimeError("agent 失败")

    wf.qa_agent.run = boom

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "降级单轮答案", ["n1"]

    wf.qa.retrieve = fake_retrieve

    import logging
    with caplog.at_level(logging.WARNING):
        result = await wf.run(query="设计题", memory=FakeMemory())
    assert str(result.response) == "降级单轮答案"   # agent 抛错 → 降级 qa.retrieve
    assert result.source_nodes == ["n1"]
    assert any("other agent 失败" in r.getMessage() for r in caplog.records)  # 降级显形


async def test_preprocess_passes_book_titles_to_classify():
    llm = FakeLLM(['{"intent": "qa", "clean_query": "openclaw是什么"}'])  # 仅 Router 调 LLM
    wf = _wf(llm)

    captured = {}

    async def fake_classify(clean_query, book_titles=None, probe=True):
        captured["clean"] = clean_query
        captured["books"] = book_titles
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    wf.qa.classify = fake_classify

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="openclaw是什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert captured["clean"] == "openclaw是什么"
    assert captured["books"] == ["openclaw"]   # scope 透传到 probe


async def test_flags_off_degrade_branches_to_single_retrieve():
    # split flag 关 → pending_split 走单轮 retrieve（baseline 对比用）
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "讲讲MySQL"}',
        '{"category": "pending_split", "rewritten_query": "讲讲MySQL", "reason": "需罗列"}',
    ])
    wf = DocQueryWorkflow(
        index_manager=None, llm=llm, similarity_top_k=3, timeout=10,
        split_enabled=False, assume_enabled=False, other_agent_enabled=False,
        probe_then_classify=False,
    )
    used = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        used["retrieve"] = True
        return "单轮答案", ["n1"]

    async def boom_split(ctx, query, book_titles):
        raise AssertionError("split 不应被调用（flag off）")

    wf.qa.retrieve = fake_retrieve
    wf.qa.split = boom_split
    result = await wf.run(query="讲讲MySQL", memory=FakeMemory())
    assert used.get("retrieve") is True
    assert str(result.response) == "单轮答案"


async def test_finalize_exposes_category_in_metadata():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "MySQL锁"}',
        '{"category": "retrievable", "rewritten_query": "MySQL锁"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", ["n1"]

    wf.qa.retrieve = fake_retrieve
    result = await wf.run(query="MySQL锁", memory=FakeMemory())
    assert result.metadata.get("category") == "retrievable"
    assert result.metadata.get("intent") == "qa"
