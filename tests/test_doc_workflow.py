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
    wf = DocQueryWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3, timeout=10)

    async def _echo_other_gate(clean_query):
        return clean_query, "other"            # 默认非 explain；explain 测试自行覆盖

    wf.qa.gate = _echo_other_gate
    return wf


# ── 全链路 dispatch 接线 ──────────────────────────────────────────────
async def test_study_plan_intent_short_circuits_without_qa_preprocess():
    llm = FakeLLM(['{"action": "dispatch_study_plan", "clean_query": "为Redis制定学习计划"}'])
    wf = _wf(llm)
    result = await wf.run(query="给我做份学Redis的计划", memory=FakeMemory())
    assert llm.calls == 1                       # 只有 Router 这一次
    assert "学习计划" in str(result.response)


async def test_qa_intent_feeds_clean_query_and_scope_to_answer():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL索引有哪些"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        captured["book_titles"] = book_titles
        return "答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")
    wf.qa.classify = fake_classify

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
        '{"action": "dispatch_qa", "clean_query": "《openclaw》讲了什么"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="这本书讲了什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]   # Router prompt 带上了选中的书


async def test_qa_preprocess_consumes_clean_query_not_original():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL索引有哪些"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        captured["clean"] = clean_query
        return PreprocessResult("retrievable")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="它有哪些", memory=FakeMemory())
    assert captured["clean"] == "MySQL索引有哪些"
    assert "它有哪些" not in captured["clean"]


async def test_router_parse_failure_defaults_to_qa_path():
    llm = FakeLLM([
        "这不是JSON",
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="B+树索引", memory=FakeMemory())
    assert llm.calls == 1
    assert captured["query"] == "B+树索引"


async def test_missing_info_clarifies_without_retrieval():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}',
    ])
    wf = _wf(llm)

    called = {"retrieve": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("missing_info", reason="指代不明")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert called["retrieve"] is False            # 反问，不检索
    assert "指代不明" in str(result.response)


# ── missing_info：自然反问 / 预算耗尽降级声明假设 ──────────────────────
async def test_missing_info_uses_natural_clarify_question():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}',
    ])
    wf = _wf(llm)

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult(
            "missing_info",
            clarify_question="你说的「这个索引」指哪一个？B+树还是全文索引？",
        )
    wf.qa.classify = fake_classify

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert "你说的「这个索引」指哪一个" in str(result.response)


async def test_other_category_answers_via_dedicated_branch():
    # other 不再与 retrievable/解析失败混走 fallback，而是独立分支（v1 暂仍单轮检索）
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "设计一个支持千万级并发的发号器"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        return "复杂问题答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("other", reason="开放设计题")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="设计一个支持千万级并发的发号器", memory=FakeMemory())
    assert captured["query"] == "设计一个支持千万级并发的发号器"
    assert str(result.response) == "复杂问题答案"
    assert result.source_nodes == ["n1"]


async def test_missing_info_budget_exhausted_assumes_and_answers():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        captured["preamble"] = preamble
        return preamble + "尽力答", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("missing_info", reason="指代不明")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(
        query="这个索引的应用场景", memory=FakeMemory(), allow_clarify=False
    )
    assert "按最可能的解读作答" in captured["preamble"]   # 声明假设
    assert "尽力答" in str(result.response)
    assert result.source_nodes == ["n1"]                   # 确实检索了（未反问）


async def test_other_dispatches_to_bounded_agent():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "对比 openclaw 的两种架构取舍"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_agent_run(ctx, query, book_titles):
        captured["query"] = query
        captured["book_titles"] = book_titles
        return "agent 综合答案", ["n1", "n2"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("other", reason="开放权衡")
    wf.qa.classify = fake_classify

    wf.qa_agent.run = fake_agent_run

    result = await wf.run(query="对比 openclaw 的两种架构取舍", memory=FakeMemory(), book_titles=["openclaw"])
    assert captured["query"] == "对比 openclaw 的两种架构取舍"
    assert captured["book_titles"] == ["openclaw"]
    assert str(result.response) == "agent 综合答案"
    assert result.source_nodes == ["n1", "n2"]


async def test_other_falls_back_to_single_retrieve_when_agent_raises(caplog):
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "设计题"}',
    ])
    wf = _wf(llm)

    async def boom(ctx, query, book_titles):
        raise RuntimeError("agent 失败")

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("other", reason="开放设计")
    wf.qa.classify = fake_classify

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
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "openclaw是什么"}'])  # 仅 Router 调 LLM
    wf = _wf(llm)

    captured = {}

    async def fake_classify(clean_query, book_titles=None, probe=True):
        captured["clean"] = clean_query
        captured["books"] = book_titles
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

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
        '{"action": "dispatch_qa", "clean_query": "讲讲MySQL"}',
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

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("pending_split", reason="需罗列")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve
    wf.qa.split = boom_split
    result = await wf.run(query="讲讲MySQL", memory=FakeMemory())
    assert used.get("retrieve") is True
    assert str(result.response) == "单轮答案"


async def test_finalize_exposes_category_in_metadata():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL锁"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve
    result = await wf.run(query="MySQL锁", memory=FakeMemory())
    assert result.metadata.get("category") == "retrievable"
    assert result.metadata.get("action") == "dispatch_qa"


async def test_converse_responds_without_retrieval_or_classify():
    # "你好" → action=converse → 门口直接回复，不进 preprocess/检索
    llm = FakeLLM(['{"action": "converse", "reply": "你好！我是文档知识库助手～"}'])
    wf = _wf(llm)
    called = {"retrieve": False, "classify": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        called["classify"] = True
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="你好", memory=FakeMemory())
    assert called["retrieve"] is False
    assert called["classify"] is False           # converse 门口拦截，不进 QA
    assert "知识库助手" in str(result.response)
    assert llm.calls == 1                         # 只有门口这一次


async def test_out_of_scope_responds_without_retrieval_or_clarify():
    # 库外问题（PostgreSQL）→ out_of_scope → 固定话术，不检索/不反问
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "PostgreSQL的MVCC是怎么实现的"}',
    ])
    wf = _wf(llm)

    called = {"retrieve": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("out_of_scope", reason="库外，召回片段均不相关")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="PostgreSQL的MVCC是怎么实现的", memory=FakeMemory())
    assert called["retrieve"] is False                 # 库外不检索
    # 对话式转场：友好告知库外 + 邀请换个问法（不再机械精确话术）
    resp = str(result.response)
    assert "知识库" in resp and ("暂" in resp or "没有" in resp or "未收录" in resp)
    assert result.source_nodes == []
    assert result.metadata.get("category") == "out_of_scope"  # 分类回流 metadata


async def test_explain_intent_routes_to_explain_branch():
    # front_door dispatch_qa → preprocess → gate intent=explain → explain_branch（不进难度分类）
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "讲讲MVCC"}'])  # 仅 front_door 调 LLM
    wf = _wf(llm)
    called = {"explain": False, "classify": False}

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        called["explain"] = True
        assert query == "讲讲MVCC"               # rewritten_query 来自 gate
        return "教学体答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        called["classify"] = True
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲讲MVCC", memory=FakeMemory())
    assert called["explain"] is True
    assert called["classify"] is False          # explain 跳过难度分类
    assert str(result.response) == "教学体答案"
    assert result.source_nodes == ["n1"]
    assert llm.calls == 1                        # 只 front_door（gate/explain 都 stub）


# ── reranker 名字 → 对象注入 QaCapability（装配单测）──────────────────
class _StubIndexManager:
    pass


class _StubLLM:
    pass


def test_no_reranker_by_default():
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert wf.qa.reranker is None


def test_reranker_name_resolved_and_injected(monkeypatch):
    sentinel = object()
    import core.workflow.doc_workflow as mod

    names = []

    def fake_make(name):
        names.append(name)   # 答案 + probe 各调一次
        return sentinel

    monkeypatch.setattr(mod, "make_reranker", fake_make)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(),
                          reranker="bge-reranker-v2-m3")

    assert "bge-reranker-v2-m3" in names
    assert wf.qa.reranker is sentinel


# ── retriever 名字 → 对象注入 QaCapability（装配单测）─────────────────
def test_default_retriever_is_vector():
    from core.retrieval.retrieve import VectorRetriever

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert isinstance(wf.qa.retriever, VectorRetriever)


def test_retriever_name_resolved_and_injected(monkeypatch):
    sentinel = object()
    import core.workflow.doc_workflow as mod

    names = []

    def fake_make(name):
        names.append(name)   # 答案 + probe 各调一次
        return sentinel

    monkeypatch.setattr(mod, "make_retriever", fake_make)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(), retriever="hybrid")

    assert "hybrid" in names
    assert wf.qa.retriever is sentinel


# ── probe 检索解耦：独立 probe_retriever / probe_reranker 注入（装配单测）──
def test_default_probe_uses_vector_retriever_and_no_reranker():
    from core.retrieval.retrieve import VectorRetriever

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert isinstance(wf.qa.probe_retriever, VectorRetriever)
    assert wf.qa.probe_reranker is None


def test_probe_retriever_name_resolved_and_injected(monkeypatch):
    import core.workflow.doc_workflow as mod

    sentinels = {}

    def fake_make(name):
        # 同名复用一个 sentinel：retriever 与 explain_retriever 都会解析 "hybrid"，
        # 按名 memoize 才能让 wf.qa.retriever is sentinels["hybrid"] 成立。
        return sentinels.setdefault(name, object())

    monkeypatch.setattr(mod, "make_retriever", fake_make)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(),
                          retriever="hybrid", probe_retriever="vector")

    assert wf.qa.probe_retriever is sentinels["vector"]
    assert wf.qa.retriever is sentinels["hybrid"]            # 答案侧仍独立解析
    assert wf.qa.explain_retriever is sentinels["hybrid"]   # explain 宽召回默认 hybrid


def test_probe_reranker_name_resolved_and_injected(monkeypatch):
    import core.workflow.doc_workflow as mod

    sentinels = {}

    def fake_make(name):
        s = object()
        sentinels[name] = s
        return s

    monkeypatch.setattr(mod, "make_reranker", fake_make)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(),
                          probe_reranker="bge-reranker-v2-m3")

    assert wf.qa.probe_reranker is sentinels["bge-reranker-v2-m3"]


# ── explain_branch：catch OutOfScope/MissingInfo + EmptySkeleton 仍落 agent（Task 6）──
from core.workflow.qa_capability import (
    EmptySkeleton, OutOfScope, MissingInfo, REFUSAL_TEXT, REFUSAL_FALLBACK,
)


async def test_explain_out_of_scope_refuses_with_category():
    # explain admit 判库外 → 抛 OutOfScope → explain_branch 拒答 + category=out_of_scope
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "PostgreSQL的MVCC"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise OutOfScope(query)

    async def fake_agent(ctx, query, book_titles):
        raise AssertionError("agent 不应被调用（库外应直接拒答）")

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("retrieve 不应被调用")

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent
    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="PostgreSQL的MVCC", memory=FakeMemory())
    assert str(result.response) == REFUSAL_TEXT
    assert result.source_nodes == []
    assert result.metadata.get("category") == "out_of_scope"
    assert result.metadata.get("intent") == "explain"


async def test_explain_missing_info_clarifies_with_category():
    # explain admit 判信息不足 → 抛 MissingInfo(反问) → explain_branch 反问 + category=missing_info
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise MissingInfo("你说的「这个索引」指哪一个？B+树还是全文索引？")

    async def fake_agent(ctx, query, book_titles):
        raise AssertionError("agent 不应被调用（信息不足应直接反问）")

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("retrieve 不应被调用")

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent
    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert str(result.response) == "你说的「这个索引」指哪一个？B+树还是全文索引？"
    assert result.source_nodes == []
    assert result.metadata.get("category") == "missing_info"


async def test_explain_missing_info_without_clarify_uses_fallback():
    # MissingInfo 缺 clarify_question → 用 REFUSAL_FALLBACK 兜底反问
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "这个索引"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise MissingInfo("")                      # 缺反问句

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain

    result = await wf.run(query="这个索引", memory=FakeMemory())
    assert str(result.response) == REFUSAL_FALLBACK
    assert result.metadata.get("category") == "missing_info"


async def test_explain_empty_skeleton_still_falls_to_agent():
    # 回归：EmptySkeleton 不被 OutOfScope/MissingInfo catch 截胡，仍落 agent 兜底
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "讲讲X"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise EmptySkeleton(query)

    agent_called = {"v": False}

    async def fake_agent(ctx, query, book_titles):
        agent_called["v"] = True
        return "agent 兜底答案", ["n1"]

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent

    result = await wf.run(query="讲讲X", memory=FakeMemory())
    assert agent_called["v"] is True
    assert str(result.response) == "agent 兜底答案"
    assert result.source_nodes == ["n1"]


async def test_out_of_scope_branch_uses_refusal_text_constant():
    # 回归：other 路库外分支话术 = REFUSAL_TEXT（单一来源，不另写一句）
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "MongoDB分片"}'])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("库外不应检索")

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("out_of_scope", reason="库外")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="MongoDB分片", memory=FakeMemory())
    assert str(result.response) == REFUSAL_TEXT
    assert result.metadata.get("category") == "out_of_scope"


# ── front_door converse + list_books 端到端（Task 4）────────────────────


class _LibCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _LibIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _LibCollection(metas)
        self._metas = metas

    def get_index(self):
        class _Idx:
            def as_retriever(self, **kw):
                raise AssertionError("元查询不应检索")
        return _Idx()


async def test_library_listing_routes_to_converse_tool_without_retrieval():
    # "现在库里都有什么" → front_door converse+list_books → reply 含书名、不检索
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    im = _LibIndexManager(metas)
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":"","reason":"元查询"}',
        '已入库的有《高性能MySQL》和《Redis》两本。',
    ])
    wf = DocQueryWorkflow(index_manager=im, llm=llm, similarity_top_k=3, timeout=10)

    retrieve_called = {"v": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        retrieve_called["v"] = True
        return "不应被调用", []

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="现在库里都有什么书", memory=FakeMemory())
    assert retrieve_called["v"] is False            # 元查询不检索
    assert "高性能MySQL" in str(result.response)
    assert "Redis" in str(result.response)
    assert result.source_nodes == []
    assert result.metadata.get("action") == "converse"
    assert llm.calls == 2                            # 1st 决策 + 2nd 组回复
    # 2nd prompt 的 {data} 被真实 tool_result 替换（含书名），非占位符
    assert "已入库书籍" in llm.prompts[1]
    assert "高性能MySQL" in llm.prompts[1]
    assert "未能读取库藏清单" not in llm.prompts[1]


async def test_library_count_question_routes_to_converse_tool_count_only():
    # "多少本" → converse+list_books+count_only → reply 含计数、不检索
    metas = [{"book_title": "甲"}, {"book_title": "乙"}, {"book_title": "丙"}]
    im = _LibIndexManager(metas)
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_count_only":true,"reply":""}',
        '目前库里一共有 3 本书。',
    ])
    wf = DocQueryWorkflow(index_manager=im, llm=llm, similarity_top_k=3, timeout=10)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("元查询不应检索")

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="现在有多少本书", memory=FakeMemory())
    assert "3" in str(result.response)
    assert result.metadata.get("action") == "converse"
    # 2nd prompt 的 {data} 被真实计数 tool_result 替换，非占位符
    assert "已入库 3 本" in llm.prompts[1]
    assert "未能读取库藏清单" not in llm.prompts[1]


# ── ConversationScoper 接线（Task 2）──────────────────────────────────
def test_scoper_constructed_with_probe_vector_retriever():
    from core.workflow.conversation_scoper import ConversationScoper
    from core.retrieval.retrieve import VectorRetriever
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert isinstance(wf.scoper, ConversationScoper)
    assert isinstance(wf.scoper.probe_retriever, VectorRetriever)


async def test_scoper_narrows_book_titles_in_full_library():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(["openclaw"], "（我按《openclaw》回答…）\n")
    wf.scoper.run = fake_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        captured["classify_books"] = book_titles
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="讲一下gateway", memory=FakeMemory([_Msg("user", "讲讲openclaw")]))
    assert captured["book_titles"] == ["openclaw"]      # 收窄透传到检索
    assert captured["classify_books"] == ["openclaw"]   # 也透传到 classify probe


async def test_scoper_called_with_user_books_and_result_flows():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    seen = {}

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        seen["args"] = (clean_query, user_book_titles)
        return ScopeDecision(user_book_titles, "")       # 模拟手选 no-op
    wf.scoper.run = fake_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="讲一下gateway", memory=FakeMemory(), book_titles=["高性能MySQL"])
    assert seen["args"] == ("讲一下gateway", ["高性能MySQL"])
    assert captured["book_titles"] == ["高性能MySQL"]



# ── 透明声明前缀（Task 3）─────────────────────────────────────────────
async def test_scope_note_prepended_to_answer():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(["openclaw"], "（我按《openclaw》回答…）\n")
    wf.scoper.run = fake_scope

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "正文答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲一下gateway", memory=FakeMemory([_Msg("user", "讲讲openclaw")]))
    resp = str(result.response)
    assert resp.startswith("（我按《openclaw》回答")     # 声明在最前
    assert "正文答案" in resp


async def test_disable_scope_skips_scoper():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway","disable_scope":true}'])
    wf = _wf(llm)

    async def boom_scope(*a, **k):
        raise AssertionError("disable_scope=true 时不应调用 scoper")
    wf.scoper.run = boom_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="在所有书里讲一下gateway", memory=FakeMemory())
    assert captured["book_titles"] is None        # 跳过收窄，保持全库


async def test_no_scope_note_when_not_narrowed():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(None, "")                    # 不收窄
    wf.scoper.run = fake_scope

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "正文答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲一下gateway", memory=FakeMemory())
    assert str(result.response) == "正文答案"             # 无前缀
