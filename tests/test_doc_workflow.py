"""DocQueryWorkflow 顶层编排接线测试：门口 Router → route → split_answer → finalize。

聚焦【编排】，不测 QA 检索/合成/拆分实质（那在 test_qa_capability.py）：
- Router 在门口跑，clean_query 是 QA 真正消费的输入（不被二次消指代）。
- intent=study_plan → 占位分支短路，不进 QA。
- intent=qa → 委托 wf.qa.answer 统一编排，收成 FinalizeEvent。
- converse/clarify → 门口直接回复，不进 QA。

QA 实质（拆分+判定+检索+合成）整体 stub 掉（替换 wf.qa.answer），只验证编排把
clean query + scope 正确喂进去、最终结果 + meta 正确回流。
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
    return wf


# ── 全链路 dispatch 接线 ──────────────────────────────────────────────
async def test_study_plan_intent_short_circuits_without_qa():
    llm = FakeLLM(['{"action": "dispatch_study_plan", "clean_query": "为Redis制定学习计划"}'])
    wf = _wf(llm)
    result = await wf.run(query="给我做份学Redis的计划", memory=FakeMemory())
    assert llm.calls == 1                       # 只有 Router 这一次
    assert "学习计划" in str(result.response)


async def test_dispatch_qa_goes_through_split_answer():
    llm = FakeLLM([])
    wf = _wf(llm)

    async def fake_front(original, memory, bt):
        from core.workflow.front_door import FrontDoorDecision, RoutedSubQuery
        return FrontDoorDecision(
            "dispatch_qa",
            clean_query="讲讲MySQL锁和Redis持久化",
            disable_scope=True,
            sub_queries=[
                RoutedSubQuery("MySQL锁", "dispatch_qa"),
                RoutedSubQuery("Redis持久化", "dispatch_qa"),
            ],
        )
    wf.front_door.run = fake_front

    captured = {}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        captured["sub_queries"] = sub_queries
        captured["disable_scope"] = disable_scope
        return (
            "合并答案",
            ["n1"],
            {"category": "multi", "categories": ["simple", "explain"], "sub_count": 2},
        )
    wf.qa.answer = fake_answer

    result = await wf.run(query="讲讲MySQL锁和Redis持久化", memory=FakeMemory(), book_titles=None)
    assert result.response == "合并答案"
    assert result.metadata["category"] == "multi"
    assert result.source_nodes == ["n1"]
    assert len(captured["sub_queries"]) >= 1     # route 计划经 ctx 透传到 split_answer
    assert captured["disable_scope"] is True     # front_door 的 disable_scope 一并透传


async def test_qa_intent_feeds_clean_query_and_scope_to_answer():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL索引有哪些"}',
        '{"sub_queries":[{"query":"MySQL索引有哪些","action":"dispatch_qa","reply":""}]}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        captured["sub_queries"] = sub_queries
        captured["book_titles"] = bt
        return "答案", ["n1"], {"category": "simple"}
    wf.qa.answer = fake_answer

    mem = FakeMemory([_Msg("user", "MySQL索引"), _Msg("assistant", "B+树……")])
    result = await wf.run(query="它有哪些", memory=mem, book_titles=["高性能MySQL"])

    # clean/降噪后，不是原始"它有哪些"
    assert captured["sub_queries"][0].query == "MySQL索引有哪些"
    assert captured["book_titles"] == ["高性能MySQL"]  # scope 透传到 answer
    assert str(result.response) == "答案"
    assert result.source_nodes == ["n1"]


async def test_route_passes_selected_books_to_router():
    # 用户选中的书 scope 要喂给门口 Router，用于把"这本书"补全
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "《openclaw》讲了什么"}',
        '{"sub_queries":[{"query":"《openclaw》讲了什么","action":"dispatch_qa","reply":""}]}',
    ])
    wf = _wf(llm)

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        return "答案", [], {"category": "simple"}
    wf.qa.answer = fake_answer

    await wf.run(query="这本书讲了什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]   # Router prompt 带上了选中的书


async def test_qa_consumes_clean_query_not_original():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL索引有哪些"}',
        '{"sub_queries":[{"query":"MySQL索引有哪些","action":"dispatch_qa","reply":""}]}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        captured["clean"] = sub_queries[0].query
        return "答案", [], {"category": "simple"}
    wf.qa.answer = fake_answer

    await wf.run(query="它有哪些", memory=FakeMemory())
    assert captured["clean"] == "MySQL索引有哪些"
    assert "它有哪些" not in captured["clean"]


async def test_router_parse_failure_defaults_to_qa_path():
    llm = FakeLLM([
        "这不是JSON",
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        captured["sub_queries"] = sub_queries
        return "答案", [], {"category": "simple"}
    wf.qa.answer = fake_answer

    await wf.run(query="B+树索引", memory=FakeMemory())
    assert llm.calls == 1   # 门口解析失败直接降级，不会进入二次拆分调用
    assert len(captured["sub_queries"]) == 1
    assert captured["sub_queries"][0].query == "B+树索引"


async def test_finalize_exposes_qa_meta_in_metadata():
    llm = FakeLLM([
        '{"action": "dispatch_qa", "clean_query": "MySQL锁"}',
        '{"sub_queries":[{"query":"MySQL锁","action":"dispatch_qa","reply":""}]}',
    ])
    wf = _wf(llm)

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        return "答案", ["n1"], {"category": "simple", "categories": ["simple"], "sub_count": 1}
    wf.qa.answer = fake_answer

    result = await wf.run(query="MySQL锁", memory=FakeMemory())
    assert result.metadata.get("category") == "simple"
    assert result.metadata.get("action") == "dispatch_qa"
    assert result.metadata.get("sub_count") == 1


async def test_converse_responds_without_retrieval_or_qa():
    # "你好" → action=converse → 门口直接回复，不进 split_answer
    llm = FakeLLM(['{"action": "converse", "reply": "你好！我是文档知识库助手～"}'])
    wf = _wf(llm)
    called = {"answer": False}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        called["answer"] = True
        return "不应被调用", [], {}
    wf.qa.answer = fake_answer

    result = await wf.run(query="你好", memory=FakeMemory())
    assert called["answer"] is False             # converse 门口拦截，不进 QA
    assert "知识库助手" in str(result.response)
    assert llm.calls == 1                         # 只有门口这一次
    assert result.metadata == {"action": "converse"}   # 非 qa 路径 meta 退化为仅 action


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

    answer_called = {"v": False}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        answer_called["v"] = True
        return "不应被调用", [], {}
    wf.qa.answer = fake_answer

    result = await wf.run(query="现在库里都有什么书", memory=FakeMemory())
    assert answer_called["v"] is False              # 元查询不进 QA
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

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        raise AssertionError("元查询不应进 QA")
    wf.qa.answer = fake_answer

    result = await wf.run(query="现在有多少本书", memory=FakeMemory())
    assert "3" in str(result.response)
    assert result.metadata.get("action") == "converse"
    # 2nd prompt 的 {data} 被真实计数 tool_result 替换，非占位符
    assert "已入库 3 本" in llm.prompts[1]
    assert "未能读取库藏清单" not in llm.prompts[1]


# ── 全库不预收窄（Task 1：移除 scoper 后）──────────────────────────────
async def test_full_library_not_narrowed_passes_none_book_titles():
    # 未手选书 → book_titles 全程为 None（全库），不再有任何预收窄
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲MySQL和openclaw的gateway"}',
        '{"sub_queries":[{"query":"讲讲MySQL和openclaw的gateway","action":"dispatch_qa","reply":""}]}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_answer(ctx, sub_queries, bt, probe=True, disable_scope=False):
        captured["book_titles"] = bt
        return "答案", [], {"category": "multi"}
    wf.qa.answer = fake_answer

    await wf.run(query="讲讲MySQL和openclaw的gateway", memory=FakeMemory())
    assert captured["book_titles"] is None        # 全库，无预收窄
    assert not hasattr(wf, "scoper")              # scoper 已从编排移除


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


# ── qa_agent 注入（Task 6）：simple/complex 升级需要 wf.qa.qa_agent is wf.qa_agent ──
def test_qa_agent_injected_into_qa_capability():
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert wf.qa.qa_agent is wf.qa_agent


# ── other_agent_enabled 透传为 qa.agent_enabled（ablation 轴重接）────────
def test_other_agent_enabled_defaults_to_true_and_propagates():
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert wf.qa.agent_enabled is True


def test_other_agent_enabled_false_propagates_to_qa_agent_enabled():
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(), other_agent_enabled=False)
    assert wf.qa.agent_enabled is False
