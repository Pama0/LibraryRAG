"""DocQueryWorkflow 接线测试：门口 Router → 按 intent dispatch → QA 直接检索+合成。

聚焦：
- Router 在门口跑，clean_query 是 QA 预处理真正消费的输入（不被二次消指代）。
- intent=study_plan → 占位分支短路，不进 QA 预处理 / 检索。
- intent=qa → QA 预处理 → 分支直接检索+合成，answer 收到 clean/降噪后的 query + scope。
- _answer：检索 + 流式合成，发检索进度事件；空命中返回范围提示。

真实合成（LLM）不在范围，stub 掉 _answer / _synthesize_stream / _stream_tokens。
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


class _FakeStore:
    def __init__(self):
        self._d = {}

    async def get(self, k, default=None):
        return self._d.get(k, default)

    async def set(self, k, v):
        self._d[k] = v


class FakeCtx:
    """实现 split_branch / _answer 用到的 write_event_to_stream + store。"""

    def __init__(self):
        self.events = []
        self.store = _FakeStore()

    def write_event_to_stream(self, ev):
        self.events.append(ev)

    async def store_set(self, k, v):  # 测试便捷入口
        await self.store.set(k, v)


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

    async def fake_answer(ctx, query, book_titles):
        captured["query"] = query
        captured["book_titles"] = book_titles
        return "答案", ["n1"]

    wf._answer = fake_answer

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

    async def fake_answer(ctx, query, book_titles):
        return "答案", []

    wf._answer = fake_answer

    await wf.run(query="这本书讲了什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]   # Router prompt 带上了选中的书


async def test_qa_preprocess_consumes_clean_query_not_original():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "MySQL索引有哪些"}',
        '{"category": "retrievable", "rewritten_query": "MySQL索引有哪些"}',
    ])
    wf = _wf(llm)

    async def fake_answer(ctx, query, book_titles):
        return "答案", []

    wf._answer = fake_answer

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

    async def fake_answer(ctx, query, book_titles):
        captured["query"] = query
        return "答案", []

    wf._answer = fake_answer

    await wf.run(query="B+树索引", memory=FakeMemory())
    assert llm.calls == 2
    assert captured["query"] == "B+树索引"


async def test_missing_info_clarifies_without_retrieval():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明"}',
    ])
    wf = _wf(llm)

    called = {"answer": False}

    async def fake_answer(ctx, query, book_titles):
        called["answer"] = True
        return "不应被调用", []

    wf._answer = fake_answer

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert called["answer"] is False              # 反问，不检索
    assert "指代不明" in str(result.response)


# ── _answer：检索 + 流式合成 ──────────────────────────────────────────
async def test_answer_retrieves_then_synthesizes_with_progress_events():
    llm = FakeLLM([])
    wf = _wf(llm, index_manager=FakeIndexManager(nodes=["n1", "n2"]))

    async def fake_synth(ctx, query, nodes):
        return "合成答案"

    wf._synthesize_stream = fake_synth
    ctx = FakeCtx()

    text, nodes = await wf._answer(ctx, "B+树", None)
    assert text == "合成答案"
    assert nodes == ["n1", "n2"]
    names = [e.__class__.__name__ for e in ctx.events]
    assert "RetrievalStartEvent" in names
    assert "RetrievalDoneEvent" in names


async def test_answer_empty_nodes_returns_scope_hint():
    llm = FakeLLM([])
    wf = _wf(llm, index_manager=FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    text, nodes = await wf._answer(ctx, "不存在的内容", ["某本书"])
    assert nodes == []
    assert "某本书" in text


async def test_synthesize_stream_emits_delta_per_token_and_joins():
    llm = FakeLLM([])
    wf = _wf(llm, index_manager=FakeIndexManager(nodes=["n1"]))

    async def fake_tokens(query, nodes):
        for t in ["合", "成", "答", "案"]:
            yield t

    wf._stream_tokens = fake_tokens
    ctx = FakeCtx()

    text = await wf._synthesize_stream(ctx, "B+树", ["n1"])
    assert text == "合成答案"
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert deltas == ["合", "成", "答", "案"]


# ── split_branch：拆解 → 逐项检索 → map-reduce 汇总 ──────────────────
def _split_wf():
    """构造一个 wf 并 stub 掉外部依赖，聚焦 split_branch 接线。"""
    wf = _wf(FakeLLM([]))
    wf._book_chapters = lambda book_titles: ["3.2.1 工具A", "3.2.2 工具B"]

    async def fake_retrieve(query, book_titles):
        # 定位与各子查询都返回带 chapter 的假节点
        class N:
            metadata = {"chapter": "3.2.1 工具A"}

            def get_content(self):
                return "正文"

        return [N()]

    wf._retrieve_nodes = fake_retrieve

    async def fake_synth(ctx, query, nodes):
        return f"[{query}的合成]"

    wf._synthesize_stream = fake_synth
    return wf


async def test_split_branch_decomposes_and_concatenates_sections():
    wf = _split_wf()

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["工具A 是什么", "工具B 怎么用"]

    wf.decomposer.run = fake_decompose
    ctx = FakeCtx()
    from core.workflow.doc_workflow import SplitEvent

    await ctx.store_set("book_titles", ["openclaw"])  # 见下方 FakeCtx.store 扩展
    ev = SplitEvent(rewritten_query="openclaw 的工具系统")
    result = await wf.split_branch(ctx, ev)

    # 答案按子项分节拼接
    assert "## 工具A 是什么" in result.answer
    assert "## 工具B 怎么用" in result.answer
    assert "[工具A 是什么的合成]" in result.answer


async def test_split_branch_emits_single_retrieval_done_and_section_headings():
    wf = _split_wf()

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["子项1", "子项2"]

    wf.decomposer.run = fake_decompose
    ctx = FakeCtx()
    await ctx.store_set("book_titles", ["openclaw"])
    from core.workflow.doc_workflow import SplitEvent

    await wf.split_branch(ctx, SplitEvent(rewritten_query="q"))
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalDoneEvent") == 1          # 只发一次
    headings = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert any("## 子项1" in h for h in headings)
    assert any("## 子项2" in h for h in headings)


async def test_split_branch_falls_back_to_single_retrieve_when_no_subqueries():
    wf = _split_wf()

    async def empty_decompose(clean_query, headings, passages, max_items):
        return []

    wf.decomposer.run = empty_decompose
    ctx = FakeCtx()
    await ctx.store_set("book_titles", ["openclaw"])
    from core.workflow.doc_workflow import SplitEvent

    result = await wf.split_branch(ctx, SplitEvent(rewritten_query="openclaw 工具系统"))
    # 降级：直接对整句合成
    assert result.answer == "[openclaw 工具系统的合成]"
