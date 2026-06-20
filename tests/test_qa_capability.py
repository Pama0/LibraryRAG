"""QaCapability 单测：检索 + 流式合成 + 拆解-检索-汇总（split）。

从 DocQueryWorkflow 抽出后，QA 的检索/合成实质逻辑在此独立测，不经 workflow
step 机制。真实合成（LLM）/真实 chroma 不在范围，stub 掉检索 / token 源 / 拆解。
"""
from core.workflow.qa_capability import QaCapability
from core.workflow.query_dimension import Dimension


# ── 替身 ─────────────────────────────────────────────────────────────
class FakeLLM:
    async def acomplete(self, prompt, **kw):  # 本文件不直接驱动 LLM
        raise AssertionError("不应被调用")


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
    """实现 split / retrieve 用到的 write_event_to_stream + store。"""

    def __init__(self):
        self.events = []
        self.store = _FakeStore()

    def write_event_to_stream(self, ev):
        self.events.append(ev)


def _qa(index_manager=None):
    return QaCapability(index_manager, FakeLLM(), similarity_top_k=3)


# ── retrieve：检索 + 流式合成 ─────────────────────────────────────────
async def test_retrieve_then_synthesizes_with_progress_events():
    qa = _qa(FakeIndexManager(nodes=["n1", "n2"]))

    async def fake_synth(ctx, query, nodes):
        return "合成答案"

    qa._synthesize_stream = fake_synth
    ctx = FakeCtx()

    text, nodes = await qa.retrieve(ctx, "B+树", None)
    assert text == "合成答案"
    assert nodes == ["n1", "n2"]
    names = [e.__class__.__name__ for e in ctx.events]
    assert "RetrievalStartEvent" in names
    assert "RetrievalDoneEvent" in names


async def test_retrieve_empty_nodes_returns_scope_hint():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    text, nodes = await qa.retrieve(ctx, "不存在的内容", ["某本书"])
    assert nodes == []
    assert "某本书" in text


async def test_synthesize_stream_emits_delta_per_token_and_joins():
    qa = _qa(FakeIndexManager(nodes=["n1"]))

    async def fake_tokens(query, nodes):
        for t in ["合", "成", "答", "案"]:
            yield t

    qa._stream_tokens = fake_tokens
    ctx = FakeCtx()

    text = await qa._synthesize_stream(ctx, "B+树", ["n1"])
    assert text == "合成答案"
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert deltas == ["合", "成", "答", "案"]


# ── split：拆解 → 逐项检索 → map-reduce 汇总 ──────────────────────────
def _split_qa():
    """构造 qa 并 stub 掉外部依赖，聚焦 split 编排。"""
    qa = _qa()
    qa._book_chapters = lambda book_titles: ["3.2.1 工具A", "3.2.2 工具B"]

    async def fake_retrieve_nodes(query, book_titles):
        class N:
            metadata = {"chapter": "3.2.1 工具A"}

            def get_content(self):
                return "正文"

        return [N()]

    qa._retrieve_nodes = fake_retrieve_nodes

    async def fake_synth(ctx, query, nodes):
        return f"[{query}的合成]"

    qa._synthesize_stream = fake_synth
    return qa


async def test_split_decomposes_and_concatenates_sections():
    qa = _split_qa()

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["工具A 是什么", "工具B 怎么用"], "list"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "openclaw 的工具系统", ["openclaw"])

    # 答案按子项分节拼接
    assert "## 工具A 是什么" in answer
    assert "## 工具B 怎么用" in answer
    assert "[工具A 是什么的合成]" in answer


async def test_split_emits_single_retrieval_done_and_section_headings():
    qa = _split_qa()

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["子项1", "子项2"], "list"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "q", ["openclaw"])
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalDoneEvent") == 1          # 只发一次
    headings = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert any("## 子项1" in h for h in headings)
    assert any("## 子项2" in h for h in headings)


async def test_split_falls_back_to_single_retrieve_when_no_subqueries():
    qa = _split_qa()

    async def empty_decompose(clean_query, headings, passages, max_items):
        return [], "list"

    qa.decomposer.run = empty_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "openclaw 工具系统", ["openclaw"])
    # 降级：直接对整句合成
    assert answer == "[openclaw 工具系统的合成]"


# ── assume：归纳维度 → 声明角度 → 逐维度检索分节 ──────────────────────


def _assume_qa():
    """构造 qa 并 stub 掉外部依赖，聚焦 assume 编排。"""
    qa = _qa()

    async def fake_retrieve_nodes(query, book_titles):
        class N:
            metadata = {"chapter": ""}

            def get_content(self):
                return "正文"

        return [N()]

    qa._retrieve_nodes = fake_retrieve_nodes

    async def fake_synth(ctx, query, nodes):
        return f"[{query}的合成]"

    qa._synthesize_stream = fake_synth
    return qa


async def test_assume_declares_angles_and_sections_per_dimension():
    qa = _assume_qa()

    async def fake_dims(clean_query, passages, max_items):
        return [
            Dimension(label="读写性能", query="Redis 缓存读写性能"),
            Dimension(label="一致性", query="Redis 缓存数据一致性"),
        ]

    qa.dimensioner.run = fake_dims
    ctx = FakeCtx()

    answer, nodes = await qa.assume(ctx, "Redis 做缓存好吗", ["Redis"])

    # 角度声明（preamble）
    assert "可以从以下角度来看" in answer
    assert "读写性能" in answer and "一致性" in answer
    # 按维度分节，合成用的是维度的子查询
    assert "## 读写性能" in answer
    assert "## 一致性" in answer
    assert "[Redis 缓存读写性能的合成]" in answer


async def test_assume_emits_single_retrieval_done_and_declares_before_sections():
    qa = _assume_qa()

    async def fake_dims(clean_query, passages, max_items):
        return [Dimension(label="角度A", query="qa"), Dimension(label="角度B", query="qb")]

    qa.dimensioner.run = fake_dims
    ctx = FakeCtx()

    await qa.assume(ctx, "q", ["书"])
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalDoneEvent") == 1          # 只发一次
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    # 声明 delta 在所有分节标题 delta 之前
    decl_idx = next(i for i, d in enumerate(deltas) if "可以从以下角度来看" in d)
    sec_idx = next(i for i, d in enumerate(deltas) if "## 角度A" in d)
    assert decl_idx < sec_idx


async def test_assume_falls_back_to_single_retrieve_when_no_dimensions():
    qa = _assume_qa()

    async def empty_dims(clean_query, passages, max_items):
        return []

    qa.dimensioner.run = empty_dims
    ctx = FakeCtx()

    answer, nodes = await qa.assume(ctx, "Redis 做缓存好吗", ["书"])
    # 降级：对整句直接合成，无角度声明、无分节标题
    assert answer == "[Redis 做缓存好吗的合成]"
    assert "可以从以下角度来看" not in answer
    assert "##" not in answer


async def test_assume_empty_located_and_no_dimensions_returns_scope_hint():
    qa = _qa()

    async def empty_retrieve_nodes(query, book_titles):
        return []

    qa._retrieve_nodes = empty_retrieve_nodes

    async def empty_dims(clean_query, passages, max_items):
        return []

    qa.dimensioner.run = empty_dims
    ctx = FakeCtx()

    answer, nodes = await qa.assume(ctx, "不存在的内容", ["某本书"])
    assert nodes == []
    assert "某本书" in answer
    assert "没有检索到" in answer


# ── retrieve preamble：降级声明假设、尽力答 ──────────────────────────
async def test_retrieve_with_preamble_prepends_declaration_and_emits_delta():
    qa = _qa(FakeIndexManager(nodes=["n1"]))

    async def fake_synth(ctx, query, nodes):
        return "正文答案"

    qa._synthesize_stream = fake_synth
    ctx = FakeCtx()

    text, nodes = await qa.retrieve(ctx, "这个索引", None, preamble="（注：按最可能解读作答）")
    assert text == "（注：按最可能解读作答）正文答案"
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert "（注：按最可能解读作答）" in deltas


async def test_retrieve_empty_nodes_ignores_preamble():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()
    text, nodes = await qa.retrieve(ctx, "这个索引", ["书"], preamble="（注：声明）")
    assert nodes == []
    assert "（注：声明）" not in text   # 空命中只给范围提示，不带声明


# ── classify：probe-then-classify ───────────────────────────────────
class _PNode:
    def __init__(self, content, book="openclaw", chapter="3.2"):
        self._c = content
        self.metadata = {"book_title": book, "chapter": chapter}

    def get_content(self):
        return self._c


async def test_classify_probes_then_passes_context_to_preprocessor():
    qa = _qa(FakeIndexManager(nodes=[_PNode("openclaw 是一个工具")]))
    captured = {}

    async def fake_run(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    qa.preprocessor.run = fake_run
    await qa.classify("给我讲明白openclaw", ["openclaw"])
    assert "openclaw 是一个工具" in captured["ctx"]   # 探测片段进了召回上下文
    assert "《openclaw》" in captured["ctx"]           # 章节分布进了上下文


async def test_classify_degrades_when_probe_fails():
    qa = _qa(index_manager=None)   # 无 index → probe 抛错
    captured = {}

    async def fake_run(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    qa.preprocessor.run = fake_run
    result = await qa.classify("openclaw", ["openclaw"])
    assert captured["ctx"] == ""           # probe 失败 → 空上下文，不阻塞
    assert result.category == "retrievable"


def test_format_probe_empty_and_nonempty():
    qa = _qa()
    assert "未召回" in qa._format_probe([], None)
    out = qa._format_probe([_PNode("片段X", book="A", chapter="1.1")], None)
    assert "共命中 1 段" in out and "《A》1.1" in out and "片段X" in out


def test_format_probe_concentrated_reports_single_chapter():
    """召回集中在同一书同一章 → 显式标注「跨 1 个章节」（有主导章，倾向 retrievable 的旁证）。"""
    qa = _qa()
    nodes = [
        _PNode("锁的分类", book="MySQL", chapter="6.1 锁"),
        _PNode("行级锁细节", book="MySQL", chapter="6.1 锁"),
        _PNode("表级锁细节", book="MySQL", chapter="6.1 锁"),
    ]
    out = qa._format_probe(nodes, ["MySQL"])
    assert "共命中 3 段" in out
    assert "跨 1 个章节" in out


def test_format_probe_scattered_reports_chapter_span():
    """一句 query 已摊到多个互不重叠章节 → 「跨 3 个章节」（无主导章，支持 pending_split 的旁证）。"""
    qa = _qa()
    nodes = [
        _PNode("索引优化", book="MySQL", chapter="5 索引"),
        _PNode("查询优化", book="MySQL", chapter="7 查询"),
        _PNode("配置调优", book="MySQL", chapter="9 配置"),
    ]
    out = qa._format_probe(nodes, ["MySQL"])
    assert "共命中 3 段" in out
    assert "跨 3 个章节" in out


# ── rerank 接入 ───────────────────────────────────────────────────────
class _RecordingReranker:
    """记录入参；把候选倒序后截 top_n（验证顺序确实被改 + 截断）。"""

    def __init__(self):
        self.calls = []

    async def rerank(self, query, nodes, top_n):
        self.calls.append((query, list(nodes), top_n))
        return list(reversed(nodes))[:top_n]


async def test_retrieve_nodes_without_reranker_keeps_top_k():
    im = FakeIndexManager(nodes=["a", "b", "c"])
    qa = QaCapability(im, FakeLLM(), similarity_top_k=3)

    nodes = await qa._retrieve_nodes("q", None)

    assert nodes == ["a", "b", "c"]
    # 基线：用 similarity_top_k 召回，不过召回
    assert im._index.last_kw["similarity_top_k"] == 3


async def test_retrieve_nodes_with_reranker_overfetches_then_truncates():
    im = FakeIndexManager(nodes=["a", "b", "c", "d", "e"])
    rr = _RecordingReranker()
    qa = QaCapability(im, FakeLLM(), similarity_top_k=2,
                      reranker=rr, rerank_candidate_k=5)

    nodes = await qa._retrieve_nodes("B+树", None)

    # 召回用候选池大小，不是 top_k
    assert im._index.last_kw["similarity_top_k"] == 5
    # reranker 收到候选并按 top_n 截断
    assert rr.calls == [("B+树", ["a", "b", "c", "d", "e"], 2)]
    assert nodes == ["e", "d"]  # 倒序后截 2


# ── retriever 接入 ────────────────────────────────────────────────────
class _RecordingRetriever:
    def __init__(self, nodes):
        self._nodes = nodes
        self.calls = []

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.calls.append((query, book_titles, top_k))
        return self._nodes


async def test_retrieve_nodes_delegates_to_injected_retriever():
    rr = _RecordingRetriever(nodes=["x", "y"])
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      similarity_top_k=4, retriever=rr)

    nodes = await qa._retrieve_nodes("B+树", ["《A》"])

    assert nodes == ["x", "y"]
    assert rr.calls == [("B+树", ["《A》"], 4)]   # 无 reranker → top_k=similarity_top_k


async def test_retrieve_nodes_default_retriever_is_vector():
    from core.retrieval.retrieve import VectorRetriever
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM())
    assert isinstance(qa.retriever, VectorRetriever)


async def test_retrieve_nodes_retriever_overfetches_when_reranker_set():
    rr_ret = _RecordingRetriever(nodes=["a", "b", "c", "d", "e"])
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      similarity_top_k=2, retriever=rr_ret,
                      reranker=_RecordingReranker(), rerank_candidate_k=5)

    await qa._retrieve_nodes("q", None)
    # retriever 拿候选池大小 5（reranker 再截 2）
    assert rr_ret.calls[0][2] == 5


# ── probe 检索解耦：独立 retriever + reranker（默认 vector / 不重排）──────
async def test_probe_retriever_defaults_to_vector_and_no_reranker():
    from core.retrieval.retrieve import VectorRetriever
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM())
    assert isinstance(qa.probe_retriever, VectorRetriever)
    assert qa.probe_reranker is None


async def _run_classify(qa, query="openclaw", books=None):
    async def fake_run(clean_query, retrieval_context=""):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    qa.preprocessor.run = fake_run
    return await qa.classify(query, books or ["openclaw"])


async def test_classify_probe_uses_probe_retriever_not_answer():
    answer_ret = _RecordingRetriever(nodes=[_PNode("答案侧片段")])
    probe_ret = _RecordingRetriever(nodes=[_PNode("probe侧片段")])
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      retriever=answer_ret, probe_retriever=probe_ret)

    await _run_classify(qa)

    assert len(probe_ret.calls) == 1   # probe 走独立 probe_retriever
    assert answer_ret.calls == []      # 答案 retriever 不被 probe 触发


async def test_classify_probe_does_not_rerank_by_default():
    probe_ret = _RecordingRetriever(nodes=[_PNode("片段")])
    answer_rr = _RecordingReranker()
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      reranker=answer_rr,            # 答案侧有重排
                      probe_retriever=probe_ret)     # probe 独立 retriever，默认不重排

    await _run_classify(qa)

    assert answer_rr.calls == []       # probe 默认不触发任何 rerank


async def test_classify_probe_uses_probe_reranker_when_explicitly_given():
    probe_ret = _RecordingRetriever(nodes=["a", "b", "c"])
    probe_rr = _RecordingReranker()
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      similarity_top_k=2, rerank_candidate_k=3,
                      probe_retriever=probe_ret, probe_reranker=probe_rr)

    await _run_classify(qa)

    assert probe_ret.calls[0][2] == 3  # 有 probe reranker → 过召回候选池
    assert len(probe_rr.calls) == 1
    assert probe_rr.calls[0][2] == 2   # 截回 similarity_top_k


import asyncio


async def test_retrieve_all_runs_concurrently_and_preserves_order():
    qa = _qa()
    started = 0
    release = asyncio.Event()

    async def fake_rn(query, book_titles):
        nonlocal started
        started += 1
        if started >= 2:
            release.set()      # 两个都进来才放行 → 串行会卡死，证明并发
        await release.wait()
        return [query]

    qa._retrieve_nodes = fake_rn
    out = await asyncio.wait_for(qa._retrieve_all(["a", "b"], None), timeout=1.0)
    assert out == [["a"], ["b"]]   # gather 保持入参顺序


# ── split：synthesize 模式（扇出 → 去重合并 → 对原始问题单次整合合成）──
class _IdNode:
    metadata = {"chapter": ""}

    def __init__(self, nid, content="正文", score=1.0):
        self.node_id = nid
        self._c = content
        self.score = score

    def get_content(self):
        return self._c


def _synth_qa(retrieve_map):
    """retrieve_map: {子查询: [节点...]}；记录 _synthesize_stream 的调用。"""
    qa = _qa()
    qa._book_chapters = lambda book_titles: []

    async def fake_retrieve_nodes(query, book_titles):
        return retrieve_map.get(query, [])

    qa._retrieve_nodes = fake_retrieve_nodes

    calls = []

    async def fake_synth(ctx, query, nodes):
        calls.append((query, list(nodes)))
        return f"[整合:{query}]"

    qa._synthesize_stream = fake_synth
    qa._synth_calls = calls
    return qa


async def test_split_synthesize_single_synthesis_over_merged_pool():
    a, b = _IdNode("1"), _IdNode("2")
    qa = _synth_qa({"locate": [a], "子查询A": [a], "子查询B": [b]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["子查询A", "子查询B"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()  # 定位整句检索走 retrieve_map["locate"]=[a]

    answer, nodes = await qa.split(ctx, "locate", ["书"])

    # 只合成一次，且用原始问题、池含两子查询去重后的节点
    assert len(qa._synth_calls) == 1
    synth_query, synth_nodes = qa._synth_calls[0]
    assert synth_query == "locate"
    assert {n.node_id for n in synth_nodes} == {"1", "2"}
    assert "##" not in answer            # 单段连贯，无分节标题
    assert answer == "[整合:locate]"


async def test_split_synthesize_dedupes_overlapping_nodes():
    shared = _IdNode("dup")
    qa = _synth_qa({"locate": [shared], "qa": [shared], "qb": [shared, _IdNode("x")]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "locate", ["书"])
    _q, synth_nodes = qa._synth_calls[0]
    assert [n.node_id for n in synth_nodes] == ["dup", "x"]   # 按 node_id 去重，保序


async def test_split_synthesize_without_reranker_sorts_pool_by_score_desc():
    # 合并序 [low, high]，无 reranker → 应按 score 降序重排为 [high, low]
    low, high = _IdNode("low", score=0.2), _IdNode("high", score=0.9)
    qa = _synth_qa({"locate": [low], "qa": [low], "qb": [high]})
    assert qa.reranker is None

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "locate", ["书"])
    _q, synth_nodes = qa._synth_calls[0]
    assert [n.node_id for n in synth_nodes] == ["high", "low"]


async def test_split_synthesize_emits_single_retrieval_done():
    qa = _synth_qa({"locate": [_IdNode("1")], "qa": [_IdNode("1")], "qb": [_IdNode("2")]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "locate", ["书"])
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalDoneEvent") == 1


async def test_split_synthesize_empty_pool_returns_scope_hint():
    qa = _synth_qa({"locate": [_IdNode("1")], "qa": [], "qb": []})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "locate", ["某本书"])
    assert nodes == []
    assert "某本书" in answer and "没有检索到" in answer


async def test_split_synthesize_single_subquery_degrades_to_single():
    only = _IdNode("only")
    qa = _synth_qa({"locate": [only], "唯一子查询": [only]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["唯一子查询"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "locate", ["书"])
    assert len(qa._synth_calls) == 1
    assert qa._synth_calls[0][0] == "locate"   # 仍对原始问题单次合成
    assert "##" not in answer
