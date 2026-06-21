"""QaCapability 单测：检索 + 流式合成 + 拆解-检索-汇总（split）。

从 DocQueryWorkflow 抽出后，QA 的检索/合成实质逻辑在此独立测，不经 workflow
step 机制。真实合成（LLM）/真实 chroma 不在范围，stub 掉检索 / token 源 / 拆解。
"""
from core.workflow.qa_capability import QaCapability, AnswerDeltaEvent
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
        return PreprocessResult("retrievable")

    async def fake_admit(query, passages):
        from core.workflow.admitter import AdmitVerdict
        return AdmitVerdict(verdict="ok")

    qa.preprocessor.run = fake_run
    qa.admitter.run = fake_admit
    await qa.classify("给我讲明白openclaw", ["openclaw"])
    assert "openclaw 是一个工具" in captured["ctx"]   # 探测片段进了召回上下文
    assert "《openclaw》" in captured["ctx"]           # 章节分布进了上下文


async def test_classify_degrades_when_probe_fails():
    qa = _qa(index_manager=None)   # 无 index → probe 抛错
    captured = {}

    async def fake_run(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    async def fake_admit(query, passages):
        from core.workflow.admitter import AdmitVerdict
        return AdmitVerdict(verdict="ok")

    qa.preprocessor.run = fake_run
    qa.admitter.run = fake_admit
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
        return PreprocessResult("retrievable")

    async def fake_admit(query, passages):
        from core.workflow.admitter import AdmitVerdict
        return AdmitVerdict(verdict="ok")

    qa.preprocessor.run = fake_run
    qa.admitter.run = fake_admit
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


# ── explain：宽召回 → 列骨架 → 每节点检索 → 教学体合成 ──────────────────
import pytest
from core.workflow.qa_capability import EmptySkeleton


class _RecallNode:
    """宽召回返回的 node 替身：explain 用 n.text 抽 passages。"""

    def __init__(self, text):
        self.text = text


async def test_explain_outlines_with_toc_then_teaches_over_merged_pool():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()
    seen = {}

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1"), _RecallNode("w2")]

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        seen["toc_hint"] = toc_hint            # 捕获 explain 传入的 TOC 提示
        return [Dimension(label="是什么", query="什么是MySQL"),
                Dimension(label="组成", query="MySQL由哪些部分组成")]

    async def fake_retrieve_all(sub_queries, book_titles):
        seen["sub_queries"] = sub_queries      # 应是各维度的 query
        return [["a1"], ["b1"]]

    async def fake_teach(ctx, query, outline, pool):
        seen["teach"] = (query, [d.label for d in outline], pool)
        return f"[teach:{query}]"

    qa._explain_recall = fake_recall
    qa._book_chapters = lambda book_titles: ["第1章 索引", "第2章 事务"]
    qa.outliner.run = fake_outline
    qa._retrieve_all = fake_retrieve_all
    qa._teach_synthesize = fake_teach

    answer, nodes = await qa.explain(ctx, "MySQL基础知识", None)

    assert seen["toc_hint"] == ["第1章 索引", "第2章 事务"]      # TOC 喂给 outliner
    assert seen["sub_queries"] == ["什么是MySQL", "MySQL由哪些部分组成"]
    assert nodes == ["a1", "b1"]                               # 去重合并池
    assert seen["teach"] == ("MySQL基础知识", ["是什么", "组成"], ["a1", "b1"])
    assert answer == "[teach:MySQL基础知识]"                    # 一次整合写的产物


async def test_explain_empty_outline_raises():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1")]

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        return []                                # 列不出教案

    qa._explain_recall = fake_recall
    qa._book_chapters = lambda book_titles: []
    qa.outliner.run = fake_outline

    with pytest.raises(EmptySkeleton):
        await qa.explain(ctx, "讲讲X", None)


async def test_explain_empty_pool_returns_scope_hint():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1")]

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        return [Dimension(label="是什么", query="什么是X")]

    async def fake_retrieve_all(sub_queries, book_titles):
        return [[]]                              # 每维度都召回空 → pool 空

    qa._explain_recall = fake_recall
    qa._book_chapters = lambda book_titles: []
    qa.outliner.run = fake_outline
    qa._retrieve_all = fake_retrieve_all

    answer, nodes = await qa.explain(ctx, "讲讲X", None)
    assert nodes == [] and "没有检索到" in answer   # 有教案但无料 → 如实告知，不强写


async def test_explain_truncates_pool_to_budget_when_no_reranker():
    qa = _qa(FakeIndexManager(nodes=[]))
    qa.rerank_candidate_k = 2
    ctx = FakeCtx()

    class _Scored:
        def __init__(self, nid, score):
            self.node_id = nid
            self.score = score
        def get_content(self):
            return self.node_id

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1")]

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        return [Dimension(label="是什么", query="q")]

    async def fake_retrieve_all(sub_queries, book_titles):
        return [[_Scored("low", 0.1), _Scored("high", 0.9), _Scored("mid", 0.5)]]

    captured = {}

    async def fake_teach(ctx, query, outline, pool):
        captured["pool"] = [n.node_id for n in pool]
        return "x"

    qa._explain_recall = fake_recall
    qa._book_chapters = lambda book_titles: []
    qa.outliner.run = fake_outline
    qa._retrieve_all = fake_retrieve_all
    qa._teach_synthesize = fake_teach

    await qa.explain(ctx, "讲讲X", None)
    assert captured["pool"] == ["high", "mid"]    # 无 reranker：按 score 降序截到 rerank_candidate_k


async def test_gate_delegates_to_query_gate():
    qa = _qa()

    async def fake_run(clean_query):
        return "降噪后", "explain"

    qa._gate.run = fake_run
    denoised, intent = await qa.gate("原始 query")
    assert (denoised, intent) == ("降噪后", "explain")


# ── _teach_synthesize：教案 + pool → 一次整合教学写作 ──────────────────
class _TeachChunk:
    def __init__(self, delta):
        self.delta = delta


class FakeStreamLLM:
    """暴露 astream_complete 的替身：记录 prompt、按预设 deltas 逐块流出。"""

    def __init__(self, deltas):
        self._deltas = list(deltas)
        self.prompts = []

    async def astream_complete(self, prompt, **kw):
        self.prompts.append(prompt)

        async def _gen():
            for d in self._deltas:
                yield _TeachChunk(d)

        return _gen()


class _PoolNode:
    """pool 节点替身：_teach_synthesize 用 get_content() 抽正文。"""

    def __init__(self, text):
        self._text = text

    def get_content(self):
        return self._text


async def test_teach_synthesize_builds_prompt_streams_once_and_emits_deltas():
    qa = QaCapability(None, FakeStreamLLM(["## 是什么\n", "MySQL 是…", "## 组成\n", "由…"]))
    ctx = FakeCtx()
    outline = [Dimension(label="是什么", query="什么是MySQL"),
               Dimension(label="组成", query="MySQL由哪些部分组成")]
    pool = [_PoolNode("片段甲"), _PoolNode("片段乙")]

    answer = await qa._teach_synthesize(ctx, "MySQL基础知识", outline, pool)

    # 一次流：只调用一次 astream_complete
    assert len(qa.llm.prompts) == 1
    prompt = qa.llm.prompts[0]
    # 教案维度 label 进 prompt
    assert "是什么" in prompt and "组成" in prompt
    # grounding 铁律进 prompt
    assert "只能来自" in prompt
    # pool 片段进 prompt
    assert "片段甲" in prompt and "片段乙" in prompt
    # 轻分节格式指令进 prompt
    assert "##" in prompt
    # 逐 token 发 AnswerDeltaEvent，拼回全文
    deltas = [e.delta for e in ctx.events if isinstance(e, AnswerDeltaEvent)]
    assert deltas == ["## 是什么\n", "MySQL 是…", "## 组成\n", "由…"]
    assert answer == "## 是什么\nMySQL 是…## 组成\n由…"


# ── 异常 + 拒答常量（Task 2：纯加法，验可导入 + 值锁定）─────────────────
from core.workflow.qa_capability import (
    OutOfScope, MissingInfo, REFUSAL_TEXT, REFUSAL_FALLBACK,
)


def test_refusal_text_matches_existing_out_of_scope_branch_wording():
    # 原样抽自 doc_workflow.out_of_scope_branch 的终结句，一字不差
    assert REFUSAL_TEXT == (
        "这个问题知识库里暂未收录相关内容，我没法基于现有资料回答。"
        "你可以换个已入库主题问我，或把问题换个角度再试试～"
    )


def test_refusal_fallback_is_a_clarify_question():
    # missing_info 缺 clarify_question 时的兜底反问：是一句引导补充的话
    assert isinstance(REFUSAL_FALLBACK, str) and len(REFUSAL_FALLBACK) > 0
    assert "？" in REFUSAL_FALLBACK or "?" in REFUSAL_FALLBACK


def test_out_of_scope_exception_carries_query():
    exc = OutOfScope("PostgreSQL的MVCC")
    assert isinstance(exc, Exception)
    assert exc.args == ("PostgreSQL的MVCC",)


def test_missing_info_exception_carries_clarify_question():
    exc = MissingInfo("你说的「这个索引」指哪一个？")
    assert isinstance(exc, Exception)
    assert exc.clarify_question == "你说的「这个索引」指哪一个？"


def test_missing_info_exception_default_clarify_empty():
    exc = MissingInfo()
    assert exc.clarify_question == ""


# ── classify：admit 短路 / ok 落 preprocessor（Task 3）──────────────────
from core.workflow.query_preprocess import PreprocessResult
from core.workflow.admitter import Admitter, AdmitVerdict


async def test_classify_admit_out_of_scope_short_circuits_before_preprocessor():
    qa = _qa(FakeIndexManager(nodes=[_PNode("openclaw 是一个工具")]))
    preprocessor_called = {"v": False}

    async def fake_preprocessor(clean_query, retrieval_context=""):
        preprocessor_called["v"] = True
        return PreprocessResult("retrievable")

    async def fake_admit(query, passages):
        return AdmitVerdict(verdict="out_of_scope", reason="库外，召回全是别的系统")

    qa.preprocessor.run = fake_preprocessor
    qa.admitter.run = fake_admit

    result = await qa.classify("PostgreSQL的MVCC", ["openclaw"])
    assert result.category == "out_of_scope"
    assert result.reason == "库外，召回全是别的系统"
    assert preprocessor_called["v"] is False    # 短路，不跑瘦身分类器


async def test_classify_admit_missing_info_short_circuits_with_clarify():
    qa = _qa(FakeIndexManager(nodes=[_PNode("索引内容")]))
    preprocessor_called = {"v": False}

    async def fake_preprocessor(clean_query, retrieval_context=""):
        preprocessor_called["v"] = True
        return PreprocessResult("retrievable")

    async def fake_admit(query, passages):
        return AdmitVerdict(
            verdict="missing_info", reason="指代不明",
            clarify_question="你说的「这个索引」指哪一个？B+树还是全文索引？",
        )

    qa.preprocessor.run = fake_preprocessor
    qa.admitter.run = fake_admit

    result = await qa.classify("这个索引的应用场景", ["openclaw"])
    assert result.category == "missing_info"
    assert result.clarify_question == "你说的「这个索引」指哪一个？B+树还是全文索引？"
    assert preprocessor_called["v"] is False


async def test_classify_admit_ok_falls_through_to_preprocessor():
    qa = _qa(FakeIndexManager(nodes=[_PNode("openclaw 是一个工具")]))
    captured = {"ctx": None}

    async def fake_preprocessor(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        return PreprocessResult("pending_split", reason="需扇出")

    async def fake_admit(query, passages):
        return AdmitVerdict(verdict="ok")

    qa.preprocessor.run = fake_preprocessor
    qa.admitter.run = fake_admit

    result = await qa.classify("讲讲openclaw", ["openclaw"])
    assert result.category == "pending_split"          # ok → 跑瘦身分类器
    assert "openclaw 是一个工具" in captured["ctx"]    # probe 证据透传给 preprocessor（行为同前）


async def test_classify_admit_failure_degrades_to_ok_and_runs_preprocessor():
    # admit 抛错 → 降级 ok → 仍跑 preprocessor（绝不阻塞）
    qa = _qa(FakeIndexManager(nodes=[_PNode("片段")]))
    preprocessor_called = {"v": False}

    async def fake_preprocessor(clean_query, retrieval_context=""):
        preprocessor_called["v"] = True
        return PreprocessResult("retrievable")

    async def boom_admit(query, passages):
        raise RuntimeError("admit 炸了")

    qa.preprocessor.run = fake_preprocessor
    qa.admitter.run = boom_admit

    result = await qa.classify("MySQL锁", ["MySQL"])
    assert result.category == "retrievable"
    assert preprocessor_called["v"] is True


# ── explain：宽召回后 admit，非 ok 抛异常（Task 5）──────────────────────


async def test_explain_admit_out_of_scope_raises_out_of_scope():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("MySQL 片段")]      # 召回到的是别的系统

    async def fake_admit(query, passages):
        return AdmitVerdict(verdict="out_of_scope", reason="PostgreSQL 不在库")

    qa._explain_recall = fake_recall
    qa.admitter.run = fake_admit

    with pytest.raises(OutOfScope):
        await qa.explain(ctx, "PostgreSQL的MVCC", None)


async def test_explain_admit_missing_info_raises_missing_info_with_clarify():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("索引内容")]

    async def fake_admit(query, passages):
        return AdmitVerdict(
            verdict="missing_info", reason="指代不明",
            clarify_question="你说的「这个索引」指哪一个？",
        )

    qa._explain_recall = fake_recall
    qa.admitter.run = fake_admit

    with pytest.raises(MissingInfo) as ei:
        await qa.explain(ctx, "这个索引的应用场景", None)
    assert ei.value.clarify_question == "你说的「这个索引」指哪一个？"


async def test_explain_admit_ok_proceeds_to_outline():
    # admit ok → 进 outline（不抛异常）；用空 outline 触发 EmptySkeleton 验证走到了 outline
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1")]

    async def fake_admit(query, passages):
        return AdmitVerdict(verdict="ok")

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        return []                                # 列不出教案 → EmptySkeleton

    qa._explain_recall = fake_recall
    qa.admitter.run = fake_admit
    qa._book_chapters = lambda book_titles: []
    qa.outliner.run = fake_outline

    with pytest.raises(EmptySkeleton):
        await qa.explain(ctx, "讲讲X", None)      # ok 放行 → 走到 outline → 空教案 → EmptySkeleton


async def test_explain_admit_failure_degrades_to_ok_and_proceeds():
    # admit 抛错 → 降级 ok → 仍进 outline（绝不阻塞）
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return [_RecallNode("w1")]

    async def boom_admit(query, passages):
        raise RuntimeError("admit 炸了")

    async def fake_outline(query, passages, toc_hint=None, max_items=8):
        return []                                # 走到 outline → EmptySkeleton

    qa._explain_recall = fake_recall
    qa.admitter.run = boom_admit
    qa._book_chapters = lambda book_titles: []
    qa.outliner.run = fake_outline

    with pytest.raises(EmptySkeleton):
        await qa.explain(ctx, "讲讲X", None)      # admit 炸 → 降级 ok → 走到 outline


# ── _decide_subq：逐子问题 probe → admit → classify 判定 ──────────────
from core.workflow.admitter import AdmitVerdict
from core.workflow.query_classifier import ClassifyResult


def _qa_for_decide():
    qa = _qa()
    async def fake_probe(q, bt): return []
    qa._probe_retrieve = fake_probe
    qa._format_probe = lambda nodes, bt: "EVIDENCE"
    return qa


async def test_decide_subq_out_of_scope_short_circuits_classify():
    qa = _qa_for_decide()
    async def fake_admit(q, passages): return AdmitVerdict(verdict="out_of_scope", reason="库外")
    qa.admitter.run = fake_admit
    async def boom(q, e): raise AssertionError("classify 不该被调用")
    qa.classifier.run = boom
    d = await qa._decide_subq("PostgreSQL的MVCC", None)
    assert d.verdict == "out_of_scope"
    assert d.category == ""


async def test_decide_subq_missing_info_carries_clarify():
    qa = _qa_for_decide()
    async def fake_admit(q, passages):
        return AdmitVerdict(verdict="missing_info", reason="指代不明", clarify_question="你说的索引指哪个？")
    qa.admitter.run = fake_admit
    d = await qa._decide_subq("这个索引的应用场景", None)
    assert d.verdict == "missing_info"
    assert d.clarify_question == "你说的索引指哪个？"


async def test_decide_subq_ok_runs_classifier():
    qa = _qa_for_decide()
    async def fake_admit(q, passages): return AdmitVerdict(verdict="ok")
    qa.admitter.run = fake_admit
    async def fake_classify(q, e): return ClassifyResult("complex", "多跳")
    qa.classifier.run = fake_classify
    d = await qa._decide_subq("MySQL默认隔离级别有哪些并发问题", None)
    assert d.verdict == "ok"
    assert d.category == "complex"


class _FakeAgent:
    def __init__(self, answer="AGENT答案", nodes=None):
        self._a = answer
        self._n = nodes or ["an"]
        self.called_with = None
    async def run(self, ctx, q, bt):
        self.called_with = q
        return self._a, self._n


async def test_execute_simple_with_enough_evidence_uses_retrieve():
    qa = _qa()
    async def fake_retrieve(ctx, q, bt, preamble=""):
        return "单轮答案", ["n1", "n2"]
    qa.retrieve = fake_retrieve
    qa.qa_agent = _FakeAgent()
    ans, nodes = await qa._execute_subq(FakeCtx(), "MySQL有哪些锁", "simple", None)
    assert ans == "单轮答案"
    assert qa.qa_agent.called_with is None  # 没升级


async def test_execute_simple_weak_evidence_escalates_to_agent():
    qa = _qa()
    async def fake_nodes(q, bt): return []   # 召回空 = 证据不足
    qa._retrieve_nodes = fake_nodes
    qa.qa_agent = _FakeAgent(answer="AGENT答案")
    ans, nodes = await qa._execute_subq(FakeCtx(), "冷门问题", "simple", None)
    assert ans == "AGENT答案"
    assert qa.qa_agent.called_with == "冷门问题"


async def test_execute_complex_uses_agent():
    qa = _qa()
    qa.qa_agent = _FakeAgent(answer="AGENT答案")
    ans, nodes = await qa._execute_subq(FakeCtx(), "怎么优化MySQL", "complex", None)
    assert ans == "AGENT答案"


async def test_execute_complex_agent_none_degrades_single_retrieve():
    qa = _qa()
    qa.qa_agent = None
    async def fake_retrieve(ctx, q, bt, preamble=""): return "降级单轮", ["n"]
    qa.retrieve = fake_retrieve
    ans, nodes = await qa._execute_subq(FakeCtx(), "怎么优化MySQL", "complex", None)
    assert ans == "降级单轮"


async def test_execute_explain_and_compare_route():
    qa = _qa()
    async def fake_explain(ctx, q, bt): return "讲解答案", ["e"]
    async def fake_assume(ctx, q, bt): return "比较答案", ["c"]
    qa.explain = fake_explain
    qa.assume = fake_assume
    a1, _ = await qa._execute_subq(FakeCtx(), "讲讲索引", "explain", None)
    a2, _ = await qa._execute_subq(FakeCtx(), "A和B区别", "compare", None)
    assert a1 == "讲解答案" and a2 == "比较答案"
