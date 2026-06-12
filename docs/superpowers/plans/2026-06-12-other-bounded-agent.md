# other → 有界 agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `other`（高难度/开放问题）分支从「占位单轮检索」实现为「有界 `FunctionAgent` 自由多轮调用检索工具探索」，设步数边界 + 超界强制作答，流式复用既有 SSE（前端零改动）。

**Architecture:** 新建 `core/agent/qa_agent.py` 的 `QaAgent`——封装 LlamaIndex `FunctionAgent`（`early_stopping_method="generate"`）+ 两个工具（`book_search` 检索器返回原文片段、`list_books`）。`run` 桥接：监听 agent 的 `ToolCall`/`ToolCallResult` 转译成 `RetrievalStart`/`RetrievalDone` 推到外层 `ctx`，source_nodes 由工具闭包收集回传。`DocQueryWorkflow.other_branch` 委托 `qa_agent.run`，agent 异常则降级 `qa.retrieve`。`query_preprocess` 的 other 判定放宽（积极）。

**Tech Stack:** Python 3.12，LlamaIndex `FunctionAgent`，DeepSeek（`OpenAILike`），pytest + pytest-asyncio。

参考 spec：`docs/superpowers/specs/2026-06-12-other-bounded-agent-design.md`

---

## File Structure

- **Create** `core/agent/qa_agent.py` — `QaAgent`（FunctionAgent + 工具 + run 桥接 + `_search` helper）。
- **Create** `tests/test_qa_agent.py` — `_search` 检索/收集测试 + run 桥接测试（MockLLM + fake agent）。
- **Modify** `core/workflow/doc_workflow.py` — `__init__` 建 `self.qa_agent`；`other_branch` 委托 + 降级。
- **Modify** `tests/test_doc_workflow.py` — `other_branch` 接线 + 降级测试。
- **Modify** `core/workflow/query_preprocess.py` — `_JUDGE_PROMPT` 的 other 段放宽为积极识别高难度。

---

### Task 1: QaAgent（工具 + run 桥接）

**Files:**
- Create: `core/agent/qa_agent.py`
- Test: `tests/test_qa_agent.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_qa_agent.py`:

```python
"""QaAgent 单测：检索工具（_search）+ run 流式桥接。

FunctionAgent 是真实组件，不在单测范围——用 MockLLM 让其构造通过，run 测试
用 fake agent 替身（产真 ToolCall/ToolCallResult 事件 + 可 await 的 final）。
"""
from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult
from llama_index.core.llms import MockLLM
from llama_index.core.tools import ToolOutput

from core.agent.qa_agent import QaAgent


# ── 替身 ─────────────────────────────────────────────────────────────
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


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class FakeIndexManager:
    def __init__(self, nodes, metas=None):
        self._index = FakeIndex(nodes)
        self.chroma_collection = _FakeCollection(metas or [])

    def get_index(self):
        return self._index


class _Node:
    def __init__(self, content):
        self._c = content

    def get_content(self):
        return self._c


class FakeCtx:
    def __init__(self):
        self.events = []

    def write_event_to_stream(self, ev):
        self.events.append(ev)


class _FakeHandler:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def stream_events(self):
        for e in self._events:
            yield e

    def __await__(self):
        async def _f():
            return self._final
        return _f().__await__()


class FakeAgent:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.last_kwargs = None

    def run(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeHandler(self._events, self._final)


def _agent(index_manager=None):
    return QaAgent(index_manager, MockLLM(), similarity_top_k=3, max_iterations=6)


# ── _search：检索器返回片段 + 收集 nodes ─────────────────────────────
async def test_search_returns_joined_passages_and_collects_nodes():
    qa = _agent(FakeIndexManager(nodes=[_Node("片段A"), _Node("片段B")]))
    qa._run_scope = None
    qa._run_sources = []
    out = await qa._search("分布式事务")
    assert "片段A" in out and "片段B" in out
    assert len(qa._run_sources) == 2


async def test_search_empty_returns_placeholder_and_collects_nothing():
    qa = _agent(FakeIndexManager(nodes=[]))
    qa._run_scope = None
    qa._run_sources = []
    out = await qa._search("不存在")
    assert out == "（未检索到相关内容）"
    assert qa._run_sources == []


# ── run：桥接 agent 事件 → 检索流式 + 返回 (答案, sources) ─────────────
async def test_run_bridges_tool_events_and_emits_final_delta():
    qa = _agent(FakeIndexManager(nodes=[]))
    events = [
        ToolCall(tool_name="book_search", tool_kwargs={"query": "子问题1"}, tool_id="1"),
        ToolCallResult(
            tool_name="book_search",
            tool_kwargs={"query": "子问题1"},
            tool_id="1",
            tool_output=ToolOutput(
                content="片段", tool_name="book_search", raw_input={}, raw_output="片段"
            ),
            return_direct=False,
        ),
    ]
    qa.agent = FakeAgent(events, final="综合答案")
    ctx = FakeCtx()

    answer, nodes = await qa.run(ctx, "openclaw 的整体架构与权衡", None)

    assert answer == "综合答案"
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalStartEvent") == 1
    assert names.count("RetrievalDoneEvent") == 1
    # ToolCall 的 query 进 RetrievalStart
    starts = [e for e in ctx.events if e.__class__.__name__ == "RetrievalStartEvent"]
    assert starts[0].query == "子问题1"
    # 最终答案作为一个 AnswerDelta 推出
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert "综合答案" in deltas


async def test_run_resets_sources_each_call_and_passes_max_iterations():
    qa = _agent(FakeIndexManager(nodes=[]))
    qa._run_sources = ["stale"]          # 上一轮残留
    qa.agent = FakeAgent([], final="答案")
    ctx = FakeCtx()

    answer, nodes = await qa.run(ctx, "q", ["书A"])
    assert nodes == []                   # run 开头清空了残留
    assert qa.agent.last_kwargs.get("max_iterations") == 6
    assert qa.agent.last_kwargs.get("user_msg") == "q"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_agent.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.agent.qa_agent'`

- [ ] **Step 3: 写实现**

创建 `core/agent/qa_agent.py`:

```python
"""QaAgent：other（高难度/开放问题）分支的有界 agent。

DocQueryWorkflow 把 intent=qa & category=other dispatch 到这里。用 LlamaIndex
FunctionAgent 让 LLM 自由多轮调用工具（检索器）探索，设步数边界与超界强制作答，
避免失控（见 ARCHITECTURE.md §2「按可预测性配控制结构」）。

- 工具是检索器：book_search 返回原文片段、list_books 返回书单，agent 多轮综合。
- 边界：max_iterations + early_stopping_method="generate"（超界基于已收集结果作答）。
- 流式：把 agent 的 ToolCall/ToolCallResult 转译成项目既有的 RetrievalStart/Done
  事件推到外层 ctx（前端零改动）；中间 thought 不外露；最终答案推一个 AnswerDelta。
- grounding：system prompt 强约束只基于检索片段；source_nodes 由工具收集回传。
- 每请求随 DocQueryWorkflow 新建，故可用实例变量持 per-run scope/sources（无并发）。
"""
from typing import Optional

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from core.workflow.qa_capability import (
    AnswerDeltaEvent,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)

QA_AGENT_SYSTEM_PROMPT = """你是技术书籍知识库的高难度问答 agent，处理需要多步推理、跨主题综合或开放权衡的复杂问题。

铁律（grounding）：
- 只能基于 book_search 返回的检索片段作答，严禁用你自己的训练知识或常识脑补事实。
- 复杂问题可多次调用 book_search（换关键词/换角度）逐步收集证据，再综合。
- 检索不足以回答时，如实说明缺口，不得编造或推断。

工具：
1. book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。
2. list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。

回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。"""


class QaAgent:
    """other 分支的有界 agent：FunctionAgent + 检索工具 + 流式桥接 + source 收集。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_iterations: int = 6,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_iterations = max_iterations
        # per-run 工作态（每请求新实例，无并发）
        self._run_scope: Optional[list[str]] = None
        self._run_sources: list = []
        self.agent = FunctionAgent(
            tools=self._make_tools(),
            llm=llm,
            system_prompt=QA_AGENT_SYSTEM_PROMPT,
            early_stopping_method="generate",
        )

    def _make_filters(self, book_titles: Optional[list[str]]):
        if not book_titles:
            return None
        return MetadataFilters(filters=[
            MetadataFilter(
                key="book_title",
                operator=FilterOperator.IN,
                value=list(book_titles),
            ),
        ])

    async def _search(self, query: str) -> str:
        """检索器主体：按 query 取 top-k 原文片段并收集 nodes。供 book_search 工具调用。"""
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return "请提供要检索的问题。"
        index = self.index_manager.get_index()
        if index is None:
            return "知识库为空，请先上传 PDF。"
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=self._make_filters(self._run_scope),
        )
        nodes = await retriever.aretrieve(query)
        if not nodes:
            return "（未检索到相关内容）"
        self._run_sources.extend(nodes)
        return "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:500]
            for n in nodes
        )

    def _make_tools(self) -> list:
        async def book_search(query: str) -> str:
            """在书籍知识库中检索与 query 相关的原文片段。

            Args:
                query: 检索问题（字符串）。
            Returns:
                拼接的检索片段；无命中返回占位提示。
            """
            return await self._search(query)

        def list_books() -> str:
            """列出当前知识库已入库的书籍清单。"""
            data = self.index_manager.chroma_collection.get(include=["metadatas"])
            counts: dict[str, int] = {}
            for meta in data.get("metadatas", []) or []:
                title = (meta or {}).get("book_title")
                if not title:
                    continue
                counts[title] = counts.get(title, 0) + 1
            if not counts:
                return "知识库当前为空。"
            return "已入库书籍：\n" + "\n".join(
                f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())
            )

        return [
            FunctionTool.from_defaults(
                fn=book_search,
                name="book_search",
                description="书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。",
            ),
            FunctionTool.from_defaults(
                fn=list_books,
                name="list_books",
                description="列出当前已入库书籍清单。",
            ),
        ]

    async def run(
        self, ctx, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """跑有界 agent，桥接流式事件到外层 ctx，返回 (答案, source_nodes)。

        agent 异常由调用方（other_branch）降级单轮检索。
        """
        self._run_scope = book_titles
        self._run_sources = []

        handler = self.agent.run(user_msg=query, max_iterations=self.max_iterations)
        async for ev in handler.stream_events():
            name = ev.__class__.__name__
            if name == "ToolCall":
                tq = (
                    ev.tool_kwargs.get("query", query)
                    if getattr(ev, "tool_name", "") == "book_search"
                    else query
                )
                ctx.write_event_to_stream(RetrievalStartEvent(query=tq))
            elif name == "ToolCallResult":
                ctx.write_event_to_stream(
                    RetrievalDoneEvent(count=len(self._run_sources))
                )
        final = await handler
        answer = str(final)
        ctx.write_event_to_stream(AnswerDeltaEvent(delta=answer))
        return answer, list(self._run_sources)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_agent.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/agent/qa_agent.py tests/test_qa_agent.py
git commit -m "feat(agent): QaAgent 有界 FunctionAgent（检索工具 + 流式桥接 + source 收集）"
```

---

### Task 2: doc_workflow other_branch 接 QaAgent + 降级

**Files:**
- Modify: `core/workflow/doc_workflow.py`
- Test: `tests/test_doc_workflow.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_doc_workflow.py` 末尾追加（复用已有 `FakeLLM` / `FakeMemory` / `_wf`）:

```python
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


async def test_other_falls_back_to_single_retrieve_when_agent_raises():
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

    result = await wf.run(query="设计题", memory=FakeMemory())
    assert str(result.response) == "降级单轮答案"   # agent 抛错 → 降级 qa.retrieve
    assert result.source_nodes == ["n1"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -k other -q`
Expected: FAIL，`AttributeError`（`wf.qa_agent` 不存在 / `other_branch` 仍调 `qa.retrieve`）

- [ ] **Step 3: 改 doc_workflow.py**

(a) 顶部 import 区，在 `from core.workflow.qa_capability import (...)` 之后追加:

```python
from core.agent.qa_agent import QaAgent
```

(b) `__init__` 中，在 `self.qa = QaCapability(index_manager, llm, similarity_top_k, max_sub_queries)` 之后追加:

```python
        self.qa_agent = QaAgent(index_manager, llm, similarity_top_k, max_iterations=6)
```

(c) 把现有 `other_branch`:

```python
    @step
    async def other_branch(self, ctx: Context, ev: OtherEvent) -> FinalizeEvent:
        # TODO(step2): 换为有界 agent（自由调用工具 + max_steps/timeout/超界降级 +
        # grounding 约束 + 流式适配）。见 ARCHITECTURE.md「按可预测性配控制结构」。
        # v1：先把 other 从 fallback 独立出来、补「定义说检索不了却走单轮检索」的裂缝；
        # 行为暂与单轮检索同构，待第二步替换。
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

替换为:

```python
    @step
    async def other_branch(self, ctx: Context, ev: OtherEvent) -> FinalizeEvent:
        """高难度/开放问题 → 有界 agent 自由多轮检索探索。

        agent 异常 → 降级单轮检索，绝不让 other 比单轮更脆。
        """
        book_titles = await ctx.store.get("book_titles")
        try:
            answer, nodes = await self.qa_agent.run(
                ctx, ev.rewritten_query, book_titles
            )
        except Exception:
            answer, nodes = await self.qa.retrieve(
                ctx, ev.rewritten_query, book_titles
            )
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -q`
Expected: 全部 passed（原有 + 新增 2）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): other_branch 委托有界 QaAgent，异常降级单轮检索"
```

---

### Task 3: preprocess 的 other 判定放宽（积极）

**Files:**
- Modify: `core/workflow/query_preprocess.py`
- Test: `tests/test_query_preprocess.py`

- [ ] **Step 1: 追加测试**

在 `tests/test_query_preprocess.py` 末尾追加:

```python
async def test_run_classifies_other_for_complex_open_question():
    llm = FakeLLM([
        '{"category": "other", "rewritten_query": "综合对比 openclaw 与传统方案的架构取舍", "reason": "跨主题综合 + 开放权衡"}'
    ])
    result = await _pp(llm).run("综合对比 openclaw 与传统方案的架构取舍")
    assert result.category == "other"
    assert result.reason == "跨主题综合 + 开放权衡"
```

- [ ] **Step 2: 运行测试确认通过（解析层已支持 other，先确认绿）**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -k other -q`
Expected: PASS（`other` 已在 Literal 枚举内；本测试锚定解析正确，prompt 放宽不改变解析）

- [ ] **Step 3: 放宽 prompt 的 other 段（积极识别高难度）**

在 `_JUDGE_PROMPT` 中，把 other 那段:

```
- other（其他无法直接检索的情况，以上三类均不符合）
  返回 {"category":"other","rewritten_query": "处理后的 query", "reason":"无法直接检索的原因"}
```

替换为:

```
- other（高难度/开放复杂问题）：需要【跨多个主题综合、多步推理，或开放设计/权衡比较】，单轮检索难以一次答全，更适合多轮检索+推理逐步求解。
  特征：要综合多处证据、需要分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  倾向（积极）：当问题明显偏复杂综合、又不属于前三类（缺信息/角度不定/单纯并列罗列）时，判为 other 交由 agent 多轮处理；仅当问题其实能单轮检索集中命中时才回到 retrievable。
  返回 {"category":"other","rewritten_query": "处理后的 query", "reason":"判为高难度的原因，如'需跨主题综合+权衡比较'"}
```

并把判类优先级那行:

```
【不可以】归类的优先级：先判断信息是否不足，再判断问题是否角度不定，再判断问题是否需要拆分，均不符合则为other
```

替换为:

```
【不可以】归类的优先级：先判断信息是否不足(missing_info)，再判断是否角度不定(ambiguous)，再判断是否单纯并列罗列(pending_split)；若以上都不是、但问题需要跨主题综合/多步推理/开放权衡，则判 other（积极）；其余能单轮集中命中的归 retrievable。
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -q`
Expected: 全部 passed（prompt 文案放宽不影响解析层单测；分类质量靠真 LLM + 评测）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "feat(workflow): other 判定放宽为积极识别高难度（交有界 agent）"
```

---

### Task 4: 全量回归 + 编译 + 分层守卫

**Files:** 无（验证 only）

- [ ] **Step 1: 编译新增/改动模块**

Run: `.venv/Scripts/python.exe -m py_compile core/agent/qa_agent.py core/workflow/doc_workflow.py core/workflow/query_preprocess.py`
Expected: 无输出（成功）

- [ ] **Step 2: 跑新链路全部测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_agent.py tests/test_doc_workflow.py tests/test_query_preprocess.py tests/test_qa_capability.py tests/test_query_dimension.py tests/test_query_decompose.py tests/test_chapter_tree.py tests/test_intent_router.py tests/test_doc_query_service.py tests/test_chat_router.py -q`
Expected: 全部 passed

- [ ] **Step 3: 分层守卫**

Run: `.venv/Scripts/python.exe scripts/check_layering.py`
Expected: 通过（core/ 未 import api/；core 内部 agent↔workflow 互依不被守卫拦）

- [ ] **Step 4: 全量（跳过遗留 book_rag 采集错误）**

Run: `.venv/Scripts/python.exe -m pytest -q --continue-on-collection-errors`
Expected: 新增测试全 passed；仅 `test_book_rag_workflow.py` / `test_book_search_tool.py` 2 errors（遗留 book_rag 语法错，本计划不处理）

- [ ] **Step 5: Commit（如有未提交的验证性微调）**

```bash
git add -A
git commit -m "test(agent): other 有界 agent 全量回归通过"
```

---

## Self-Review Notes

- **Spec coverage:** QaAgent 载体 + 两工具 → Task 1；max_iterations=6 + generate → Task 1（构造 early_stopping_method + run 传 max_iterations）；流式桥接 ToolCall/ToolCallResult→RetrievalStart/Done + thought 不外露 → Task 1 run + 测试；source 工具收集回传 → Task 1 `_search` + run 返回；other_branch 委托 + 异常降级 → Task 2；判定积极 → Task 3。grounding → Task 1 system prompt。不做项（会话 memory/中间 thought 流式/token 预算/多工具）→ spec §8。
- **Type consistency:** `QaAgent.run(ctx, query, book_titles) -> tuple[str, list]`（与 `qa.retrieve`/`qa.split` 同形，other_branch 统一收 FinalizeEvent）；`_search(query) -> str`；事件 `RetrievalStartEvent(query)` / `RetrievalDoneEvent(count)` / `AnswerDeltaEvent(delta)` 复用 qa_capability 定义。
- **No placeholders:** 所有步骤含完整代码与确切命令。
- **风险点:** ① FunctionAgent 构造需合法 LLM——测试用 `MockLLM()`，run 测试替换 `qa.agent` 为 fake（避免真跑 LLM）；② `ToolCallResult` 构造需 `ToolOutput(content, tool_name, raw_input, raw_output)`，测试已给全字段；③ 依赖方向 core/agent→core/workflow.qa_capability（复用事件类），仍 core 内部、分层守卫只拦 api→core，不破；④ 判定积极的真实分类质量靠 LLM + 评测，单测只锚定解析层。
```
