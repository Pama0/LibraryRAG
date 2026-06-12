# ambiguous → assume（归纳维度 + 声明角度）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `QaCapability.assume`（`ambiguous` 分支）从占位（等同单轮检索）改造为「定位 → LLM 归纳评判维度 → 声明所选角度 → 逐维度检索分节合成」，让"角度不定"的问题被透明地多角度回答。

**Architecture:** 新增 `DimensionExtractor`（注入 LLM，从「问题+召回正文」产 `(label, query)` 维度对，与 `QueryDecomposer` 同模式、独立可测）。把 split / assume 同构的「逐项检索→一次 RetrievalDone→可选声明→逐节合成拼接」抽成 `QaCapability._retrieve_and_reduce` helper；split 主路径改用它（行为不变，现有测试守护），assume 复用它并传入角度声明 preamble。流式复用既有 SSE 词汇，前端零改动。

**Tech Stack:** Python 3.12，LlamaIndex Workflow，DeepSeek（`OpenAILike`，json_object + Pydantic 校验），pytest + pytest-asyncio。

参考 spec：`docs/superpowers/specs/2026-06-12-ambiguous-assume-dimensions-design.md`

---

## File Structure

- **Create** `core/workflow/query_dimension.py` — `DimensionExtractor`（注入 LLM）+ `Dimension` / `DimensionSet` schema。
- **Create** `tests/test_query_dimension.py` — mock LLM 单测。
- **Modify** `core/workflow/qa_capability.py` — 新增 `_retrieve_and_reduce` helper；`split` 主路径改用它；`__init__` 加 `self.dimensioner`；重写 `assume`。
- **Modify** `tests/test_qa_capability.py` — split 测试保持绿（行为不变）；追加 assume 测试。

---

### Task 1: DimensionExtractor（问题+正文 → 评判维度）

**Files:**
- Create: `core/workflow/query_dimension.py`
- Test: `tests/test_query_dimension.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_query_dimension.py`:

```python
"""DimensionExtractor 单测：把"问题 + 召回正文"归纳成 ≤N 个评判维度。

mock LLM 控返回，验证：解析 / 上限裁剪 / 去空 / 失败降级为空 / prompt 带素材。
维度质量本身依赖真 LLM，不在单测范围。
"""
from core.workflow.query_dimension import DimensionExtractor


class _Resp:
    def __init__(self, text):
        self._t = text

    def __str__(self):
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


def _ext(llm):
    return DimensionExtractor(llm)


async def test_run_parses_dimensions():
    payload = '{"dimensions": [{"label": "读写性能", "query": "Redis 缓存读写性能"}, {"label": "一致性", "query": "Redis 缓存数据一致性"}]}'
    dims = await _ext(FakeLLM([payload])).run("Redis 做缓存好吗", ["正文片段"])
    assert [(d.label, d.query) for d in dims] == [
        ("读写性能", "Redis 缓存读写性能"),
        ("一致性", "Redis 缓存数据一致性"),
    ]


async def test_run_caps_at_max_items():
    items = ", ".join(
        '{"label": "L%d", "query": "Q%d"}' % (i, i) for i in range(7)
    )
    payload = '{"dimensions": [%s]}' % items
    dims = await _ext(FakeLLM([payload])).run("q", ["p"], max_items=3)
    assert [d.label for d in dims] == ["L0", "L1", "L2"]


async def test_run_drops_items_with_blank_label_or_query():
    payload = '{"dimensions": [{"label": "有效", "query": "有效检索"}, {"label": "", "query": "缺label"}, {"label": "缺query", "query": "  "}]}'
    dims = await _ext(FakeLLM([payload])).run("q", ["p"])
    assert [(d.label, d.query) for d in dims] == [("有效", "有效检索")]


async def test_run_returns_empty_on_parse_failure():
    dims = await _ext(FakeLLM(["这不是JSON"])).run("q", ["p"])
    assert dims == []


async def test_run_returns_empty_on_empty_content():
    dims = await _ext(FakeLLM([""])).run("q", ["p"])
    assert dims == []


async def test_run_prompt_includes_query_and_passages():
    payload = '{"dimensions": [{"label": "x", "query": "xq"}]}'
    llm = FakeLLM([payload])
    await _ext(llm).run("Redis 做缓存好吗", ["这是召回正文ZZZ"])
    assert "Redis 做缓存好吗" in llm.prompts[0]
    assert "这是召回正文ZZZ" in llm.prompts[0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_query_dimension.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.workflow.query_dimension'`

- [ ] **Step 3: 写实现**

创建 `core/workflow/query_dimension.py`:

```python
"""DimensionExtractor（QA capability 内部）：把"角度不定"的问题归纳成 ≤N 个评判维度。

输入「问题 + 召回正文」，由 LLM 在【给定素材】上产出并列的评判维度（如 性能 / 一致性 /
成本），每个维度含 label（维度名，进分节标题与角度声明）+ query（该维度的检索子查询）。
铁律：只依据召回正文，严禁编造素材里没有的维度。LLM 在此是归纳器，不是知识源。

解析失败 / 空 -> 返回空列表，由调用方（assume）降级为单轮检索，绝不阻塞。
"""
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_DIMENSION_PROMPT = """你是检索角度归纳器。下面给出一个"主题已具体、但回答角度/评判维度不定"的问题，以及与它相关的【召回正文片段】。请【只依据给定素材】归纳出若干个并列的评判维度，便于分角度回答。

铁律：
- 维度只能来自召回正文里真实出现的内容，严禁编造素材里没有的维度。
- 每个维度给 label（简短维度名，如"读写性能""数据一致性""成本"）和 query（该维度下能独立检索的完整子查询）。
- 维度数量不超过 {max} 个；取最重要、区分度最高的若干个。

问题：{query}

召回正文片段：
{passages}

只返回 JSON，不要其他任何内容：
{"dimensions": [{"label": "维度名1", "query": "子查询1"}, {"label": "维度名2", "query": "子查询2"}]}"""


class Dimension(BaseModel):
    """单个评判维度：label 进分节标题/角度声明，query 进检索。"""

    label: str = ""
    query: str = ""


class DimensionSet(BaseModel):
    """LLM 归纳结果的目标 schema（代码侧 Pydantic 校验）。"""

    dimensions: List[Dimension] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class DimensionExtractor:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控归纳输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        clean_query: str,
        passages: List[str],
        max_items: int = 6,
    ) -> List[Dimension]:
        prompt = (
            _DIMENSION_PROMPT.replace("{query}", clean_query)
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{max}", str(max_items))
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            data = DimensionSet.model_validate_json(text)
            dims = [
                Dimension(label=d.label.strip(), query=d.query.strip())
                for d in data.dimensions
                if d.label and d.label.strip() and d.query and d.query.strip()
            ]
            return dims[:max_items]
        except Exception:
            # 任何失败都返回空，交由 assume 降级为单轮检索，绝不阻塞
            return []
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_query_dimension.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_dimension.py tests/test_query_dimension.py
git commit -m "feat(workflow): DimensionExtractor 从问题+正文归纳评判维度"
```

---

### Task 2: 抽 `_retrieve_and_reduce` helper，split 改用它（行为不变）

**Files:**
- Modify: `core/workflow/qa_capability.py`（新增 helper；`split` 主路径改用它）
- Test: `tests/test_qa_capability.py`（不改，现有 split 测试守护行为不变）

- [ ] **Step 1: 新增 helper**

在 `core/workflow/qa_capability.py` 的 helpers 区，`_book_chapters` 之前插入：

```python
    # ── 公共流水线：逐项检索 → 一次 RetrievalDone →（可选声明）→ 逐节合成拼接 ──
    async def _retrieve_and_reduce(
        self,
        ctx: Context,
        sections: list[tuple[str, str]],
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """sections: [(分节标题, 检索/合成用子查询)]。split / assume 共用。

        - 先全检索（便于只发一次 RetrievalDone）。
        - preamble 非空 → 进入答案阶段后先推一个 AnswerDeltaEvent，并拼在答案最前。
        - 每节：推标题 delta → 流式合成该节（空命中给占位）。
        """
        retrieved: list = []
        all_nodes: list = []
        for _heading, sub_query in sections:
            ns = await self._retrieve_nodes(sub_query, book_titles)
            retrieved.append(ns)
            all_nodes.extend(ns)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        parts: list[str] = []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
            parts.append(preamble)
        for (heading, sub_query), ns in zip(sections, retrieved):
            h = f"\n## {heading}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=h))
            body = (
                await self._synthesize_stream(ctx, sub_query, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(h + body)
        return "".join(parts).strip(), all_nodes
```

- [ ] **Step 2: split 主路径改用 helper**

在 `split` 方法中，把第 3、4 步（`# 3) 逐项检索` 到方法结尾 `return answer, all_nodes`）整段：

```python
        # 3) 逐项检索（先全检索，便于只发一次 RetrievalDone）
        sections: list[tuple[str, list]] = []
        all_nodes: list = []
        for sq in sub_queries:
            ns = await self._retrieve_nodes(sq, book_titles)
            sections.append((sq, ns))
            all_nodes.extend(ns)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        # 4) 汇总（map-reduce）：每子项各自合成一段，按骨架拼接
        parts: list[str] = []
        for sq, ns in sections:
            heading = f"\n## {sq}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
            body = (
                await self._synthesize_stream(ctx, sq, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(heading + body)
        answer = "".join(parts).strip()
        return FinalizeEvent(answer=answer, source_nodes=all_nodes)
```

替换为（注意：split 的标题与检索子查询同为 `sq`）：

```python
        # 3-4) 逐项检索 + map-reduce 汇总（与 assume 共用同一 helper）
        sections = [(sq, sq) for sq in sub_queries]
        return await self._retrieve_and_reduce(ctx, sections, book_titles)
```

> 注：当前 `split` 末尾返回的是 `(answer, all_nodes)` tuple（QaCapability 版本，非 FinalizeEvent）。以你文件中实际的「逐项检索 + 汇总」整段为准替换为上面两行；`return` 形态对齐为 `return await self._retrieve_and_reduce(...)`（返回 `(answer, nodes)` tuple）。

- [ ] **Step 3: 跑 split 测试确认行为不变（仍绿）**

Run: `python -m pytest tests/test_qa_capability.py -k split -q`
Expected: 3 passed（`test_split_decomposes_and_concatenates_sections` / `test_split_emits_single_retrieval_done_and_section_headings` / `test_split_falls_back_to_single_retrieve_when_no_subqueries` 全过）

- [ ] **Step 4: 跑整文件确认无回归**

Run: `python -m pytest tests/test_qa_capability.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py
git commit -m "refactor(workflow): 抽 _retrieve_and_reduce，split 改用共用流水线"
```

---

### Task 3: 重写 assume（归纳维度 + 声明角度 + 逐维度分节）

**Files:**
- Modify: `core/workflow/qa_capability.py`（`__init__` 加 `self.dimensioner`；import；重写 `assume`）
- Test: `tests/test_qa_capability.py`（追加 assume 测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_qa_capability.py` 末尾追加（复用文件内已有的 `FakeLLM` / `FakeCtx` / `_qa`）:

```python
# ── assume：归纳维度 → 声明角度 → 逐维度检索分节 ──────────────────────
from core.workflow.query_dimension import Dimension  # noqa: E402


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py -k assume -q`
Expected: FAIL，`AttributeError`（`qa.dimensioner` 不存在 / `assume` 仍等同 retrieve，无角度声明与分节）

- [ ] **Step 3: 改 qa_capability.py**

(a) 顶部 import 区，在 `from core.workflow.query_decompose import QueryDecomposer` 之后追加:

```python
from core.workflow.query_dimension import DimensionExtractor
```

(b) `__init__` 末尾，在 `self.decomposer = QueryDecomposer(llm)` 之后追加:

```python
        self.dimensioner = DimensionExtractor(llm)
```

(c) 把现有 `assume` 方法:

```python
    # v1：声明角度逻辑后续补，先等同单轮检索
    async def assume(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        return await self.retrieve(ctx, query, book_titles)
```

替换为:

```python
    async def assume(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """角度不定：定位 → LLM 归纳评判维度 → 声明所选角度 → 逐维度检索分节合成。

        归纳不出维度 → 降级为单轮合成（复用已定位结果，绝不阻塞）。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿正文供归纳维度
        located = await self._retrieve_nodes(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 2) 归纳维度：从「问题 + 召回正文」产 (label, query) 维度对
        dimensions = await self.dimensioner.run(query, passages, self.max_sub_queries)

        # 降级：归纳不出维度 → 整句单轮合成
        if not dimensions:
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 3) 声明所选角度（透明 + 可纠偏）
        labels = "、".join(d.label for d in dimensions)
        preamble = f"「{query}」可以从以下角度来看：{labels}。下面分别说明——\n"

        # 4) 逐维度检索 + 分节合成（与 split 共用 helper）
        sections = [(d.label, d.query) for d in dimensions]
        return await self._retrieve_and_reduce(ctx, sections, book_titles, preamble)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_qa_capability.py -q`
Expected: 12 passed（原 9 + 新增 assume 3）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(workflow): assume 归纳维度+声明角度+逐维度分节（ambiguous 真实现）"
```

---

### Task 4: 全量回归 + 编译 + 分层守卫

**Files:** 无（验证 only）

- [ ] **Step 1: 编译新增/改动模块**

Run: `python -m py_compile core/workflow/query_dimension.py core/workflow/qa_capability.py`
Expected: 无输出（成功）

- [ ] **Step 2: 跑新链路全部测试**

Run: `python -m pytest tests/test_query_dimension.py tests/test_qa_capability.py tests/test_doc_workflow.py tests/test_query_decompose.py tests/test_chapter_tree.py tests/test_intent_router.py tests/test_query_preprocess.py tests/test_doc_query_service.py tests/test_chat_router.py -q`
Expected: 全部 passed

- [ ] **Step 3: 分层守卫**

Run: `python scripts/check_layering.py`
Expected: 通过（core/ 未 import api/）

- [ ] **Step 4: 全量（跳过遗留 book_rag 采集错误）**

Run: `python -m pytest -q --continue-on-collection-errors`
Expected: 新增测试全 passed；仅 `test_book_rag_workflow.py` / `test_book_search_tool.py` 2 errors（遗留 book_rag 语法错，本计划不处理）

- [ ] **Step 5: Commit（如有未提交的验证性微调）**

```bash
git add -A
git commit -m "test(workflow): assume 维度分节全量回归通过"
```

---

## Self-Review Notes

- **Spec coverage:** 流水线五步 → Task 3 assume（定位/归纳/声明/逐维度/降级）；维度来源「纯 LLM 从问题+正文归纳」→ Task 1 `DimensionExtractor`；与 split 复用「逐项检索→汇总」→ Task 2 `_retrieve_and_reduce` + split 改用；角度声明 preamble 透明 → Task 3 preamble + helper 的 preamble 分支 + 测试断言「声明在分节前」；流式一次 RetrievalDone + 声明 + 每节标题 → Task 2 helper + Task 3 测试；降级（空维度→单轮、空命中→提示）→ Task 3 分支；错误处理（LLM 失败降级）→ Task 1 `run` 返回 []。覆盖完整。L3 反问选项 / 章节锚定维度 → spec §7 明确不做。
- **Type consistency:** `DimensionExtractor.run(clean_query, passages, max_items) -> list[Dimension]`、`Dimension(label, query)`、`_retrieve_and_reduce(ctx, sections: list[tuple[str,str]], book_titles, preamble="") -> tuple[str,list]`、`assume(ctx, query, book_titles) -> tuple[str,list]`，各 Task 引用一致；复用既有 `_retrieve_nodes(sub_query, book_titles)` / `_synthesize_stream(ctx, query, nodes)` / 事件 `RetrievalStartEvent(query)` / `RetrievalDoneEvent(count)` / `AnswerDeltaEvent(delta)`。
- **No placeholders:** 所有步骤含完整代码与确切命令。
- **风险点:** ① Task 2 替换 split 内段落须以文件中**实际**「逐项检索+汇总」整段为准（QaCapability 版返回 tuple，非 FinalizeEvent），Step 2 注已点明；② 维度 query 与原问题相关性依赖 LLM，单测用替身，真实质量靠评测；③ `max_sub_queries` 复用为维度上限，语义上 split 子查询与 assume 维度共享同一上限（spec §7 YAGNI，不新增配置）。
```
