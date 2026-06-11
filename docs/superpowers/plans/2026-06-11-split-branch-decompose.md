# split_branch 拆解-检索-汇总 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `doc_workflow.split_branch`（`pending_split` 分支）从占位改造为「定位 → 建骨架(结构主+内容辅) → 逐项检索 → map-reduce 汇总」流水线，让 LLM 不懂的概念也能基于文档结构与召回内容被拆解回答。

**Architecture:** 新增两个可独立测试的单元——`chapter_tree.py`（纯逻辑：章节编号解析 / 主导前缀 / 子节点）与 `query_decompose.py`（`QueryDecomposer`：注入 LLM，把"标题+召回正文"拆成 ≤N 个子查询）。`doc_workflow.split_branch` 编排：先一轮宽召回定位，用命中 chapter 的主导前缀从章节树取子节点标题做骨架，连同召回正文交 `QueryDecomposer` 产出子查询；每子查询各自检索+流式合成，按骨架拼成结构化答案。流式复用既有 SSE 词汇（一次 `RetrievalStart`/`RetrievalDone` + 每节标题 delta），前端零改动。

**Tech Stack:** Python 3.12，LlamaIndex Workflow，DeepSeek（`OpenAILike`，json_object + Pydantic 校验），pytest + pytest-asyncio。

参考 spec：`docs/superpowers/specs/2026-06-11-split-branch-decompose-design.md`

---

## File Structure

- **Create** `core/workflow/chapter_tree.py` — 纯函数：`chapter_number` / `unique_chapters` / `dominant_prefix` / `children`。无 LLM / chroma 依赖。
- **Create** `tests/test_chapter_tree.py` — 纯函数单测。
- **Create** `core/workflow/query_decompose.py` — `QueryDecomposer`（注入 LLM）+ `Decomposition` schema。
- **Create** `tests/test_query_decompose.py` — mock LLM 单测。
- **Modify** `core/workflow/doc_workflow.py` — `__init__` 加 `max_sub_queries` + `self.decomposer`；新增 `_book_chapters` helper；重写 `split_branch`。
- **Modify** `tests/test_doc_workflow.py` — 追加 split_branch 接线 + 降级 + 流式分节测试。

---

### Task 1: chapter_tree.py 纯逻辑

**Files:**
- Create: `core/workflow/chapter_tree.py`
- Test: `tests/test_chapter_tree.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_chapter_tree.py`:

```python
"""章节树纯逻辑单测：编号解析 / 去重 / 主导前缀 / 子节点。"""
from core.workflow.chapter_tree import (
    chapter_number,
    children,
    dominant_prefix,
    unique_chapters,
)


def test_chapter_number_parses_dotted_prefix():
    assert chapter_number("1.2.1  工具A") == (1, 2, 1)
    assert chapter_number("3.2 工具系统") == (3, 2)


def test_chapter_number_none_when_no_leading_number():
    assert chapter_number("(messages/prompt + tool") is None
    assert chapter_number("") is None
    assert chapter_number("前言") is None


def test_unique_chapters_filters_book_dedups_and_drops_empty():
    metas = [
        {"book_title": "A", "chapter": "1.1 X"},
        {"book_title": "A", "chapter": "1.1 X"},   # 重复
        {"book_title": "A", "chapter": ""},          # 空
        {"book_title": "B", "chapter": "9.9 别的书"},  # 别的书
        {"book_title": "A", "chapter": "1.2 Y"},
        None,                                          # 脏
    ]
    assert unique_chapters(metas, "A") == ["1.1 X", "1.2 Y"]


def test_dominant_prefix_returns_deepest_majority_prefix():
    # 多数命中聚在 3.2.* 下
    hits = ["3.2.1 a", "3.2.2 b", "3.2.3 c", "3.5 别处"]
    assert dominant_prefix(hits) == (3, 2)


def test_dominant_prefix_none_when_scattered():
    hits = ["1.1 a", "2.3 b", "5.1 c", "(噪声"]
    assert dominant_prefix(hits) is None


def test_children_under_prefix_returns_direct_children_sorted():
    all_ch = ["3.2 工具系统", "3.2.1 工具A", "3.2.2 工具B", "3.2.1.1 细节", "3.3 别节"]
    assert children(all_ch, (3, 2)) == ["3.2.1 工具A", "3.2.2 工具B"]


def test_children_none_prefix_returns_top_level_per_group():
    all_ch = ["1.1 概述", "1.2 细节", "2.1 进阶", "2.2 更深"]
    # 每个一级分组取最浅标题
    assert children(all_ch, None) == ["1.1 概述", "2.1 进阶"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_chapter_tree.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.workflow.chapter_tree'`

- [ ] **Step 3: 写实现**

创建 `core/workflow/chapter_tree.py`:

```python
"""章节树纯逻辑：把带编号的 chapter 标题（如 "1.2.1 工具A"）解析成可遍历的树。

供 split_branch 的"建骨架"用：从命中 chunk 的 chapter 推主导子树，再取该子树的
直接子节点标题作为拆解骨架。纯函数，无 LLM / chroma 依赖，便于单测。
"""
import re
from typing import Optional

_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def chapter_number(heading: str) -> Optional[tuple[int, ...]]:
    """'1.2.1  工具A' -> (1, 2, 1)；无前导编号 -> None。"""
    if not heading:
        return None
    m = _NUM_RE.match(heading)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def unique_chapters(metadatas: list, book_title: Optional[str] = None) -> list[str]:
    """从元数据列表抽该书去重 chapter（保序，去空）。book_title=None 不过滤。"""
    seen: set = set()
    out: list[str] = []
    for m in metadatas or []:
        if not m:
            continue
        if book_title is not None and m.get("book_title") != book_title:
            continue
        ch = (m.get("chapter") or "").strip()
        if not ch or ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def dominant_prefix(
    hit_chapters: list[str], threshold: float = 0.5
) -> Optional[tuple[int, ...]]:
    """命中 chapter 的主导编号前缀：被 >=threshold 命中共享的最深前缀。

    逐层下钻：每层取出现最多的前缀，若其占比 >= threshold*总数 且延续上层前缀，
    则继续下钻；否则停。命中散乱 / 无编号占多 -> None（信号：取顶层）。
    """
    paths = [p for p in (chapter_number(c) for c in hit_chapters) if p]
    if not paths:
        return None
    total = len(paths)
    prefix: tuple[int, ...] = ()
    depth = 1
    while True:
        counts: dict = {}
        for p in paths:
            if len(p) >= depth:
                key = p[:depth]
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            break
        best, cnt = max(counts.items(), key=lambda kv: kv[1])
        if cnt < threshold * total:
            break
        if prefix and best[: len(prefix)] != prefix:
            break
        prefix = best
        depth += 1
    return prefix or None


def children(all_chapters: list[str], prefix: Optional[tuple[int, ...]]) -> list[str]:
    """prefix 下的直接子节点标题（按编号排序）。

    prefix=None / () -> 顶层骨架：每个一级编号分组取 path 最浅的标题。
    """
    numbered = [(chapter_number(c), c) for c in all_chapters]
    numbered = [(p, c) for p, c in numbered if p]

    if not prefix:
        by_top: dict = {}
        for p, c in numbered:
            top = p[:1]
            if top not in by_top or len(p) < len(by_top[top][0]):
                by_top[top] = (p, c)
        return [c for _, (_, c) in sorted(by_top.items())]

    depth = len(prefix) + 1
    kids = [
        (p, c) for p, c in numbered if len(p) == depth and p[: len(prefix)] == prefix
    ]
    return [c for _, c in sorted(kids)]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_chapter_tree.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/chapter_tree.py tests/test_chapter_tree.py
git commit -m "feat(workflow): chapter_tree 章节编号树纯逻辑（骨架来源）"
```

---

### Task 2: QueryDecomposer（标题+正文 → 子查询）

**Files:**
- Create: `core/workflow/query_decompose.py`
- Test: `tests/test_query_decompose.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_query_decompose.py`:

```python
"""QueryDecomposer 单测：把"章节标题 + 召回正文"拆成 ≤N 个子查询。

mock LLM 控返回，验证：解析 / 上限裁剪 / 去空 / 失败降级为空 / prompt 带素材。
拆解质量本身依赖真 LLM，不在单测范围。
"""
from core.workflow.query_decompose import QueryDecomposer


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


def _dec(llm):
    return QueryDecomposer(llm)


async def test_run_parses_sub_queries():
    llm = FakeLLM(['{"sub_queries": ["工具A 是什么", "工具B 怎么用"]}'])
    subs = await _dec(llm).run("openclaw 的工具系统", ["3.2.1 工具A", "3.2.2 工具B"], ["正文片段"])
    assert subs == ["工具A 是什么", "工具B 怎么用"]


async def test_run_caps_at_max_items():
    payload = '{"sub_queries": ["a", "b", "c", "d", "e", "f", "g"]}'
    subs = await _dec(FakeLLM([payload])).run("q", [], ["p"], max_items=3)
    assert subs == ["a", "b", "c"]


async def test_run_drops_blank_sub_queries():
    llm = FakeLLM(['{"sub_queries": ["有效", "  ", ""]}'])
    subs = await _dec(llm).run("q", [], ["p"])
    assert subs == ["有效"]


async def test_run_returns_empty_on_parse_failure():
    subs = await _dec(FakeLLM(["这不是JSON"])).run("q", [], ["p"])
    assert subs == []


async def test_run_returns_empty_on_empty_content():
    subs = await _dec(FakeLLM([""])).run("q", [], ["p"])
    assert subs == []


async def test_run_prompt_includes_headings_and_passages():
    llm = FakeLLM(['{"sub_queries": ["x"]}'])
    await _dec(llm).run("openclaw 工具系统", ["3.2.1 工具A"], ["这是召回正文ZZZ"])
    assert "3.2.1 工具A" in llm.prompts[0]
    assert "这是召回正文ZZZ" in llm.prompts[0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_query_decompose.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.workflow.query_decompose'`

- [ ] **Step 3: 写实现**

创建 `core/workflow/query_decompose.py`:

```python
"""QueryDecomposer（QA capability 内部）：把宽问题拆成 ≤N 个可检索子查询。

结构主 + 内容辅：输入「章节子树标题（结构，保完整）+ 召回正文（内容，补正文级
实体、去噪）」，由 LLM 在【给定素材】上产出并列子查询——禁止编造文档里没有的
实体。LLM 在此是归纳器，不是知识源，故对训练时未见的概念同样有效。

解析失败 / 空 -> 返回空列表，由调用方（split_branch）降级为单轮检索，绝不阻塞。
"""
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_DECOMPOSE_PROMPT = """你是检索 query 拆解器。下面给出一个较宽的问题，以及与它相关的
【章节标题】和【召回正文片段】。请【只依据给定素材】把问题拆成若干并列的子查询，
每个子查询聚焦一个具体子项/小节/对比维度，便于逐个检索。

铁律：
- 子查询只能来自给定的章节标题或召回正文里真实出现的内容，严禁编造素材里没有的实体。
- 若问题是"对比/区别"，子查询应是各对比维度（如"X 与 Y 在适用场景上的区别"）。
- 子查询数量不超过 {max} 个；素材子项更多时，归并或取最重要的若干个。
- 每个子查询是能独立检索的完整短句。

问题：{query}

章节标题：
{headings}

召回正文片段：
{passages}

只返回 JSON，不要其他任何内容：
{"sub_queries": ["子查询1", "子查询2", ...]}"""


class Decomposition(BaseModel):
    """LLM 拆解结果的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryDecomposer:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控拆解输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        clean_query: str,
        headings: List[str],
        passages: List[str],
        max_items: int = 6,
    ) -> List[str]:
        prompt = (
            _DECOMPOSE_PROMPT.replace("{query}", clean_query)
            .replace("{headings}", "\n".join(f"- {h}" for h in headings) or "（无）")
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
            data = Decomposition.model_validate_json(text)
            subs = [s.strip() for s in data.sub_queries if s and s.strip()]
            return subs[:max_items]
        except Exception:
            # 任何失败都返回空，交由 split_branch 降级为单轮检索，绝不阻塞
            return []
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_query_decompose.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_decompose.py tests/test_query_decompose.py
git commit -m "feat(workflow): QueryDecomposer 结构主内容辅拆子查询"
```

---

### Task 3: 重写 split_branch 编排 + 流式

**Files:**
- Modify: `core/workflow/doc_workflow.py`（`__init__`、新增 `_book_chapters`、重写 `split_branch`）
- Test: `tests/test_doc_workflow.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_doc_workflow.py` 末尾追加（复用文件内已有的 `FakeLLM` / `FakeIndexManager` / `FakeCtx` / `_wf`）:

```python
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
```

同时在该测试文件的 `FakeCtx` 里补一个 `store` 假实现（split_branch 会 `await ctx.store.get("book_titles")`）。把现有 `FakeCtx` 替换为：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_doc_workflow.py -k split -q`
Expected: FAIL，`AttributeError`（`_book_chapters` 不存在 / `split_branch` 仍是旧整句检索，无分节）

- [ ] **Step 3: 改 doc_workflow.py**

(a) 顶部 import 区，在 `from core.workflow.query_preprocess import QueryPreprocessor` 之后追加:

```python
from core.workflow.chapter_tree import children, dominant_prefix, unique_chapters
from core.workflow.query_decompose import QueryDecomposer
```

(b) `__init__` 签名加 `max_sub_queries`，并初始化 decomposer。把:

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        **kw,
    ):
        super().__init__(**kw)
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        # 门口 Router（消指代 + 规范化 + 意图分类）与 QA 内部预处理（降噪 + 难度分类）
        # 各自独立、各自可测。
        self.router = IntentRouter(llm)
        self.preprocessor = QueryPreprocessor(llm)
```

替换为:

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        **kw,
    ):
        super().__init__(**kw)
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        # 门口 Router、QA 预处理、拆解器各自独立、各自可测。
        self.router = IntentRouter(llm)
        self.preprocessor = QueryPreprocessor(llm)
        self.decomposer = QueryDecomposer(llm)
```

(c) 把现有 `split_branch` 整个方法:

```python
    @step
    async def split_branch(self, ctx: Context, ev: SplitEvent) -> FinalizeEvent:
        # TODO: 真·拆子问题（再调一次 LLM）→ 多路检索 → 汇总；v1 先按整句直接检索。
        book_titles = await ctx.store.get("book_titles")

        answer, nodes = await self._answer(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

替换为:

```python
    @step
    async def split_branch(self, ctx: Context, ev: SplitEvent) -> FinalizeEvent:
        """定位 → 建骨架(结构主+内容辅) → 逐项检索 → map-reduce 汇总。

        拆解失败/空 → 降级为单轮检索+合成（等同 retrieve_branch），绝不阻塞。
        """
        book_titles = await ctx.store.get("book_titles")
        query = ev.rewritten_query

        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿命中 chunk 的 chapter
        located = await self._retrieve_nodes(query, book_titles)

        # 2) 建骨架：章节子树标题（结构）+ 召回正文（内容）→ 子查询
        all_chapters = self._book_chapters(book_titles)
        hit_chapters = [(n.metadata or {}).get("chapter", "") for n in located]
        prefix = dominant_prefix(hit_chapters)
        headings = children(all_chapters, prefix)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]
        sub_queries = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )

        # 降级：拆不出子查询 → 整句单轮合成
        if not sub_queries:
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return FinalizeEvent(
                    answer=f"在{scope}没有检索到与「{query}」相关的内容。",
                    source_nodes=[],
                )
            answer = await self._synthesize_stream(ctx, query, located)
            return FinalizeEvent(answer=answer, source_nodes=located)

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

(d) 在 helpers 区（`_make_filters` 之前）新增:

```python
    def _book_chapters(self, book_titles: Optional[list[str]]) -> list[str]:
        """取单一选定书的去重 chapter 列表；未选或多选 → []（结构缺失，倒向内容主导）。"""
        if not book_titles or len(book_titles) != 1:
            return []
        data = self.index_manager.chroma_collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        return unique_chapters(metas, book_titles[0])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -q`
Expected: 全部 passed（原有 + 新增 split 3 个）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): split_branch 拆解-检索-map-reduce 汇总流水线"
```

---

### Task 4: 全量回归 + 编译 + 分层守卫

**Files:** 无（验证 only）

- [ ] **Step 1: 编译新增/改动模块**

Run: `python -m py_compile core/workflow/chapter_tree.py core/workflow/query_decompose.py core/workflow/doc_workflow.py`
Expected: 无输出（成功）

- [ ] **Step 2: 跑新链路全部测试**

Run: `python -m pytest tests/test_chapter_tree.py tests/test_query_decompose.py tests/test_doc_workflow.py tests/test_intent_router.py tests/test_query_preprocess.py tests/test_doc_query_service.py tests/test_chat_router.py -q`
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
git commit -m "test(workflow): split_branch 全量回归通过"
```

---

## Self-Review Notes

- **Spec coverage:** 流水线四步 → Task 3 split_branch；结构主内容辅骨架 → Task 1（chapter_tree）+ Task 2（QueryDecomposer 同时吃 headings+passages）；定位命中分布判粒度 → `dominant_prefix`（Task 1）+ split_branch 用其结果（Task 3）；map-reduce 汇总 + 子项上限 → Task 3（`max_sub_queries`、逐节合成拼接）；流式一次 RetrievalDone + 每节标题 delta → Task 3 + 测试断言；降级（空骨架→单轮、无章节→内容主导、空命中→提示）→ Task 3 split_branch 分支 + `_book_chapters` 返回 []；错误处理（LLM 失败降级）→ Task 2 `run` 返回 []；测试三组 → Task 1/2/3。覆盖完整。对比类（entity×dimension）由 `QueryDecomposer` prompt 的"对比→维度"规则承载（Task 2），无需单独分支。
- **Type consistency:** `chapter_number -> tuple[int,...]|None`、`dominant_prefix -> tuple|None`、`children(all, prefix)`、`unique_chapters(metas, book)`、`QueryDecomposer.run(clean_query, headings, passages, max_items) -> list[str]`、`_book_chapters(book_titles) -> list[str]`、`split_branch` 复用既有 `_retrieve_nodes(query, book_titles)` / `_synthesize_stream(ctx, query, nodes)` / 事件 `RetrievalStartEvent(query)` / `RetrievalDoneEvent(count)` / `AnswerDeltaEvent(delta)`，各 Task 引用一致。
- **No placeholders:** 所有步骤含完整代码与确切命令。
- **风险点:** ① 真实节点对象的 `.metadata` / `.get_content()` 在 LlamaIndex 是 `NodeWithScore`——`.metadata` 与 `.get_content()` 均可用（与 `node_to_source_ref` 一致），测试用鸭子类型替身；② chroma `.get(include=["metadatas"])` 拉全量元数据，库大时略重，可后续改 `where={"book_title": ...}` 过滤，本期不优化。
