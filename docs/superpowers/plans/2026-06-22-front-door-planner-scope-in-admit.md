# front_door 规划器 + scope 下沉 admit 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删掉拆分前整句收窄的 `ConversationScoper`，把拆分并入 front_door 成「规划器」，scope 改由每个子问题在 admit 阶段用自己干净的全库 probe 现挣——根治"多主体被收窄丢少数派主体的书"bug，并支持知识消歧与逐子问题 intent 路由。

**Architecture:** 自底向上、每步保持测试绿：① 先移除 scoper 预收窄（bug 当即修好）；② Admitter 增「从召回 nodes 算主导书 scope」能力（加性，不破坏）；③ front_door 升级产「路由计划」（拆分 + 逐子问题 QA/非QA 路由）；④ qa.answer 消费路由计划 + per-subq scope；⑤ 拆分加按需 probe 消歧；⑥ 清理溶解 `QuerySplitter`。

**Tech Stack:** Python 3.12 / asyncio / LlamaIndex Workflow / Pydantic（LLM 结构化输出校验）/ pytest（`pytest-asyncio` auto 模式，测试函数直接 `async def`，无需 decorator）。

## Global Constraints

- 所有判定单元沿用约定：注入 LLM、对外只暴露 `run`、`response_format={"type":"json_object"}`、Pydantic 校验、**失败一律优雅降级**（方向=放行/不拆/全库）、自带 `_strip_fences` 副本。
- 子模块内用相对导入（`from .admitter import ...`）；根目录/测试用绝对导入（`from core.workflow.xxx import ...`）。
- core 不依赖 api；守卫 `python scripts/check_layering.py`。
- 所有 I/O `async/await`；函数签名加类型注解；中文注释可接受。
- 测试用 mock LLM / mock probe，**只验解析/接线/降级/scope 计算，不验真 LLM 判断质量**。
- 测试运行从项目根目录：`python -m pytest tests/xxx.py -v`。
- 每个 Task 结束提交一次（frequent commits）。提交信息末尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- 设计依据：`docs/superpowers/specs/2026-06-22-front-door-planner-scope-in-admit-design.md`。

---

## File Structure

| 文件 | 责任 | 本计划动作 |
|---|---|---|
| `core/workflow/conversation_scoper.py` | 旧：拆分前整句收窄 | **Task 6 删除** |
| `tests/test_conversation_scoper.py` | scoper 单测 | **Task 6 删除** |
| `core/workflow/admitter.py` | 可答性判定 | Task 2：加 `scope` 字段 + 从 nodes 算主导书 |
| `tests/test_admitter.py` | admit 单测 | Task 2：加 scope 用例 |
| `core/workflow/front_door.py` | 门口决策 → 规划器 | Task 3：加 `RoutedSubQuery` + 拆分 + 逐子问题路由；Task 5：消歧 probe |
| `tests/test_front_door.py` | front_door 单测 | Task 3 / Task 5 |
| `core/workflow/qa_capability.py` | QA 检索合成编排 | Task 1（移 split 来源无关）/ Task 4：消费路由计划 + per-subq scope |
| `tests/test_qa_capability.py` | qa 单测 | Task 4 |
| `core/workflow/doc_workflow.py` | 顶层编排 | Task 1：去 scoper 预收窄；Task 4：route 传 sub_queries |
| `tests/test_doc_workflow.py` | 编排接线单测 | Task 1 / Task 4 |
| `core/workflow/query_splitter.py` | 纯文本拆分 | **Task 6 删除**（能力并入 front_door） |
| `tests/test_query_splitter.py` | splitter 单测 | **Task 6 删除** |

---

## Task 1: 移除 scoper 预收窄（bug 当即修好）

> 删 `doc_workflow` 里的 scoper 调用 + `scope_note`/`_scope_prefix`。删除后 `book_titles` 只剩用户手选（或 None=全库）；`qa.answer` 内部对每个子问题本就用 `book_titles`（None 时全库）probe，故多主体不再被整句预收窄——MySQL 子问题全库 probe 必命中 MySQL 书。**本任务即修复 bug**；scoper/QuerySplitter 文件本身留到 Task 6 删（先保持 import 不炸）。

**Files:**
- Modify: `core/workflow/doc_workflow.py`（`route` / `split_answer` / 删 `_scope_prefix` / 删 `self.scoper`）
- Test: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: `FrontDoorDecision`（现状，不变）、`qa.answer(ctx, clean_query, book_titles, probe)`（现状签名，Task 4 才改）。
- Produces: `route` 不再产 `scope_note`；`split_answer` 直接 `qa.answer(ctx, clean_query, book_titles, probe=self._probe)`；`DocQueryWorkflow` 不再有 `self.scoper`。

- [ ] **Step 1: 改测试——删 scoper 相关用例、改"全库不收窄"断言**

把 `tests/test_doc_workflow.py` 中以下用例**删除**（它们断言的是即将移除的行为）：
`test_scoper_constructed_with_probe_vector_retriever`、`test_scoper_narrows_book_titles_in_full_library`、`test_scoper_called_with_user_books_and_result_flows`、`test_scope_note_prepended_to_answer`、`test_no_scope_note_when_not_narrowed`。

`test_disable_scope_skips_scoper` 改为下方"全库直透"语义（不再有 scoper 可跳过）：

```python
async def test_full_library_not_narrowed_passes_none_book_titles():
    # 未手选书 → book_titles 全程为 None（全库），不再有任何预收窄
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲讲MySQL和openclaw的gateway"}'])
    wf = _wf(llm)

    captured = {}

    async def fake_answer(ctx, cq, bt, probe=True):
        captured["book_titles"] = bt
        return "答案", [], {"category": "multi"}
    wf.qa.answer = fake_answer

    await wf.run(query="讲讲MySQL和openclaw的gateway", memory=FakeMemory())
    assert captured["book_titles"] is None        # 全库，无预收窄
    assert not hasattr(wf, "scoper")              # scoper 已从编排移除
```

- [ ] **Step 2: 跑测试看新用例失败**

Run: `python -m pytest tests/test_doc_workflow.py::test_full_library_not_narrowed_passes_none_book_titles -v`
Expected: FAIL —— `assert not hasattr(wf, "scoper")` 失败（当前仍构造 `self.scoper`）。

- [ ] **Step 3: 改 `doc_workflow.py`——删 scoper 接线与 scope_note**

删除 `__init__` 里的 scoper 构造（约 `core/workflow/doc_workflow.py:125-128`）：

```python
        # 全库多轮作用域收窄：probe 复用 workflow 的 probe_retriever 名字（None→vector）
        self.scoper = ConversationScoper(
            index_manager, probe_retriever=make_retriever(probe_retriever)
        )
```

删除顶部 import：`from core.workflow.conversation_scoper import ConversationScoper`。

`route` 改为（去掉 `scope`/`scope_note` 分支，约 `:166-173`）：

```python
        # dispatch_qa（含降级）—— memory/book_titles 在 route 顶部已取
        await ctx.store.set("clean_query", decision.clean_query)
        return SplitAnswerEvent()
```

删除 `_scope_prefix` 方法（`:175-180`）。`split_answer` 改为（去掉 prefix，`:197-207`）：

```python
    @step
    async def split_answer(self, ctx: Context, ev: SplitAnswerEvent) -> FinalizeEvent:
        # dispatch_qa → 委托 QA capability 统一编排
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")
        answer, nodes, meta = await self.qa.answer(
            ctx, clean_query, book_titles, probe=self._probe
        )
        await ctx.store.set("qa_meta", meta)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

（`book_titles` 仍在 `start` 里从 `ev.book_titles` 存入 ctx，是用户手选；不再被 scoper 覆盖。）

- [ ] **Step 4: 跑全 workflow 测试**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: PASS（含新用例；converse/clarify/study_plan/reranker 注入等回归不变）。

- [ ] **Step 5: 提交**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "fix: 移除 scoper 拆分前整句预收窄，多主体不再丢少数派主体的书

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Admitter 从召回 nodes 算主导书 scope（加性）

> 把旧 `ConversationScoper._decide` 的"主导书多数票"逻辑搬进 `Admitter`：`run` 多收一个可选 `nodes`，非空时算 `scope` 填进 `AdmitVerdict.scope`。`passages`（喂 LLM 判 verdict）保持不变——verdict 与 scope 同源一次调用产出。纯加性，旧调用方不传 nodes 时 `scope=None`，行为不变。

**Files:**
- Modify: `core/workflow/admitter.py`
- Test: `tests/test_admitter.py`

**Interfaces:**
- Consumes: 召回 node 对象（`NodeWithScore`/`TextNode`），`node.metadata["book_title"]`。
- Produces:
  - `AdmitVerdict.scope: Optional[list[str]]`（默认 None）。
  - `Admitter.run(query: str, passages: list[str], nodes: Optional[list] = None) -> AdmitVerdict`。
  - `Admitter.__init__(llm, dominant_share=0.60, dominant_ratio=2.0, cover_share=0.80, max_books=2, min_count=2)`。

- [ ] **Step 1: 写失败测试——scope 计算各情形**

追加到 `tests/test_admitter.py`（复用文件顶部 `FakeLLM`/`_Resp`）：

```python
from llama_index.core.schema import NodeWithScore, TextNode


def _nodes(books):
    return [
        NodeWithScore(node=TextNode(text="x", id_=str(i), metadata={"book_title": b}))
        for i, b in enumerate(books)
    ]


async def test_scope_single_dominant_book():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("讲讲MySQL", ["片段"], nodes=_nodes(["MySQL"] * 6 + ["X"] * 2))
    assert v.verdict == "ok"
    assert v.scope == ["MySQL"]


async def test_scope_two_books_when_concept_spans():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("讲讲索引", ["片段"], nodes=_nodes(["A"] * 4 + ["B"] * 3 + ["C"] * 1))
    assert v.scope == ["A", "B"]


async def test_scope_none_when_diffuse():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"], nodes=_nodes(["A"] * 3 + ["B"] * 3 + ["C"] * 2))
    assert v.scope is None


async def test_scope_none_when_no_nodes():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"], nodes=[])
    assert v.scope is None


async def test_scope_none_when_nodes_arg_omitted():
    # 旧调用方不传 nodes → scope=None，行为不变（回归）
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"])
    assert v.scope is None


async def test_scope_none_on_verdict_parse_failure():
    # LLM 坏 → 降级 ok，scope 仍可从 nodes 算出（verdict 与 scope 解耦）
    llm = FakeLLM(["这不是JSON"])
    v = await _adm(llm).run("讲讲MySQL", ["片段"], nodes=_nodes(["MySQL"] * 6))
    assert v.verdict == "ok"
    assert v.scope == ["MySQL"]
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest tests/test_admitter.py::test_scope_single_dominant_book -v`
Expected: FAIL —— `AdmitVerdict` 无 `scope` 属性 / `run` 不接受 `nodes`。

- [ ] **Step 3: 改 `admitter.py`——加 scope 字段 + 计算逻辑**

`AdmitVerdict` 加字段：

```python
from typing import Literal, Optional

class AdmitVerdict(BaseModel):
    verdict: Literal["ok", "missing_info", "out_of_scope"] = "ok"
    reason: str = Field(default="", description="判定理由（日志/调试）")
    clarify_question: str = Field(default="", description="missing_info 专用：面向用户的自然反问句")
    scope: Optional[list[str]] = Field(default=None, description="从召回 nodes 算的主导书集合；None=不收窄/全库")
```

> 注意：`scope` 是代码侧填充字段、**不**由 LLM 产出。`model_validate_json` 解析 LLM JSON 时 LLM 不返回 scope → 取默认 None，随后代码覆盖。

`Admitter` 加阈值参数 + 两个 helper（搬自 `conversation_scoper.py`）：

```python
from collections import Counter


def _book_of(node) -> str:
    """从 NodeWithScore/TextNode 取 book_title（缺失返回空串）。"""
    meta = getattr(node, "metadata", None) or {}
    return meta.get("book_title") or ""


class Admitter:
    def __init__(
        self,
        llm: LLM,
        dominant_share: float = 0.60,
        dominant_ratio: float = 2.0,
        cover_share: float = 0.80,
        max_books: int = 2,
        min_count: int = 2,
    ):
        self.llm = llm
        self.dominant_share = dominant_share
        self.dominant_ratio = dominant_ratio
        self.cover_share = cover_share
        self.max_books = max_books
        self.min_count = min_count

    def _decide_scope(self, nodes: list) -> Optional[list[str]]:
        """命中 nodes → 主导书集合 or None（不收窄）。搬自旧 ConversationScoper._decide。"""
        titles = [t for t in (_book_of(n) for n in nodes) if t]
        if not titles:
            return None
        counts = Counter(titles).most_common()          # [(book, n), ...] 降序
        total = sum(n for _b, n in counts)
        top_book, top_n = counts[0]
        second_n = counts[1][1] if len(counts) > 1 else 0
        if (
            top_n / total >= self.dominant_share
            and top_n >= self.dominant_ratio * second_n
            and top_n >= self.min_count
        ):
            return [top_book]
        prefix: list[str] = []
        acc = 0
        for book, n in counts:
            if n < self.min_count:
                break
            prefix.append(book)
            acc += n
            if len(prefix) > self.max_books:
                return None
            if acc / total >= self.cover_share:
                tail = counts[len(prefix):]
                if all(tn < self.min_count for _tb, tn in tail):
                    return prefix
                return None
        return None
```

`run` 改为接收 `nodes` 并填 scope（无论 verdict 解析成败都算 scope）：

```python
    async def run(
        self, query: str, passages: list[str], nodes: Optional[list] = None
    ) -> AdmitVerdict:
        scope = self._decide_scope(nodes) if nodes else None
        passages_text = "\n---\n".join(passages) or "（无召回片段）"
        prompt = (
            _ADMIT_PROMPT.replace("{query}", query)
            .replace("{passages}", passages_text)
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            verdict = AdmitVerdict.model_validate_json(text)
            verdict.scope = scope
            logger.info("admit: verdict=%s scope=%s reason=%s", verdict.verdict, scope, verdict.reason)
            return verdict
        except Exception as exc:
            logger.warning("admit 解析失败，降级 ok（放行）：%s", exc)
            return AdmitVerdict(verdict="ok", scope=scope)
```

- [ ] **Step 4: 跑 admit 全测**

Run: `python -m pytest tests/test_admitter.py -v`
Expected: PASS（新 scope 用例 + 原有 verdict 用例全绿）。

- [ ] **Step 5: 提交**

```bash
git add core/workflow/admitter.py tests/test_admitter.py
git commit -m "feat: Admitter 从召回 nodes 算主导书 scope（搬入旧 scoper._decide）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: front_door 升级为规划器——拆分 + 逐子问题路由（无消歧）

> front_door 在 `dispatch_qa` 时，对 `clean_query` 做一次拆分 LLM 调用，产出 `[RoutedSubQuery]`：每个子问题带 `action`（`dispatch_qa` / `converse`）。非 QA 片段（童话/写代码/闲聊）→ `converse` + 婉拒 `reply`。本任务先做**单趟 LLM 拆分+路由**，消歧 probe 留 Task 5。非 dispatch_qa 的整句路径（converse/clarify/study_plan）不变，`sub_queries` 留空。

**Files:**
- Modify: `core/workflow/front_door.py`
- Test: `tests/test_front_door.py`

**Interfaces:**
- Consumes: `FrontDoorAgent.run(original, memory, book_titles)`（现状，不变签名）。
- Produces:
  - `RoutedSubQuery`（dataclass，front_door.py 内）：`query: str`、`action: str = "dispatch_qa"`、`reply: str = ""`。
  - `FrontDoorDecision.sub_queries: list = field(default_factory=list)`（元素为 `RoutedSubQuery`；仅 dispatch_qa 非空，≥1）。
  - `FrontDoorAgent._split_and_route(clean_query: str, book_titles) -> list[RoutedSubQuery]`（私有，Task 5 加 probe）。

- [ ] **Step 1: 写失败测试——拆分 + 路由 + 降级**

追加到 `tests/test_front_door.py`（沿用其 `FakeLLM`/`_Resp`/`FakeMemory`）：

```python
from core.workflow.front_door import RoutedSubQuery


async def test_dispatch_qa_splits_into_routed_subqueries():
    # 整句多主体 → 两个 dispatch_qa 子问题
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲MySQL锁和Redis持久化"}',  # 1st：门口决策
        '{"sub_queries":[{"query":"讲讲MySQL锁","action":"dispatch_qa"},'
        '{"query":"讲讲Redis持久化","action":"dispatch_qa"}]}',                 # 2nd：拆分+路由
    ])
    d = await FrontDoorAgent(llm).run("讲讲MySQL锁和Redis持久化", FakeMemory(), None)
    assert d.action == "dispatch_qa"
    assert [s.query for s in d.sub_queries] == ["讲讲MySQL锁", "讲讲Redis持久化"]
    assert all(s.action == "dispatch_qa" for s in d.sub_queries)


async def test_mixed_intent_routes_nonqa_to_converse():
    # "讲讲mysql和编个童话故事" → mysql=dispatch_qa，童话=converse 婉拒
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲mysql和编个童话故事"}',
        '{"sub_queries":[{"query":"讲讲MySQL","action":"dispatch_qa"},'
        '{"query":"编个童话故事","action":"converse","reply":"我是书籍知识库助手，没法编童话故事～"}]}',
    ])
    d = await FrontDoorAgent(llm).run("讲讲mysql和编个童话故事", FakeMemory(), None)
    qa = [s for s in d.sub_queries if s.action == "dispatch_qa"]
    conv = [s for s in d.sub_queries if s.action == "converse"]
    assert [s.query for s in qa] == ["讲讲MySQL"]
    assert conv and "书籍知识库助手" in conv[0].reply


async def test_single_subject_yields_one_subquery():
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"MySQL索引有哪些"}',
        '{"sub_queries":[{"query":"MySQL索引有哪些","action":"dispatch_qa"}]}',
    ])
    d = await FrontDoorAgent(llm).run("MySQL索引有哪些", FakeMemory(), None)
    assert len(d.sub_queries) == 1
    assert d.sub_queries[0].query == "MySQL索引有哪些"


async def test_split_llm_failure_degrades_to_single_subquery():
    # 拆分 LLM 坏 → 不拆，单元素 = clean_query（绝不阻塞）
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"MySQL索引有哪些"}',
        "这不是JSON",
    ])
    d = await FrontDoorAgent(llm).run("MySQL索引有哪些", FakeMemory(), None)
    assert len(d.sub_queries) == 1
    assert d.sub_queries[0].query == "MySQL索引有哪些"
    assert d.sub_queries[0].action == "dispatch_qa"


async def test_converse_path_has_no_subqueries():
    # 整句 converse（寒暄）→ 不拆，sub_queries 空，行为不变
    llm = FakeLLM(['{"action":"converse","reply":"你好！我是文档知识库助手～"}'])
    d = await FrontDoorAgent(llm).run("你好", FakeMemory(), None)
    assert d.action == "converse"
    assert d.sub_queries == []
    assert llm.calls == 1                    # 不触发拆分 LLM
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest tests/test_front_door.py::test_dispatch_qa_splits_into_routed_subqueries -v`
Expected: FAIL —— `RoutedSubQuery` 不存在 / `FrontDoorDecision` 无 `sub_queries`。

- [ ] **Step 3: 改 `front_door.py`——加 RoutedSubQuery + 拆分路由**

顶部 import 加 `field`：

```python
from dataclasses import dataclass, field
from typing import List, Literal, Optional
```

加 dataclass + 拆分 prompt + Pydantic schema（放在 `FrontDoorDecision` 附近）：

```python
@dataclass
class RoutedSubQuery:
    """拆分后的一个子问题及其路由出口。"""
    query: str
    action: str = "dispatch_qa"      # dispatch_qa | converse
    reply: str = ""                  # converse 婉拒文案（dispatch_qa 时空）


# 拆分 + 逐子问题路由（单趟，无消歧 probe；Task 5 加按需 probe）
_SPLIT_PROMPT = """你是知识库助手的子问题规划器。下面的 query 已净化（指代已消解、错别字已纠正）。做两件事：先降噪并按"多主体"拆分，再给每个子问题判一个出口。

第一步 拆分（只以"多主体"为判据，宁可不拆）：
【拆】同时满足：① 显式并列（A和B、A与B、A、B分别…）；② 两侧话题不同或带"分别/各自"标记；③ 无比较/对比/区别词；④ 无依赖。把每个子问题写成降噪后、能独立检索的自包含短句。
【不拆】（任一即整体作为单元素返回）：比较/评价（"A和B的区别/哪个好"）；多跳依赖；单主题广度发散（"怎么优化X"）；话题共享且无"分别"标记的居中句式（"讲讲A和B的缓存机制"）。
铁律：拆是不可逆的，拿不准一律不拆，返回单元素。

第二步 给每个子问题判出口（二选一）：
- dispatch_qa：对已入库书籍/文档内容的知识提问（默认）。reply 留空。
- converse：这个子问题根本不是知识提问——是闲聊、寒暄，或要求你创作/写代码/编故事等本系统不做的事。reply 放一句婉拒，如"我是书籍知识库助手，没法编童话故事～"。

只返回 JSON，不要其它任何内容：
{"sub_queries":[{"query":"降噪自包含子问题","action":"dispatch_qa 或 converse","reply":"converse 时的婉拒话，dispatch_qa 留空"}]}

query：{query}"""


class _RoutedSubQueryModel(BaseModel):
    query: str
    action: Literal["dispatch_qa", "converse"] = "dispatch_qa"
    reply: str = ""


class _SplitResultModel(BaseModel):
    sub_queries: List[_RoutedSubQueryModel] = Field(default_factory=list)
```

`FrontDoorDecision` 加字段：

```python
@dataclass
class FrontDoorDecision:
    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""
    tool: str = ""
    tool_filter: str = ""
    tool_count_only: bool = False
    disable_scope: bool = False
    sub_queries: list = field(default_factory=list)   # list[RoutedSubQuery]，仅 dispatch_qa 非空
```

在 `FrontDoorAgent.run` 的 dispatch_qa 分支（`:185-193`）补拆分，并加私有方法：

```python
            if d.action in ("dispatch_qa", "dispatch_study_plan"):
                clean = (d.clean_query or original).strip() or original
                logger.info("front_door: action=%s clean_query=%r", d.action, clean[:80])
                sub_queries = (
                    await self._split_and_route(clean, book_titles)
                    if d.action == "dispatch_qa" else []
                )
                return FrontDoorDecision(
                    d.action, clean_query=clean, reason=d.reason,
                    disable_scope=d.disable_scope, sub_queries=sub_queries,
                )
```

```python
    async def _split_and_route(
        self, clean_query: str, book_titles: Optional[list[str]]
    ) -> list[RoutedSubQuery]:
        """clean_query → ≥1 个 RoutedSubQuery。失败/空 → 单元素（不拆，dispatch_qa）。

        Task 5 在此加"按需 probe 消歧"；当前为单趟 LLM 拆分+路由。
        """
        fallback = [RoutedSubQuery(clean_query, "dispatch_qa")]
        prompt = _SPLIT_PROMPT.replace("{query}", clean_query)
        try:
            resp = await self.llm.acomplete(prompt, response_format={"type": "json_object"})
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            result = _SplitResultModel.model_validate_json(text)
            subs = [
                RoutedSubQuery(s.query.strip(), s.action, s.reply.strip())
                for s in result.sub_queries if s.query and s.query.strip()
            ]
            if not subs:
                raise ValueError("empty sub_queries")
            logger.info("front_door split: %d 子问题", len(subs))
            return subs
        except Exception as exc:
            logger.warning("front_door 拆分失败，降级不拆：%s", exc)
            return fallback
```

- [ ] **Step 4: 跑 front_door 全测**

Run: `python -m pytest tests/test_front_door.py -v`
Expected: PASS（新拆分/路由/降级用例 + 原有门口决策回归全绿）。

- [ ] **Step 5: 提交**

```bash
git add core/workflow/front_door.py tests/test_front_door.py
git commit -m "feat: front_door 升级为规划器——拆分 + 逐子问题 QA/非QA 路由

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: qa.answer 消费路由计划 + per-subq scope；route 传 sub_queries

> `qa.answer` 不再内部 `split_query`，改吃 front_door 产的 `[RoutedSubQuery]`。QA 子问题逐个：probe → admit（同时拿 verdict + scope）→ classify → execute（**在 admit 算出的 scope 内**）；converse 子问题直接用 `reply` 装饰；末尾合并 missing_info/out_of_scope。`doc_workflow.split_answer` 改传 `decision.sub_queries` + `disable_scope`。

**Files:**
- Modify: `core/workflow/qa_capability.py`（`answer` / `_decide_subq` / `_execute_subq` / `_SubDecision`）
- Modify: `core/workflow/doc_workflow.py`（`route` 存 sub_queries+disable_scope；`split_answer` 传参）
- Test: `tests/test_qa_capability.py`、`tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: `RoutedSubQuery`（Task 3）；`Admitter.run(q, passages, nodes) -> AdmitVerdict{verdict, scope}`（Task 2）。
- Produces:
  - `_SubDecision` 加 `scope: Optional[list[str]] = None`。
  - `qa.answer(ctx, sub_queries: list, book_titles, probe=True, disable_scope=False) -> tuple[str, list, dict]`（`sub_queries` 元素为 `RoutedSubQuery`）。
  - `_decide_subq(q, book_titles, probe=True, disable_scope=False) -> _SubDecision`（含 scope）。
  - `_execute_subq(ctx, q, category, scope) -> tuple[str, list]`（第四参数语义从"工作域 book_titles"明确为"该子问题 scope"）。

- [ ] **Step 1: 写失败测试——消费路由计划 + per-subq scope**

在 `tests/test_qa_capability.py` 追加（沿用其既有 fixtures；下方 `_make_qa` 指该文件构造 QaCapability 的既有 helper，按文件实际名替换）：

```python
from core.workflow.front_door import RoutedSubQuery


async def test_answer_consumes_routed_subqueries_per_subq_scope(qa_factory):
    # 两 QA 子问题，各自 admit 算出不同 scope，execute 收到各自 scope
    qa = qa_factory()
    exec_scopes = []

    async def fake_decide(q, book_titles, probe=True, disable_scope=False):
        from core.workflow.qa_capability import _SubDecision
        scope = ["MySQL"] if "MySQL" in q else ["openclaw"]
        return _SubDecision(q, "ok", category="simple", scope=scope)
    qa._decide_subq = fake_decide

    async def fake_execute(ctx, q, category, scope):
        exec_scopes.append((q, scope))
        return f"[{q}]", ["n"]
    qa._execute_subq = fake_execute

    routed = [
        RoutedSubQuery("讲讲MySQL", "dispatch_qa"),
        RoutedSubQuery("openclaw的gateway", "dispatch_qa"),
    ]
    ans, nodes, meta = await qa.answer(_FakeCtx(), routed, None)
    assert ("讲讲MySQL", ["MySQL"]) in exec_scopes
    assert ("openclaw的gateway", ["openclaw"]) in exec_scopes
    assert "[讲讲MySQL]" in ans and "[openclaw的gateway]" in ans
    assert meta["sub_count"] == 2


async def test_answer_converse_subquery_decorated_without_retrieval(qa_factory):
    qa = qa_factory()

    async def boom_execute(ctx, q, category, scope):
        raise AssertionError("converse 子问题不应进 execute")
    qa._execute_subq = boom_execute

    async def fake_decide(q, book_titles, probe=True, disable_scope=False):
        from core.workflow.qa_capability import _SubDecision
        return _SubDecision(q, "ok", category="simple", scope=["MySQL"])
    qa._decide_subq = fake_decide

    routed = [
        RoutedSubQuery("讲讲MySQL", "dispatch_qa"),
        RoutedSubQuery("编个童话故事", "converse", "我是书籍知识库助手，没法编童话～"),
    ]
    ans, _nodes, _meta = await qa.answer(_FakeCtx(), routed, None)
    assert "没法编童话" in ans
    assert "知识库里暂未收录" not in ans       # 不当失败检索拒


async def test_answer_single_subquery_no_section_heading(qa_factory):
    # 单子问题 → 裸答，无 "## 标题"（回归）
    qa = qa_factory()

    async def fake_decide(q, book_titles, probe=True, disable_scope=False):
        from core.workflow.qa_capability import _SubDecision
        return _SubDecision(q, "ok", category="simple", scope=None)
    qa._decide_subq = fake_decide

    async def fake_execute(ctx, q, category, scope):
        return "正文", ["n"]
    qa._execute_subq = fake_execute

    ans, _n, _m = await qa.answer(_FakeCtx(), [RoutedSubQuery("MySQL锁", "dispatch_qa")], None)
    assert ans == "正文"
    assert "## " not in ans


async def test_decide_subq_uses_admit_scope_in_full_library(qa_factory):
    # 全库（book_titles=None, 非 disable）→ scope 来自 admit
    qa = qa_factory()

    async def fake_probe(q, bt):
        return ["node"]
    qa._probe_retrieve = fake_probe
    qa._format_probe = lambda nodes, bt: "evidence"

    class _V:
        verdict, reason, clarify_question, scope = "ok", "", "", ["MySQL"]

    async def fake_admit(q, passages, nodes=None):
        return _V()
    qa.admitter.run = fake_admit

    async def fake_classify(q, evidence):
        from core.workflow.query_classifier import ClassifyResult
        return ClassifyResult("simple")
    qa.classifier.run = fake_classify

    d = await qa._decide_subq("讲讲MySQL", None, probe=True, disable_scope=False)
    assert d.verdict == "ok"
    assert d.scope == ["MySQL"]


async def test_decide_subq_user_books_override_scope(qa_factory):
    # 手选书 → scope=手选，admit 不自动算（nodes 不喂 admit 做 scope）
    qa = qa_factory()

    async def fake_probe(q, bt):
        return ["node"]
    qa._probe_retrieve = fake_probe
    qa._format_probe = lambda nodes, bt: "evidence"

    seen = {}

    class _V:
        verdict, reason, clarify_question, scope = "ok", "", "", None

    async def fake_admit(q, passages, nodes=None):
        seen["nodes"] = nodes
        return _V()
    qa.admitter.run = fake_admit
    qa.classifier.run = lambda q, e: _classify_simple()

    d = await qa._decide_subq("讲讲MySQL", ["高性能MySQL"], probe=True, disable_scope=False)
    assert d.scope == ["高性能MySQL"]
    assert seen["nodes"] is None        # 手选 → 不让 admit 自动算 scope
```

> 上方用到的 `qa_factory`/`_FakeCtx`/`_classify_simple`：若 `tests/test_qa_capability.py` 已有等价 fixture/helper（构造 `QaCapability`、假 `Context.write_event_to_stream`、假分类结果），直接复用其名；否则在文件顶部按现有风格补一份最小实现（`_FakeCtx` 只需 `write_event_to_stream(self, ev): pass`）。

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest tests/test_qa_capability.py::test_answer_consumes_routed_subqueries_per_subq_scope -v`
Expected: FAIL —— `answer` 仍按 `clean_query` 签名 / `_execute_subq` 仍按 `book_titles`。

- [ ] **Step 3: 改 `qa_capability.py`**

`_SubDecision` 加 scope：

```python
@dataclass
class _SubDecision:
    query: str
    verdict: str = "ok"
    category: str = ""
    reason: str = ""
    clarify_question: str = ""
    scope: Optional[list[str]] = None
```

`_decide_subq` 改（probe nodes → admit 拿 verdict+scope，手选/disable 时不自动算）：

```python
    async def _decide_subq(
        self, q: str, book_titles: Optional[list[str]],
        probe: bool = True, disable_scope: bool = False,
    ) -> "_SubDecision":
        nodes: list = []
        evidence = ""
        if probe:
            try:
                nodes = await self._probe_retrieve(q, book_titles)
                evidence = self._format_probe(nodes, book_titles)
            except Exception as exc:
                logger.warning("_decide_subq probe 失败，纯文本判定：%s", exc)
        # 手选书 / disable_scope → 不自动算 scope（手选即 scope；disable → 全库）
        auto = (not book_titles) and (not disable_scope)
        try:
            verdict = await self.admitter.run(q, [evidence], nodes=nodes if auto else None)
        except Exception as exc:
            logger.warning("_decide_subq admit 抛错，降级 ok：%s", exc)
            verdict = None
        scope = (verdict.scope if (auto and verdict is not None) else book_titles)
        if verdict is not None and verdict.verdict == "out_of_scope":
            return _SubDecision(q, "out_of_scope", reason=verdict.reason, scope=scope)
        if verdict is not None and verdict.verdict == "missing_info":
            return _SubDecision(
                q, "missing_info", reason=verdict.reason,
                clarify_question=verdict.clarify_question, scope=scope,
            )
        result = await self.classifier.run(q, evidence)
        return _SubDecision(q, "ok", category=result.category, reason=result.reason, scope=scope)
```

`_execute_subq` 第四参数改名 `scope`（其内部所有 `book_titles` 用法替换为 `scope`，逻辑不变）。签名与首行：

```python
    async def _execute_subq(
        self, ctx: Context, q: str, category: str, scope: Optional[list[str]]
    ) -> tuple[str, list]:
        """按 category 分派执行；在该子问题 scope 内检索。"""
        # ...（原体内 book_titles 全部改成 scope）
```

> 改名要点：`explain/assume/complex/simple` 各分支调用处把 `book_titles` 实参换成 `scope`；`retrieve(ctx, q, scope, ...)` 等同理。逻辑零改动，仅形参名与传递。

`split_query` 方法**删除**（拆分已上移 front_door；Task 6 删 QuerySplitter）。`__init__` 里 `self.splitter = QuerySplitter(llm)` 删除，import 删除。

`answer` 重写（吃路由计划）：

```python
    async def answer(
        self,
        ctx: Context,
        sub_queries: list,                 # list[RoutedSubQuery]
        book_titles: Optional[list[str]],
        probe: bool = True,
        disable_scope: bool = False,
    ) -> tuple[str, list, dict]:
        """消费 front_door 路由计划：QA 子问题逐个判定+执行，converse 子问题装饰，合并。"""
        qa_subs = [s for s in sub_queries if s.action == "dispatch_qa"]
        # 阶段一：并行判定 QA 子问题（无用户可见输出）
        decided = await asyncio.gather(
            *(self._decide_subq(s.query, book_titles, probe=probe, disable_scope=disable_scope)
              for s in qa_subs)
        )
        by_query = {d.query: d for d in decided}
        oks = [d for d in decided if d.verdict == "ok"]
        missing = [d for d in decided if d.verdict == "missing_info"]
        oos = [d for d in decided if d.verdict == "out_of_scope"]
        multi = len(sub_queries) > 1
        meta = {
            "categories": [d.category for d in oks],
            "sub_count": len(sub_queries),
            "category": (oks[0].category if oks else "out_of_scope")
            if len(sub_queries) == 1 else "multi",
        }

        parts: list[str] = []
        all_nodes: list = []
        produced_visible = False
        # 按路由计划顺序输出：converse 装饰 + ok 子问题执行
        for s in sub_queries:
            if s.action == "converse":
                text = s.reply or REFUSAL_FALLBACK
                if multi:
                    heading = f"\n## {s.query}\n"
                    ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
                    parts.append(heading)
                ctx.write_event_to_stream(AnswerDeltaEvent(delta=text))
                parts.append(text)
                produced_visible = True
                continue
            d = by_query.get(s.query)
            if d is None or d.verdict != "ok":
                continue
            if multi:
                heading = f"\n## {d.query}\n"
                ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
                parts.append(heading)
            ans, nodes = await self._execute_subq(ctx, d.query, d.category, d.scope)
            parts.append(ans)
            all_nodes.extend(nodes)
            produced_visible = True

        # 全无可见输出（无 ok、无 converse）：纯拒答/反问
        if not produced_visible:
            if missing:
                return (missing[0].clarify_question or REFUSAL_FALLBACK), [], meta
            return REFUSAL_TEXT, [], meta

        # 末尾装饰：out_of_scope / missing_info（仅多问题）
        tail = self._compose_tail(oos, missing) if multi else ""
        if tail:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=tail))
            parts.append(tail)
        return "".join(parts).strip(), all_nodes, meta
```

- [ ] **Step 4: 改 `doc_workflow.py`——route 存 sub_queries，split_answer 传参**

`route` 的 dispatch_qa 分支存 sub_queries + disable_scope：

```python
        await ctx.store.set("clean_query", decision.clean_query)
        await ctx.store.set("sub_queries", decision.sub_queries)
        await ctx.store.set("disable_scope", decision.disable_scope)
        return SplitAnswerEvent()
```

`split_answer` 改：

```python
    @step
    async def split_answer(self, ctx: Context, ev: SplitAnswerEvent) -> FinalizeEvent:
        sub_queries = await ctx.store.get("sub_queries")
        book_titles = await ctx.store.get("book_titles")
        disable_scope = await ctx.store.get("disable_scope", False)
        answer, nodes, meta = await self.qa.answer(
            ctx, sub_queries, book_titles, probe=self._probe, disable_scope=disable_scope
        )
        await ctx.store.set("qa_meta", meta)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

更新 `tests/test_doc_workflow.py` 里 stub 的 `fake_answer` 签名（现为 `(ctx, cq, bt, probe=True)`）改为 `(ctx, sub_queries, bt, probe=True, disable_scope=False)`，断言相应改成断 `sub_queries`（如 `test_dispatch_qa_goes_through_split_answer` 断 `len(sub_queries)>=1`）。`test_router_parse_failure_defaults_to_qa_path`：front_door 降级产 `sub_queries=[RoutedSubQuery(original)]`，断 `sub_queries[0].query == "B+树索引"`。

- [ ] **Step 5: 跑 qa + workflow 全测**

Run: `python -m pytest tests/test_qa_capability.py tests/test_doc_workflow.py -v`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add core/workflow/qa_capability.py core/workflow/doc_workflow.py tests/test_qa_capability.py tests/test_doc_workflow.py
git commit -m "feat: qa.answer 消费路由计划 + per-subq scope；route 传 sub_queries

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 拆分加按需 probe 消歧（两趟）

> front_door 拆分升级为"先判歧义，无歧义直出、有歧义才 probe 存疑挂法再重拆"。给 front_door 注入 `probe_retriever`（+ 复用 `index_manager`）。第一趟 LLM 多产 `ambiguous`(bool) + `probe_term`(存疑挂法，如 `mysql的gateway`)；歧义则 probe 一次 → 把召回喂第二趟 LLM 重拆。失败/超时退回第一趟结果。

**Files:**
- Modify: `core/workflow/front_door.py`（`_split_and_route` + 构造注入 probe_retriever + 第二趟 prompt）
- Modify: `core/workflow/doc_workflow.py`（构造 front_door 时传 probe_retriever）
- Test: `tests/test_front_door.py`

**Interfaces:**
- Consumes: `Retriever.retrieve(query, *, index_manager, book_titles, top_k)`（现有 probe_retriever 协议）。
- Produces: `FrontDoorAgent.__init__(llm, index_manager=None, probe_retriever=None, probe_k=8)`；`_split_and_route` 内部按需 probe。

- [ ] **Step 1: 写失败测试——无歧义零 probe / 有歧义 probe 后重拆**

追加到 `tests/test_front_door.py`（复用 `test_conversation_scoper.py` 风格的 `_FakeProbe`：按 book_title 序列造 node、记录 `last_query`）：

```python
from llama_index.core.schema import NodeWithScore, TextNode


class _FakeProbe:
    def __init__(self, books):
        self._books = books
        self.last_query = None
        self.calls = 0

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.calls += 1
        self.last_query = query
        return [
            NodeWithScore(node=TextNode(text="x", id_=str(i), metadata={"book_title": b}))
            for i, b in enumerate(self._books)
        ]


async def test_no_ambiguity_skips_probe():
    probe = _FakeProbe(["X"] * 8)
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲MySQL锁和Redis持久化"}',
        '{"ambiguous":false,"probe_term":"","sub_queries":'
        '[{"query":"讲讲MySQL锁","action":"dispatch_qa"},'
        '{"query":"讲讲Redis持久化","action":"dispatch_qa"}]}',
    ])
    d = await FrontDoorAgent(llm, probe_retriever=probe).run(
        "讲讲MySQL锁和Redis持久化", FakeMemory(), None
    )
    assert probe.calls == 0                       # 无歧义不 probe
    assert len(d.sub_queries) == 2


async def test_ambiguity_triggers_probe_then_resplit():
    # 趟1 标歧义、给存疑挂法 "MySQL的gateway"；probe 召回全 openclaw（MySQL 无 gateway）；
    # 趟2 据证据把 gateway 只挂 openclaw
    probe = _FakeProbe(["openclaw"] * 8)
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲MySQL和openclaw的gateway"}',
        '{"ambiguous":true,"probe_term":"MySQL的gateway","sub_queries":'
        '[{"query":"MySQL的gateway","action":"dispatch_qa"},'
        '{"query":"openclaw的gateway","action":"dispatch_qa"}]}',     # 趟1 暂定
        '{"sub_queries":[{"query":"讲讲MySQL","action":"dispatch_qa"},'
        '{"query":"openclaw的gateway","action":"dispatch_qa"}]}',     # 趟2 据证据修正
    ])
    d = await FrontDoorAgent(llm, probe_retriever=probe).run(
        "讲讲MySQL和openclaw的gateway", FakeMemory(), None
    )
    assert probe.calls == 1
    assert probe.last_query == "MySQL的gateway"        # 探的是存疑挂法
    assert [s.query for s in d.sub_queries] == ["讲讲MySQL", "openclaw的gateway"]


async def test_probe_failure_falls_back_to_pass1_split():
    probe = _BoomProbe()
    llm = FakeLLM([
        '{"action":"dispatch_qa","clean_query":"讲讲MySQL和openclaw的gateway"}',
        '{"ambiguous":true,"probe_term":"MySQL的gateway","sub_queries":'
        '[{"query":"MySQL的gateway","action":"dispatch_qa"},'
        '{"query":"openclaw的gateway","action":"dispatch_qa"}]}',
    ])
    d = await FrontDoorAgent(llm, probe_retriever=probe).run(
        "讲讲MySQL和openclaw的gateway", FakeMemory(), None
    )
    # probe 炸 → 用趟1 结果，不阻塞
    assert [s.query for s in d.sub_queries] == ["MySQL的gateway", "openclaw的gateway"]


class _BoomProbe:
    async def retrieve(self, *a, **k):
        raise RuntimeError("probe down")
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest tests/test_front_door.py::test_ambiguity_triggers_probe_then_resplit -v`
Expected: FAIL —— `FrontDoorAgent` 不接受 `probe_retriever` / 单趟拆分不读 `ambiguous`。

- [ ] **Step 3: 改 `front_door.py`——注入 probe + 两趟拆分**

`__init__` 加 probe：

```python
    def __init__(self, llm, index_manager=None, probe_retriever=None, probe_k: int = 8):
        self.llm = llm
        self.index_manager = index_manager
        self.probe_retriever = probe_retriever
        self.probe_k = probe_k
```

第一趟 prompt（`_SPLIT_PROMPT`）JSON 行替换为带 ambiguity 字段：

```python
# （在第二步说明后追加）
第三步 判歧义：若出现"A和B的X"这类修饰语作用域不定、且某挂法的存在性取决于知识（如 X 是否是 A 的概念），置 ambiguous=true，并把【存疑挂法】写进 probe_term（如 "MySQL的gateway"）；否则 ambiguous=false、probe_term 空。

只返回 JSON：
{"ambiguous":false,"probe_term":"","sub_queries":[{"query":"...","action":"dispatch_qa 或 converse","reply":""}]}
```

`_SplitResultModel` 加字段：

```python
class _SplitResultModel(BaseModel):
    ambiguous: bool = False
    probe_term: str = ""
    sub_queries: List[_RoutedSubQueryModel] = Field(default_factory=list)
```

加第二趟 prompt：

```python
_RESPLIT_PROMPT = """你之前在拆分"{query}"时，对"{probe_term}"这个挂法拿不准。下面是它在知识库的探测召回。据此判断该挂法是否成立，重新给出最终子问题拆分（消歧后、降噪自包含）。若召回里找不到该挂法主体的相关内容，说明该挂法不成立，应改挂到真正拥有该概念的主体。

探测召回：
{evidence}

只返回 JSON：
{"sub_queries":[{"query":"...","action":"dispatch_qa 或 converse","reply":""}]}"""
```

`_split_and_route` 升级为两趟：

```python
    def _to_subs(self, models) -> list[RoutedSubQuery]:
        subs = [
            RoutedSubQuery(m.query.strip(), m.action, m.reply.strip())
            for m in models if m.query and m.query.strip()
        ]
        return subs

    async def _split_and_route(
        self, clean_query: str, book_titles: Optional[list[str]]
    ) -> list[RoutedSubQuery]:
        fallback = [RoutedSubQuery(clean_query, "dispatch_qa")]
        # 趟1：拆分 + 路由 + 判歧义
        try:
            resp = await self.llm.acomplete(
                _SPLIT_PROMPT.replace("{query}", clean_query),
                response_format={"type": "json_object"},
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            r1 = _SplitResultModel.model_validate_json(text)
            subs1 = self._to_subs(r1.sub_queries)
            if not subs1:
                raise ValueError("empty sub_queries")
        except Exception as exc:
            logger.warning("front_door 拆分趟1失败，降级不拆：%s", exc)
            return fallback

        # 无歧义 / 无 probe 能力 → 直接用趟1
        if not (r1.ambiguous and r1.probe_term and self.probe_retriever is not None):
            return subs1

        # 趟2：probe 存疑挂法 → 据证据重拆（任何失败退回趟1）
        try:
            nodes = await self.probe_retriever.retrieve(
                r1.probe_term, index_manager=self.index_manager,
                book_titles=book_titles, top_k=self.probe_k,
            )
            evidence = "\n".join(
                f"《{(getattr(n, 'metadata', None) or {}).get('book_title', '?')}》 "
                + (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:120]
                for n in nodes[:8]
            ) or "（无召回）"
            resp2 = await self.llm.acomplete(
                _RESPLIT_PROMPT.replace("{query}", clean_query)
                .replace("{probe_term}", r1.probe_term)
                .replace("{evidence}", evidence),
                response_format={"type": "json_object"},
            )
            r2 = _SplitResultModel.model_validate_json(_strip_fences(str(resp2)).strip())
            subs2 = self._to_subs(r2.sub_queries)
            return subs2 or subs1
        except Exception as exc:
            logger.warning("front_door 消歧趟2失败，用趟1结果：%s", exc)
            return subs1
```

- [ ] **Step 4: 改 `doc_workflow.py`——构造 front_door 传 probe_retriever**

`__init__` 里（约 `:112`）：

```python
        self.front_door = FrontDoorAgent(
            llm, index_manager, probe_retriever=make_retriever(probe_retriever)
        )
```

- [ ] **Step 5: 跑 front_door + workflow 全测**

Run: `python -m pytest tests/test_front_door.py tests/test_doc_workflow.py -v`
Expected: PASS（Task 3 的单趟用例需相应补足第二趟 mock 响应——若 `ambiguous` 缺省 false，原单趟用例的 LLM 第二条响应即趟1，probe 不触发，仍绿）。

> 若 Task 3 的 `test_dispatch_qa_splits_into_routed_subqueries` 等用例的拆分 JSON 未含 `ambiguous` 字段：`_SplitResultModel.ambiguous` 默认 false，解析不受影响，无需改。

- [ ] **Step 6: 提交**

```bash
git add core/workflow/front_door.py core/workflow/doc_workflow.py tests/test_front_door.py
git commit -m "feat: 拆分加按需 probe 消歧——有歧义才探存疑挂法再重拆

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 清理——删除 ConversationScoper 与 QuerySplitter

> 两单元的能力已分别搬入 admit（scope）和 front_door（拆分）。删文件 + 删测试 + 删残留 import。`grep` 确认无引用残留。

**Files:**
- Delete: `core/workflow/conversation_scoper.py`、`tests/test_conversation_scoper.py`
- Delete: `core/workflow/query_splitter.py`、`tests/test_query_splitter.py`
- Modify: 任何残留 import（预期已在 Task 1/4 清掉，本步兜底）

**Interfaces:** 无新接口；纯删除。

- [ ] **Step 1: 确认无引用残留**

Run: `python -m pytest tests/ -q`（先确保删前全绿）
然后检索引用：
Run: `git grep -n "conversation_scoper\|ConversationScoper\|ScopeDecision\|query_splitter\|QuerySplitter\|split_query" -- "core" "api" "tests"`
Expected: 仅匹配到将删的两组文件自身；若 `core/` 下还有引用，回到对应 Task 修正（应已无）。

- [ ] **Step 2: 删除文件**

```bash
git rm core/workflow/conversation_scoper.py tests/test_conversation_scoper.py
git rm core/workflow/query_splitter.py tests/test_query_splitter.py
```

- [ ] **Step 3: 跑全量测试 + 分层守卫**

Run: `python -m pytest tests/ -q`
Expected: PASS（全绿，无 import 错误）。
Run: `python scripts/check_layering.py`
Expected: 分层无违例。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "refactor: 删除 ConversationScoper 与 QuerySplitter（能力已并入 admit/front_door）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review（计划对照 spec）

**Spec coverage**：
- 删 ConversationScoper → Task 1（去接线）+ Task 6（删文件）✓
- `_decide` 搬进 admit → Task 2 ✓
- 拆分并入 front_door / 升级规划器 → Task 3 ✓
- 带消歧的 probe（按需、两趟）→ Task 5 ✓
- 逐子问题 4 出口路由（最小集 QA/converse）→ Task 3（routing）+ Task 4（converse 装饰）✓
- scope 下沉 admit（per-subq）→ Task 2 + Task 4 ✓
- 手选硬锁 / disable_scope 重指向 → Task 4（`_decide_subq` 的 `auto` 判定）✓
- 删全局 scope_note / `_scope_prefix` → Task 1 ✓
- QuerySplitter 溶解 → Task 4（删用法）+ Task 6（删文件）✓
- 测试清单（复现回归、front_door 规划、admit 定 scope、混合 intent、scope 锁 execute、workflow 接线）→ 分散在各 Task Step 1 ✓

**Placeholder scan**：无 TBD/TODO；每个代码步给出完整改动代码。`qa_factory`/`_FakeCtx`/`_classify_simple` 已注明"复用现有或按现有风格补最小实现"，非占位。

**Type consistency**：`RoutedSubQuery{query,action,reply}` 在 Task 3 定义、Task 4 消费一致；`AdmitVerdict.scope` Task 2 定义、Task 4 读 `verdict.scope` 一致；`_SubDecision.scope` Task 4 定义并在 `answer` 读 `d.scope` 传 `_execute_subq(ctx,q,category,d.scope)` 一致；`_execute_subq` 第四参数 Task 4 统一为 `scope`；`qa.answer(ctx, sub_queries, book_titles, probe, disable_scope)` 在 Task 4 定义、doc_workflow.split_answer 与测试 stub 同步。

## 已知缺口（spec 已列，实现不覆盖，留后续）
- per-subq probe 放大（子问题多时检索次数线性增）。
- 手选多书时各子问题是否再细分 scope（本期手选即统一 scope）。
- 裸概念续问完全依赖 front_door 消指代（scoper 历史兜底已删）——需真实冷烟确认。
- 消歧 agent 判歧义本身一次 LLM 的成本/命中率，待冷烟观测。
- 真实冷烟（多主体库内+库外、修饰语歧义、混合 intent、续问）需 DEEPSEEK_API_KEY + 索引人读。
