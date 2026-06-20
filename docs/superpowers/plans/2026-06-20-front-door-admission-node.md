# Front-Door Admission Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 workflow 门口从"只分意图的 `IntentRouter`"升级成带会话记忆、能应付对话表层的 `FrontDoorAgent` 准入决策节点（4 个有界出口）。

**Architecture:** `DocQueryWorkflow` 仍是顶层编排器。`route` step 改调 `FrontDoorAgent`（单次 LLM 结构化决策），按 `action` 返回事件：`dispatch_qa`→现有 QA 流程、`dispatch_study_plan`→占位、`converse`/`clarify`→新 `DirectReplyEvent`（不检索）。库外（scope）仍由下游 `QueryPreprocessor` 判，门口不碰。

**Tech Stack:** Python 3.12 async，LlamaIndex Workflow（step 图），DeepSeek（`acomplete` + `json_object` 模式），Pydantic 校验，pytest（`pytest-asyncio` 已配，async 测试函数直接写）。

## Global Constraints

- **从项目根目录运行**，绝对导入（`from core.workflow.x import Y`），子模块内相对导入。
- **红线**（写进 FrontDoorAgent prompt）：承载知识的具体提问一律 `dispatch_qa`，门口绝不自答内容；门口不做检索。
- **scope 不在门口判**：内容问题一律 `dispatch_qa`，库外由下游 `QueryPreprocessor` 按 probe 召回判（本计划不动）。
- **失败降级绝不阻塞**：FrontDoorAgent 任何解析失败 → `dispatch_qa(原 query)`。
- **会话记忆只在 `finalize` 写**；门口只读 memory，工作态（clean_query/action）只走 `ctx.store`。
- LLM 调用传 `response_format={"type": "json_object"}`，**只按调用传，不塞全局 llm**。
- 中文注释可接受；所有 I/O `async/await`；函数签名带类型注解。
- 每次 commit 用**显式文件路径** `git add <file> ...`（仓库有其它未提交改动，禁止 `git add -A/.`）。
- 设计依据：`docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md`。

**执行前**：在分支上做（master 有无关未提交改动）。`git checkout -b feat/front-door-admission-node`。

---

### Task 1: FrontDoorAgent 决策单元（新模块 + 单测）

新模块自包含，不碰 workflow。`format_history`/`format_scope`/`_strip_fences` 在本模块自带副本（与既有按模块复制 `_strip_fences` 的约定一致；`IntentRouter` 暂保持不变并存，Task 4 再删）。

**Files:**
- Create: `core/workflow/front_door.py`
- Test: `tests/test_front_door.py`

**Interfaces:**
- Consumes: `core.workflow.summarizer.SUMMARY_MARKER`（已存在）。
- Produces:
  - `FrontDoorDecision`（dataclass）: `action: str`, `clean_query: str = ""`, `reply: str = ""`, `reason: str = ""`
  - `FrontDoorAgent.run(self, original: str, memory=None, book_titles: Optional[list[str]] = None) -> FrontDoorDecision`
  - `format_history(memory, max_msgs=MAX_HISTORY_MSGS) -> str`、`format_scope(book_titles) -> str`
  - action 取值：`"dispatch_qa" | "dispatch_study_plan" | "converse" | "clarify"`

- [ ] **Step 1: 写失败测试** — 创建 `tests/test_front_door.py`

```python
"""FrontDoorAgent（Layer 1 对话准入节点）单测。

mock LLM 控返回，验证：4 出口解析 / 净化输入透传 / 失败降级。
对话/意图判断质量依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md。
"""
from core.workflow.front_door import FrontDoorAgent, FrontDoorDecision, format_history


class _Resp:
    def __init__(self, text: str):
        self._t = text

    def __str__(self) -> str:
        return self._t


class FakeLLM:
    """按队列依次返回预设文本，并记录收到的 prompt（断言历史/scope 拼接）。"""

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


def _agent(llm):
    return FrontDoorAgent(llm)


async def test_dispatch_qa_carries_clean_query():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"什么是聚簇索引","reply":""}'])
    d = await _agent(llm).run("什么是聚簇索引啊")
    assert isinstance(d, FrontDoorDecision)
    assert d.action == "dispatch_qa"
    assert d.clean_query == "什么是聚簇索引"


async def test_dispatch_qa_resolves_coreference():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL索引的应用场景"}'])
    d = await _agent(llm).run("它的应用场景是什么")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "MySQL索引的应用场景"


async def test_dispatch_study_plan():
    llm = FakeLLM(['{"action":"dispatch_study_plan","clean_query":"为《Redis设计与实现》制定学习计划"}'])
    d = await _agent(llm).run("给我做份学Redis的计划")
    assert d.action == "dispatch_study_plan"
    assert d.clean_query == "为《Redis设计与实现》制定学习计划"


async def test_converse_carries_reply():
    llm = FakeLLM(['{"action":"converse","reply":"你好！我是文档知识库助手～"}'])
    d = await _agent(llm).run("你好")
    assert d.action == "converse"
    assert "知识库助手" in d.reply


async def test_clarify_carries_reply():
    llm = FakeLLM(['{"action":"clarify","reply":"你说的「那个」是指前面的聚簇索引还是锁？"}'])
    d = await _agent(llm).run("那个再讲讲")
    assert d.action == "clarify"
    assert "聚簇索引" in d.reply


async def test_history_passed_to_prompt():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL索引的应用场景"}'])
    memory = FakeMemory([_Msg("user", "MySQL索引有哪些"), _Msg("assistant", "B+树索引……")])
    await _agent(llm).run("它的应用场景是什么", memory)
    assert "MySQL索引有哪些" in llm.prompts[0]


async def test_selected_books_injected_into_prompt():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"《openclaw》讲了什么"}'])
    await _agent(llm).run("这本书讲了什么", None, book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]


async def test_parse_failure_degrades_to_dispatch_qa_original(caplog):
    import logging
    llm = FakeLLM(["这不是JSON"])
    with caplog.at_level(logging.WARNING):
        d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"
    assert any("front_door 解析失败" in r.getMessage() for r in caplog.records)


async def test_empty_content_degrades_to_dispatch_qa():
    llm = FakeLLM([""])
    d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"


async def test_invalid_action_degrades_to_dispatch_qa():
    llm = FakeLLM(['{"action":"do_magic","clean_query":"x"}'])
    d = await _agent(llm).run("讲讲数据库")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "讲讲数据库"


async def test_dispatch_qa_empty_clean_query_uses_original():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":""}'])
    d = await _agent(llm).run("什么是B+树")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "什么是B+树"


async def test_converse_empty_reply_gets_fallback():
    llm = FakeLLM(['{"action":"converse","reply":""}'])
    d = await _agent(llm).run("你好")
    assert d.action == "converse"
    assert d.reply  # 非空兜底


def test_format_history_keeps_summary_head_beyond_window():
    from core.workflow.summarizer import SUMMARY_MARKER
    msgs = [_Msg("user", f"{SUMMARY_MARKER}\n远期摘要内容")]
    msgs += [_Msg("user", f"q{i}") for i in range(10)]
    out = format_history(FakeMemory(msgs), max_msgs=3)
    assert "远期摘要内容" in out
    assert "q9" in out
    assert "q0" not in out


def test_format_history_without_summary_just_tail():
    msgs = [_Msg("user", f"q{i}") for i in range(10)]
    out = format_history(FakeMemory(msgs), max_msgs=3)
    assert "q9" in out and "q0" not in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_front_door.py -q`
Expected: FAIL（`ModuleNotFoundError: core.workflow.front_door`）

- [ ] **Step 3: 写实现** — 创建 `core/workflow/front_door.py`

```python
"""对话准入节点（Layer 1 门口）：净化 + 四出口决策。

把门口从"只分意图的 IntentRouter"升级成"带会话记忆、能应付对话表层的准入决策"。
职责：把【用户原始 query + 会话历史 + 选中的书】→ 一个有界决策：
- dispatch_qa：内容提问 → clean_query 下沉 QA 流程（红线：绝不在此自答内容）
- dispatch_study_plan：学习计划请求
- converse：寒暄/元问题/对上一轮的反馈不满 → reply 直接回复（不检索）
- clarify：指会话里某物但历史定不出所指 → reply 反问

单次 LLM 调用的结构化决策单元（非工具循环 agent）。沿用 IntentRouter/QueryPreprocessor
模式：注入 LLM、json_object、Pydantic 校验、失败降级、对外只暴露一个 run。

scope（库外）不在此判——内容问题一律 dispatch_qa，库外由下游 QueryPreprocessor 按
probe 召回证据判。设计见 docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md。
"""
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer

from core.workflow.summarizer import SUMMARY_MARKER

logger = logging.getLogger(__name__)

# 门口消指代只取最近几轮历史，别灌全量（省 token，也避免远古上下文误导）
MAX_HISTORY_MSGS = 6

# 兜底回复：converse/clarify 万一返回空 reply 时用，绝不给用户空答复
_FALLBACK_REPLY = "你好！我是文档知识库助手，可以问我已入库书籍/文档里的内容～"

# 【prompt 顺序约定】稳定指令在前、每轮变化输入（history/scope/query）在末尾，命中 DeepSeek 缓存。
# 用 .replace 注入，避免 JSON 示例花括号被 str.format 误当占位符。
_FRONT_DOOR_PROMPT = """你是知识库助手的对话门口。对下面的 query 做两件事：先净化，再决定交给哪个出口。

第一步 净化（产出 clean_query，自包含、规范）：
1) 指代消解：用【对话历史】+【当前选中的书】把"它/这个/上面说的/前面提到的/那个/这本书"等补全成不依赖上文、能独立成立的句子。无指代则不动。
2) 规范化：纠错别字/同音形近字、统一全半角、仅展开无歧义缩写（如 K8s→Kubernetes）。只改形式不改意图。
已自包含且规范则原样保留。

第二步 选出口（四选一，基于会话状态判断，不要自己回答任何知识内容）：
- dispatch_qa：对已入库书籍/文档内容的【具体知识提问】。把净化后的自包含问句放进 clean_query。
  铁律：凡承载知识的具体提问，哪怕你自己知道答案，也绝不在这里作答——一律 dispatch_qa 交检索系统按知识库回答。
- dispatch_study_plan：要求基于某本书生成学习计划/学习路线。clean_query 放净化后的请求。
- converse：寒暄/问候/致谢/闲聊、问你是谁或能做什么这类元问题，以及【对上一轮回答的反馈、质疑、不满、调侃】（如"你逗我呢""为什么答不了""不对吧"——参考对话历史里上一轮系统的回复来判断）。reply 放面向用户的自然回复；若上一轮是拒答/没答好而本轮是不满，先如实承认再引导。
- clarify：本轮明显在指会话里的某个东西，但你无法从历史中确定所指（落在很早、或有歧义）。reply 放一句自然反问，点明不明之处，能列候选就列。

判断本轮与上一轮的关系，以【对话历史】为准，别只看这句话的字面。

只返回 JSON，不要其它任何内容：
{"action":"dispatch_qa / dispatch_study_plan / converse / clarify","clean_query":"净化后的自包含 query（dispatch 时填）","reply":"面向用户的话（converse/clarify 时填）","reason":"简短理由"}

对话历史：
{history}

当前选中的书：{scope}

query：{query}"""


@dataclass
class FrontDoorDecision:
    """门口产出：action 决定 dispatch；dispatch_* 带 clean_query，converse/clarify 带 reply。"""

    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""


class FrontDoorDecisionModel(BaseModel):
    """LLM 判定的目标 schema（json_object 不保 schema，这步 Pydantic 校验才是约束）。

    action 用 Literal 锁枚举，非法值在 model_validate 阶段被拒、走降级。
    """

    action: Literal["dispatch_qa", "dispatch_study_plan", "converse", "clarify"]
    clean_query: str = Field(default="", description="dispatch_* 的自包含 query")
    reply: str = Field(default="", description="converse/clarify 面向用户的回复")
    reason: str = Field(default="", description="简短理由")


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def format_history(
    memory: Optional[ChatMemoryBuffer], max_msgs: int = MAX_HISTORY_MSGS
) -> str:
    """取最近几轮历史拼成文本，喂给门口做指代消解 + 对话判断。

    若首条是摘要消息（SUMMARY_MARKER 前缀），【永远保留】它再接最近 max_msgs 条——
    摘要承载被压缩掉的远期上下文，落窗口外被截断则压缩白做。
    """
    if memory is None:
        return ""
    msgs = memory.get()
    if not msgs:
        return ""
    head: list = []
    rest = msgs
    first = msgs[0]
    if first.content and str(first.content).startswith(SUMMARY_MARKER):
        head = [first]
        rest = msgs[1:]
    rest = rest[-max_msgs:]
    return "\n".join(f"{m.role}: {m.content}" for m in (head + rest))


def format_scope(book_titles: Optional[list[str]]) -> str:
    """把用户选中的书拼成文本，喂给门口消解"这本书"类指代。"""
    if not book_titles:
        return "（用户未选择特定书籍，范围为全部已入库书籍）"
    return "".join(f"《{t}》" for t in book_titles)


class FrontDoorAgent:
    """注入 LLM，对外只暴露一个 run。单次结构化决策，便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        original: str,
        memory: Optional[ChatMemoryBuffer] = None,
        book_titles: Optional[list[str]] = None,
    ) -> FrontDoorDecision:
        history = format_history(memory)
        scope = format_scope(book_titles)
        prompt = (
            _FRONT_DOOR_PROMPT.replace("{query}", original)
            .replace("{history}", history)
            .replace("{scope}", scope)
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            d = FrontDoorDecisionModel.model_validate_json(text)
            if d.action in ("dispatch_qa", "dispatch_study_plan"):
                clean = (d.clean_query or original).strip() or original
                logger.info(
                    "front_door: action=%s clean_query=%r", d.action, clean[:80]
                )
                return FrontDoorDecision(d.action, clean_query=clean, reason=d.reason)
            # converse / clarify：对话表层，直接回复（空 reply 兜底）
            reply = (d.reply or "").strip() or _FALLBACK_REPLY
            logger.info("front_door: action=%s", d.action)
            return FrontDoorDecision(d.action, reply=reply, reason=d.reason)
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）→ 降级 dispatch_qa + 原 query，绝不阻塞
            logger.warning("front_door 解析失败，降级 dispatch_qa + 原 query：%s", exc)
            return FrontDoorDecision("dispatch_qa", clean_query=original)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_front_door.py -q`
Expected: PASS（14 passed）

- [ ] **Step 5: 提交**

```bash
git add core/workflow/front_door.py tests/test_front_door.py
git commit -m "feat(workflow): 新增 FrontDoorAgent 对话准入决策单元

四出口（dispatch_qa/dispatch_study_plan/converse/clarify）+ 净化 + 失败降级。
暂未接入 workflow（Task 2）。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 接入 DocQueryWorkflow（route → FrontDoorAgent）

把 `route` step 改调 `FrontDoorAgent`，新增 `DirectReplyEvent` 承载 converse/clarify 回复，删掉 `ChitchatEvent`/`chitchat_branch`，并同步更新 `test_doc_workflow.py` 里所有门口 mock（`intent`→`action`）。

**Files:**
- Modify: `core/workflow/doc_workflow.py`
- Modify: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: `core.workflow.front_door.FrontDoorAgent`（Task 1）。
- Produces: `DocQueryWorkflow` 的 `route` 现返回 `PreprocessEvent | StudyPlanEvent | DirectReplyEvent`；`finalize` 结果 metadata 含 `action`（替代原 `intent`）+ `category`。

- [ ] **Step 1: 改实现 — `doc_workflow.py` 导入与构造**

把 import 与 `self.router` 替换：

将
```python
from core.workflow.intent_router import IntentRouter
```
改为
```python
from core.workflow.front_door import FrontDoorAgent
```
（`DirectReplyEvent` 定义在 `doc_workflow.py` 本文件内，见 Step 2，不需 import。）

将
```python
        self.router = IntentRouter(llm)
```
改为
```python
        self.front_door = FrontDoorAgent(llm)
```

- [ ] **Step 2: 改实现 — 事件定义**

删除 `ChitchatEvent` 类：
```python
class ChitchatEvent(Event):
    """intent=chitchat → 寒暄/闲聊，门口直接友好回应，不进检索。"""
```
新增 `DirectReplyEvent`（放在原 `ChitchatEvent` 位置）：
```python
class DirectReplyEvent(Event):
    """converse / clarify → 门口直接回复（不检索/不分类）。"""

    reply: str
    action: str = ""
```

- [ ] **Step 3: 改实现 — `route` step**

将整个 `route` 方法替换为：
```python
    # ── 门口准入决策：读会话记忆做净化 + 四出口决策，确定性 dispatch。 ──
    @step
    async def route(
        self, ctx: Context, ev: RouteEvent
    ) -> "PreprocessEvent | StudyPlanEvent | DirectReplyEvent":
        original = await ctx.store.get("original_query")
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        book_titles = await ctx.store.get("book_titles")

        decision = await self.front_door.run(original, memory, book_titles)

        # 工作态落 ctx：action 供观测；clean_query 是门口横切产物，绝不写会话记忆。
        await ctx.store.set("action", decision.action)

        if decision.action == "dispatch_study_plan":
            await ctx.store.set("clean_query", decision.clean_query)
            return StudyPlanEvent()
        if decision.action in ("converse", "clarify"):
            return DirectReplyEvent(reply=decision.reply, action=decision.action)
        # dispatch_qa（含降级）
        await ctx.store.set("clean_query", decision.clean_query)
        return PreprocessEvent()
```

- [ ] **Step 4: 改实现 — 分支：`chitchat_branch` → `direct_reply_branch`**

将
```python
    @step
    async def chitchat_branch(self, ctx: Context, ev: ChitchatEvent) -> FinalizeEvent:
        # 寒暄/闲聊：门口直接友好回应，不进 probe/检索（避免把"你好"当知识库查询）。
        return FinalizeEvent(
            answer="你好！我是文档知识库助手，可以问我已入库书籍/文档里的内容～",
            source_nodes=[],
        )
```
替换为
```python
    @step
    async def direct_reply_branch(
        self, ctx: Context, ev: DirectReplyEvent
    ) -> FinalizeEvent:
        # converse/clarify：门口已生成面向用户的回复，直接收尾，不进 probe/检索。
        return FinalizeEvent(answer=ev.reply, source_nodes=[])
```

- [ ] **Step 5: 改实现 — `finalize` metadata（intent → action）**

将
```python
        meta = {
            "category": await ctx.store.get("category", None),
            "intent": await ctx.store.get("intent", None),
        }
```
替换为
```python
        meta = {
            "category": await ctx.store.get("category", None),
            "action": await ctx.store.get("action", None),
        }
```

- [ ] **Step 6: 改测试 — `tests/test_doc_workflow.py` 门口 mock 全量 intent→action**

在 `tests/test_doc_workflow.py` 全文做以下精确替换（这些是每个测试的**第一条** FakeLLM 响应，即门口返回）：
- 所有 `'{"intent": "qa", "clean_query":` → `'{"action": "dispatch_qa", "clean_query":`
- 所有 `'{"intent": "study_plan", "clean_query":` → `'{"action": "dispatch_study_plan", "clean_query":`

涉及测试（确认覆盖）：`test_study_plan_intent_short_circuits_without_qa_preprocess`、`test_qa_intent_feeds_clean_query_and_scope_to_answer`、`test_route_passes_selected_books_to_router`、`test_qa_preprocess_consumes_clean_query_not_original`、`test_missing_info_clarifies_without_retrieval`、`test_missing_info_uses_natural_clarify_question`、`test_other_category_answers_via_dedicated_branch`、`test_missing_info_budget_exhausted_assumes_and_answers`、`test_other_dispatches_to_bounded_agent`、`test_preprocess_passes_book_titles_to_classify`、`test_flags_off_degrade_branches_to_single_retrieve`、`test_finalize_exposes_category_in_metadata`。

`test_router_parse_failure_defaults_to_qa_path` 不动（首条 `"这不是JSON"` 仍降级 dispatch_qa→preprocess，`llm.calls == 2` 仍成立）。

- [ ] **Step 7: 改测试 — finalize metadata 断言**

在 `test_finalize_exposes_category_in_metadata` 中，将
```python
    assert result.metadata.get("intent") == "qa"
```
替换为
```python
    assert result.metadata.get("action") == "dispatch_qa"
```

- [ ] **Step 8: 改测试 — chitchat → converse**

将 `test_chitchat_responds_without_retrieval_or_classify` 整个替换为：
```python
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
        return PreprocessResult("retrievable", clean_query)

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="你好", memory=FakeMemory())
    assert called["retrieve"] is False
    assert called["classify"] is False           # converse 门口拦截，不进 QA
    assert "知识库助手" in str(result.response)
    assert llm.calls == 1                         # 只有门口这一次
```

- [ ] **Step 9: 跑测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -q`
Expected: PASS（全绿；注：`test_out_of_scope_responds_without_retrieval_or_clarify` 此时仍断言旧话术，应仍 PASS——其门口 mock 是 `dispatch_qa` 格式？不是，它首条是 `{"intent": "qa", ...}`，已被 Step 6 规则替换为 `dispatch_qa`，out_of_scope 话术未改，仍 PASS）

- [ ] **Step 10: 提交**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): route 接入 FrontDoorAgent，converse/clarify 直接回复

route 改调 FrontDoorAgent，新增 DirectReplyEvent 取代写死的 ChitchatEvent；
finalize metadata intent→action。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: out_of_scope 分支改对话式转场

库外回复从机械话术改成更友好的转场。仅 `out_of_scope_branch` 文案 + 对应断言。

**Files:**
- Modify: `core/workflow/doc_workflow.py`
- Modify: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: 无新增。
- Produces: out_of_scope 答案文案变更（下游/前端不依赖其精确文本，仅评测 golden 的 out_of_scope 条目断言 outcome=empty、不比文本）。

本任务是文案变更（非新行为），断言从"精确等于旧话术"放宽为"语义包含"，与新文案一起改，跑通即可。

- [ ] **Step 1: 改测试 — 放宽断言为语义包含**

在 `test_out_of_scope_responds_without_retrieval_or_clarify` 中，将
```python
    assert str(result.response) == "知识库里暂无与该问题相关的内容。"  # 固定话术（精确）
```
替换为
```python
    # 对话式转场：友好告知库外 + 邀请换个问法（不再机械精确话术）
    resp = str(result.response)
    assert "知识库" in resp and ("暂" in resp or "没有" in resp or "未收录" in resp)
```
（注：该测试已断言 `called["retrieve"] is False`，无需重复。）

- [ ] **Step 2: 改实现 — `out_of_scope_branch` 文案**

将
```python
    @step
    async def out_of_scope_branch(self, ctx: Context, ev: OutOfScopeEvent) -> FinalizeEvent:
        # 库外：探测召回片段与问题主题无关 → 如实告知，不检索/不合成/不反问。
        return FinalizeEvent(
            answer="知识库里暂无与该问题相关的内容。", source_nodes=[]
        )
```
替换为
```python
    @step
    async def out_of_scope_branch(self, ctx: Context, ev: OutOfScopeEvent) -> FinalizeEvent:
        # 库外：探测召回片段与问题主题无关 → 对话式转场，友好告知 + 邀请换问法，不检索/不反问。
        return FinalizeEvent(
            answer=(
                "这个问题知识库里暂未收录相关内容，我没法基于现有资料回答。"
                "你可以换个已入库主题问我，或把问题换个角度再试试～"
            ),
            source_nodes=[],
        )
```

- [ ] **Step 3: 跑测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -q`
Expected: PASS（全绿）

- [ ] **Step 4: 提交**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): out_of_scope 改对话式转场文案

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 删除 IntentRouter（已无 importer）

`doc_workflow` 已不再用 `IntentRouter`（Task 2）。删除旧模块与其测试，更新两处过时注释引用。

**Files:**
- Delete: `core/workflow/intent_router.py`
- Delete: `tests/test_intent_router.py`
- Modify: `core/workflow/doc_query_service.py`（注释引用）
- Modify: `core/workflow/summarizer.py`（注释引用）

**Interfaces:** 无（纯删除 + 注释）。

- [ ] **Step 1: 确认无代码 importer**

Run: `python -m pytest -q --co 2>/dev/null; grep -rn "intent_router\|IntentRouter\|RouterResult" core/ api/ eval/ tests/ --include=*.py`
Expected: 仅 `tests/test_intent_router.py`（将删）+ `core/workflow/doc_query_service.py`/`summarizer.py` 的**注释**命中 `intent_router`；无 `from core.workflow.intent_router import` 的活代码（`doc_workflow` 已改）。若出现其它活引用，停下排查再删。

- [ ] **Step 2: 删除文件**

```bash
git rm core/workflow/intent_router.py tests/test_intent_router.py
```

- [ ] **Step 3: 更新过时注释 — `doc_query_service.py`**

将 `core/workflow/doc_query_service.py` 中
```python
        intent_router.format_history 会据该标记【永远保留】此头部。db_messages 应只传
```
改为
```python
        front_door.format_history 会据该标记【永远保留】此头部。db_messages 应只传
```

- [ ] **Step 4: 更新过时注释 — `summarizer.py`**

将 `core/workflow/summarizer.py` 中
```python
# 摘要消息在 memory 里的前缀标记：build_memory 用它前置摘要，intent_router.format_history
```
改为
```python
# 摘要消息在 memory 里的前缀标记：build_memory 用它前置摘要，front_door.format_history
```

- [ ] **Step 5: 跑全量测试确认无回归**

Run: `python -m pytest tests/ -q`
Expected: PASS（`test_intent_router.py` 已删；`test_front_door.py` + `test_doc_workflow.py` 绿；其余不受影响。注：`tests/test_eval_compare.py::test_build_sut_workflow_variant_returns_workflow_system` 是**既有**失败——变体名重命名未同步，与本计划无关）

- [ ] **Step 6: 提交**

```bash
git add core/workflow/doc_query_service.py core/workflow/summarizer.py
git commit -m "refactor(workflow): 删除被 FrontDoorAgent 取代的 IntentRouter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 验证（end-to-end，可选手动冒烟）

需 `.env` 的 `DEEPSEEK_API_KEY`。起服务后用真实多轮验证准入决策：

```bash
python -m uvicorn api.main:app --port 8000
```
- "你好" → 友好回复，不检索（日志无 RetrievalStart）。
- "什么是聚簇索引" → 正常走 qa（检索 + 合成）。
- 先问一道库外题（PostgreSQL）→ 对话式转场；紧接 "你逗我呢，你不是有 MySQL 书吗" → **converse**（不再被判 out_of_scope），承认上轮 + 引导。
- "它的应用场景是什么"（承接上文）→ dispatch_qa，clean_query 已消指代。

## 自查（spec 覆盖）

- 四出口决策 → Task 1（FrontDoorAgent）。
- converse/clarify 直接回复、替代 chitchat → Task 2（DirectReplyEvent）。
- dispatch 下沉、下游 qa 不动 → Task 2（route → PreprocessEvent）。
- out_of_scope 对话式转场 → Task 3。
- IntentRouter 取代/删除、helper 迁入 front_door → Task 1（helper 自带）+ Task 4（删除）。
- 不判 scope / 不做 preamble / 不做会话感知合成 / 不加结构化 last_outcome → 本计划范围外，未触碰。
- 评测口径不变 → golden 全 qa → dispatch_qa → preprocess，category 仍回流 metadata。
