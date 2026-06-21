# 讲师级 explain（slice 2）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 explain 从"罗列、讲碎片"升到"讲师讲透一个概念"——骨架由教学维度词表 + 书的 TOC 定（自上而下、不从召回派生），合成从逐节孤立拼接换成一次整合教学写作。

**Architecture:** 三处改动。(1) `AnswerOutliner` 从"召回锚定、出 `list[str]`"改成"教学 schema 维度化、出 `list[Dimension]`，'组成'维吃 TOC 提示"。(2) `QaCapability` 新增 `_teach_synthesize`（教案 + 截断后的 pool → 讲师 prompt → 一次流式写）。(3) `qa.explain` 改成：宽召回 → 出教案(Dimension) → 每维度检索扇出 → 合并截断 pool → `_teach_synthesize`；保留 `EmptySkeleton → agent → 单轮` 降级阶梯。eval 加 explain golden 样例 + 锁定"explain 行得 faithfulness/answer_relevancy、不计入分类准确率"的回归测试。

**Tech Stack:** Python 3 / asyncio、LlamaIndex（`LLM.acomplete` + `LLM.astream_complete`、`Context.write_event_to_stream`）、Pydantic（`llama_index.core.bridge.pydantic`）、pytest（`pytest-asyncio`，`asyncio_mode=auto`）、ragas（eval 侧 faithfulness/answer_relevancy）。

## Global Constraints

- **工作目录**：项目根 = `C:\Users\11394\PycharmProjects\llmaLearn`（shell cwd 可能显示 `D:\AgentLearn\gpt-researcher`，那是 add-dir 的参考目录；所有 git/文件操作在 llmaLearn）。所有命令从项目根运行。
- **提交粒度**：用显式 `git add <文件路径>`，**绝不** `git add -A` / `git add .`（仓库里有大量本刀无关的未提交改动）。不跳 hooks、不跳签名。提交信息结尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **当前分支 master/main**：开工前先开 feature 分支（见 Task 1 Step 0）。
- **测试运行**：Windows 下中文 prompt 需 UTF-8，统一用 `PYTHONUTF8=1 python -m pytest ...`。
- **决策单元约定（沿用，不是缺陷）**：注入 LLM、对外只暴露 `run`、`response_format={"type":"json_object"}`、Pydantic 校验、失败优雅降级返回空、`_strip_fences` 每模块各自带一份副本——照抄，不要"消重"。
- **固定教学维度词表（六个，模型只能选用、不得自创）**：`是什么` / `作用` / `组成` / `原理` / `适用·边界` / `关系`。注意 `适用·边界` 中间是间隔号 `·`，全篇一字不差。
- **grounding 红线**：事实只来自检索 chunk；结构（维度/TOC）是安全教学元知识，不算编事实。运行时**不**做 faithfulness 拦截/再生成（只在 eval 侧度量）。
- **命名决策（spec 评审已定）**：教案条目**复用** `core.workflow.query_dimension.Dimension(label, query)`，不另立 `OutlineItem`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `core/workflow/qa_capability.py` | 修改 | 新增 `_teach_synthesize` / `_teach_tokens`（Task 1）；重写 `explain()`（Task 2） |
| `core/workflow/answer_outliner.py` | 修改 | 重写为教学维度化 outliner，出 `list[Dimension]`（Task 2） |
| `tests/test_qa_capability.py` | 修改 | 新增 `_teach_synthesize` 测试（Task 1）；重写 explain 测试（Task 2） |
| `tests/test_answer_outliner.py` | 重写 | 维度化输出 / TOC 进 prompt / 词表过滤 / 降级（Task 2） |
| `eval/dataset/golden.explain.jsonl` | 新建 | explain 类 golden 样例（无 `category`，宽题 + 具体题）（Task 3） |
| `tests/test_eval_run.py` | 修改 | 加"explain 行得分不被 REFUSE 短路、不计入分类"的回归测试（Task 3） |

`core/workflow/doc_workflow.py` 的 `explain_branch` / `ExplainEvent` / finalize 的 `intent` metadata **不动**——`qa.explain(ctx, query, book_titles)` 签名保持不变，`EmptySkeleton` 仍由 `explain_branch` 接住落 agent。

---

## Task 1: 给 QaCapability 加整合教学写作 `_teach_synthesize`

新增一个"教案 + pool → 一次整合写"的合成方法，纯加法，暂不接进 `explain()`（Task 2 才接线）。本任务结束后测试全绿、运行时行为不变。

**Files:**
- Modify: `core/workflow/qa_capability.py`（顶部加 `_TEACH_PROMPT` 常量；`QaCapability` 加 `_teach_tokens` + `_teach_synthesize` 两个方法）
- Test: `tests/test_qa_capability.py`（文件末尾追加一段 `_teach_synthesize` 测试）

**Interfaces:**
- Consumes：现有 `AnswerDeltaEvent`、`Dimension(label, query)`（来自 `core.workflow.query_dimension`）。
- Produces（Task 2 依赖）：
  - `QaCapability._teach_synthesize(self, ctx, query: str, outline: list[Dimension], pool: list) -> str`：组装讲师 prompt → 调 `_teach_tokens` 流式 → 每 token 发 `AnswerDeltaEvent` → 返回拼接全文。
  - `QaCapability._teach_tokens(self, prompt: str)`：async generator，`await self.llm.astream_complete(prompt)` 后逐 chunk `yield chunk.delta or ""`（单独成方法便于单测替身）。

- [ ] **Step 0: 开 feature 分支**

Run:
```bash
cd /c/Users/11394/PycharmProjects/llmaLearn
git checkout -b explain-slice2-lecturer
```
Expected: `Switched to a new branch 'explain-slice2-lecturer'`

- [ ] **Step 1: 写失败测试**

在 `tests/test_qa_capability.py` **末尾**追加（文件顶部已 `from core.workflow.query_dimension import Dimension`、已 `import pytest`，无需重复 import）：

```python
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
```

文件顶部 import 区需要 `AnswerDeltaEvent`。检查 `tests/test_qa_capability.py` 顶部是否已 import；若没有，在 `from core.workflow.qa_capability import QaCapability` 那行改为：
```python
from core.workflow.qa_capability import QaCapability, AnswerDeltaEvent
```
（若已存在对 `AnswerDeltaEvent` 的其它来源 import，则沿用现状，不重复。）

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py::test_teach_synthesize_builds_prompt_streams_once_and_emits_deltas -v`
Expected: FAIL，`AttributeError: 'QaCapability' object has no attribute '_teach_synthesize'`

- [ ] **Step 3: 加 `_TEACH_PROMPT` 常量**

在 `core/workflow/qa_capability.py` 顶部（`logger = logging.getLogger(__name__)` 之后、`class EmptySkeleton` 之前）加：

```python
# explain 整合教学写作的讲师 prompt（用 .replace 注入，避免 JSON/花括号被 str.format 误解）
_TEACH_PROMPT = """你是一位讲师，要把"{query}"讲清楚、讲透。下面给你一份讲解骨架（要讲的几个维度，按顺序）和一批从知识库检索到的资料片段。请据此写一篇连贯的讲解。

写作要求：
- 像老师讲课：先用一段话开场，点出这个主题整体是什么、要从哪几个方面讲；然后按骨架分节展开；最后一两句收束。
- 分节用轻量小标题：每个维度一个「## 维度名」小标题，只写骨架里列出的维度；节与节之间要有承接，不要各写各的。
- 【做减法、选高度】你是讲师不是资料堆砌机：只讲帮助理解主题的内容；资料片段里与当前维度无关的零碎细节（具体字段名、内部常量等）一律略去，除非它直接支撑某个维度的论点。

铁律（grounding，不可违反）：
- 事实只能来自下面的【资料片段】，严禁用你自己的训练知识或常识补充片段里没有的事实。
- 片段没覆盖到的维度，如实说"资料中未涉及"，不要编造。

讲解骨架（按此顺序分节）：
{plan}

资料片段：
{passages}"""
```

- [ ] **Step 4: 加 `_teach_tokens` + `_teach_synthesize` 方法**

在 `core/workflow/qa_capability.py` 的 `_synthesize_stream` 方法**之后**（类内最末）加：

```python
    async def _teach_tokens(self, prompt: str):
        """讲师整合写作的 token 源：直接对 prompt 流式 complete。单独成方法便于单测替身。"""
        handle = await self.llm.astream_complete(prompt)
        async for chunk in handle:
            yield chunk.delta or ""

    async def _teach_synthesize(
        self, ctx: Context, query: str, outline: list, pool: list
    ) -> str:
        """教案(维度顺序) + 截断后的 pool → 讲师 prompt → 一次流式整合写作。

        结构来自教案（教学先验/TOC，安全元知识）；事实只来自 pool 片段（prompt 立铁律）。
        逐 token 发 AnswerDeltaEvent，前端零改动。
        """
        plan = "\n".join(f"- {d.label}：{d.query}" for d in outline)
        passages = "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))
            for n in pool
        )
        prompt = (
            _TEACH_PROMPT.replace("{query}", query)
            .replace("{plan}", plan)
            .replace("{passages}", passages or "（无）")
        )
        parts: list[str] = []
        async for tok in self._teach_tokens(prompt):
            parts.append(tok)
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=tok))
        return "".join(parts)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py::test_teach_synthesize_builds_prompt_streams_once_and_emits_deltas -v`
Expected: PASS

- [ ] **Step 6: 跑全文件确认无回归**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py -q`
Expected: 全绿（原有 explain 测试仍用 stub 的 outliner，未受影响）

- [ ] **Step 7: 提交**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(explain): add _teach_synthesize for integrative teaching write

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: outliner 教学维度化 + explain 改走整合写作（原子，类型耦合）

`AnswerOutliner` 的返回类型从 `list[str]` 变 `list[Dimension]`，`explain()` 随之改造消费它——两者类型耦合，必须同一任务落地（否则中间某次提交 `explain()` 运行时类型不匹配）。本任务结束后运行时一致、测试全绿。

**Files:**
- Modify: `core/workflow/answer_outliner.py`（重写 prompt / `run` 签名 / schema 复用 `DimensionSet` / 词表过滤 / TOC 注入）
- Modify: `core/workflow/qa_capability.py`（重写 `explain()` 方法体，line 216-263）
- Test: `tests/test_answer_outliner.py`（整文件重写）
- Test: `tests/test_qa_capability.py`（重写 `test_explain_builds_sections_from_skeleton`、`test_explain_empty_skeleton_raises`）

**Interfaces:**
- Consumes（来自 Task 1）：`QaCapability._teach_synthesize(ctx, query, outline, pool) -> str`。
- Produces：
  - `AnswerOutliner.run(self, query: str, passages: list[str], toc_hint: list[str] | None = None, max_items: int = 8) -> list[Dimension]`：返回 label∈固定词表、query 非空的维度；解析失败/空 → `[]`。
  - `QaCapability.explain(self, ctx, query, book_titles) -> tuple[str, list]`：签名不变；内部改为教案驱动 + 整合写；空教案仍 `raise EmptySkeleton(query)`。

### Part A — 重写 AnswerOutliner

- [ ] **Step 1: 重写 outliner 测试（失败）**

整体覆盖 `tests/test_answer_outliner.py` 为：

```python
"""AnswerOutliner（教学维度化列骨架）单测。mock LLM 控返回，验解析/词表过滤/TOC/降级。"""
from core.workflow.answer_outliner import AnswerOutliner
from core.workflow.query_dimension import Dimension


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_outline_returns_dimensions_from_vocab():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"什么是MySQL"},'
                   '{"label":"组成","query":"MySQL由哪些部分组成"}]}'])
    dims = await AnswerOutliner(llm).run("MySQL基础知识", ["片段1", "片段2"])
    assert dims == [Dimension(label="是什么", query="什么是MySQL"),
                    Dimension(label="组成", query="MySQL由哪些部分组成")]


async def test_outline_drops_labels_not_in_vocab():
    # FIL_PAGE_UNDO_LOG 这种碎细节被模型当 label → 必须被词表过滤丢掉
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"什么是X"},'
                   '{"label":"FIL_PAGE_UNDO_LOG","query":"FIL_PAGE_UNDO_LOG细节"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == [Dimension(label="是什么", query="什么是X")]


async def test_outline_drops_dimension_with_empty_query():
    llm = FakeLLM(['{"dimensions":[{"label":"作用","query":""},'
                   '{"label":"原理","query":"X的原理"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == [Dimension(label="原理", query="X的原理")]


async def test_outline_atomic_single_dimension():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"脏读的定义"}]}'])
    dims = await AnswerOutliner(llm).run("什么是脏读", ["片段"])
    assert dims == [Dimension(label="是什么", query="脏读的定义")]


async def test_outline_passages_passed_to_prompt():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"x"}]}'])
    await AnswerOutliner(llm).run("讲讲X", ["关键片段ABC"])
    assert "关键片段ABC" in llm.prompts[0]


async def test_outline_toc_hint_passed_to_prompt():
    llm = FakeLLM(['{"dimensions":[{"label":"组成","query":"x"}]}'])
    await AnswerOutliner(llm).run("讲讲X", ["片段"], toc_hint=["第1章 索引", "第2章 事务"])
    assert "第1章 索引" in llm.prompts[0] and "第2章 事务" in llm.prompts[0]


async def test_outline_respects_max_items():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"a"},{"label":"作用","query":"b"},'
                   '{"label":"组成","query":"c"},{"label":"原理","query":"d"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"], max_items=2)
    assert [d.label for d in dims] == ["是什么", "作用"]


async def test_outline_empty_on_parse_failure():
    llm = FakeLLM(["这不是JSON"])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == []          # 空 → explain 将落 agent 兜底


async def test_outline_empty_on_empty_list():
    llm = FakeLLM(['{"dimensions":[]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_answer_outliner.py -q`
Expected: FAIL（`run` 仍返回 `list[str]`，断言 Dimension 不匹配；`toc_hint` 参数不存在）

- [ ] **Step 3: 重写 `core/workflow/answer_outliner.py`**

整体覆盖为：

```python
"""AnswerOutliner：据【教学维度词表 + 书的 TOC】给"讲清楚"问题定讲解骨架。

explain 专用。结构【自上而下】来自固定教学维度词表（是什么/作用/组成/原理/适用·边界/关系）
+ 书的目录章节（撑"组成"高度），【绝不从召回派生】——召回片段只用来判这个主题该讲哪几维、
每维填什么子查询，碎细节（FIL_PAGE_UNDO_LOG 之类）不会被提成顶层小节。
出 list[Dimension]（label∈词表、query=检索子查询）。空/失败 → []，由 qa.explain 落 agent 兜底。
设计见 docs/superpowers/specs/2026-06-21-explain-lecturer-synthesis-design.md。
"""
import logging
from typing import List, Optional

from llama_index.core.llms import LLM

from core.workflow.query_dimension import Dimension, DimensionSet

logger = logging.getLogger(__name__)

# 固定教学维度词表：模型只能选用，不得自创；不在词表里的 label 一律过滤掉。
_ALLOWED_LABELS = {"是什么", "作用", "组成", "原理", "适用·边界", "关系"}

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_OUTLINE_PROMPT = """你是讲师备课助手。下面给出一个用户想"讲清楚/讲透"的问题、知识库里宽召回到的相关片段，以及（若有）这本书的目录章节。请像备课的老师一样，【从固定教学维度词表里】挑选这个问题该讲的几个维度，定出讲解骨架。

固定教学维度词表（label 只能原样取自下面，不得自创，更不得把召回片段里的碎细节当成维度）：
- 是什么：概念的定义与定位
- 作用：解决什么问题、为什么需要（动机先行）
- 组成：由哪些部件/子结构构成
- 原理：部件怎么协作、工作机制
- 适用·边界：什么场景用、何时当心
- 关系：与相邻概念的联系或对比

定维度规则：
- 高度由问题决定：宽/入门问题（如"讲懂MySQL"）选靠前维度（是什么/作用/组成）、停在高处别下钻到部件内部细节；具体/深问题（如"MVCC怎么实现"）聚焦被问那一维（原理）下钻、前置维度一句带过。
- "组成"维度优先参考下面给出的【目录章节】——书的顶层章节就是部件高度；没给目录时按通用教学常识定组成。
- 每个维度配一个能独立检索的 query（含问题的主体技术实体，别只写"作用"这种裸词）。
- 维度数量自适应：原子概念 1~2 个即可，宽主题最多 {max} 个；按上面词表顺序排列。

只返回 JSON，不要其它任何内容：
{"dimensions":[{"label":"是什么","query":"……"},{"label":"组成","query":"……"}]}

问题：{query}

目录章节（可能为空）：
{toc}

召回片段：
{passages}"""


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class AnswerOutliner:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        query: str,
        passages: List[str],
        toc_hint: Optional[List[str]] = None,
        max_items: int = 8,
    ) -> List[Dimension]:
        toc_text = "、".join(toc_hint) if toc_hint else "（无干净目录，按通用教学常识定"组成"）"
        prompt = (
            _OUTLINE_PROMPT.replace("{query}", query)
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{toc}", toc_text)
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
                if d.label.strip() in _ALLOWED_LABELS and d.query and d.query.strip()
            ][:max_items]
            logger.info(
                "outline: 列出 %d 个教学维度：%s",
                len(dims), " | ".join(d.label for d in dims),
            )
            return dims
        except Exception as exc:
            logger.warning("outline 解析失败，返回空（explain 将落 agent 兜底）：%s", exc)
            return []
```

> 注：prompt 字符串里 `定"组成"` 用了中文引号，避免与 Python 字符串定界冲突；`_ALLOWED_LABELS` 用集合做 O(1) 过滤。

- [ ] **Step 4: 跑 outliner 测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_answer_outliner.py -q`
Expected: 全绿

### Part B — 重写 qa.explain 消费教案 + 整合写

- [ ] **Step 5: 重写 explain 测试（失败）**

把 `tests/test_qa_capability.py` 里 `test_explain_builds_sections_from_skeleton` 与 `test_explain_empty_skeleton_raises` 两个函数替换为下面四个（`_RecallNode` / `EmptySkeleton` import 已在文件内，保留）：

```python
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
```

- [ ] **Step 6: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py -k explain -v`
Expected: FAIL（旧 `explain()` 调 `outliner.run` 只传两参、不传 toc，且按旧的逐节拼接；新断言不匹配）

- [ ] **Step 7: 重写 `qa.explain()`**

把 `core/workflow/qa_capability.py` 里 `explain` 方法体（当前 line 216-263）整体替换为：

```python
    async def explain(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """讲清楚：宽覆盖召回 → 教学维度教案(吃 TOC) → 每维度检索 → 合并截断 → 一次整合教学写作。

        空教案 → raise EmptySkeleton（由 explain_branch 落 agent 兜底）。
        结构来自教学 schema + 书的 TOC（自上而下、不被召回碎片带偏）；事实只来自检索 pool。
        """
        # 1. 宽覆盖召回（内部，不发流事件——空教案时要静默落 agent，别先污染 UI）
        located = await self._explain_recall(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 2. 出教案：教学维度词表 + 书的 TOC 提示（单书才有，多书/未选 → []）
        toc_hint = self._book_chapters(book_titles)
        outline = await self.outliner.run(query, passages, toc_hint)
        if not outline:
            raise EmptySkeleton(query)

        # 3. 每维度检索扇出 → 去重合并 → 截断/重排到上下文预算（此时才发 RetrievalStart）
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        retrieved = await self._retrieve_all([d.query for d in outline], book_titles)
        pool = self._merge_pool(retrieved)
        if self.reranker:
            pool = await self.reranker.rerank(query, pool, self.rerank_candidate_k)
        else:
            pool = sorted(
                pool, key=lambda n: getattr(n, "score", 0) or 0, reverse=True
            )[: self.rerank_candidate_k]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
        if not pool:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []

        # 4. 一次整合教学写作（教案当脚手架、讲师 prompt 立 grounding + 做减法）
        answer = await self._teach_synthesize(ctx, query, outline, pool)
        return answer, pool
```

- [ ] **Step 8: 跑 explain 测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py -k explain -v`
Expected: 全绿

- [ ] **Step 9: 跑两个全文件确认无回归**

Run: `PYTHONUTF8=1 python -m pytest tests/test_qa_capability.py tests/test_answer_outliner.py tests/test_doc_workflow.py -q`
Expected: 全绿（`doc_workflow` 的 explain_branch 仍调 `qa.explain(ctx, rewritten, book_titles)`、签名未变；`EmptySkeleton` 仍抛出）

- [ ] **Step 10: 提交**

```bash
git add core/workflow/answer_outliner.py core/workflow/qa_capability.py tests/test_answer_outliner.py tests/test_qa_capability.py
git commit -m "feat(explain): teaching-schema outline + integrative teaching synthesis

AnswerOutliner now emits list[Dimension] from a fixed teaching vocabulary
(是什么/作用/组成/原理/适用·边界/关系) with TOC hint feeding 组成, instead of
recall-anchored sub_queries. qa.explain consumes the outline, fans out
per-dimension retrieval, merges+truncates a pool, and writes once via
_teach_synthesize. EmptySkeleton->agent->single-retrieve ladder preserved.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: eval —— explain golden 样例 + 锁定指标口径回归测试

eval harness 的 `score_row`/`aggregate` 已天然正确处理 explain 行（无 `category` → 不被 REFUSE 短路、不计入分类准确率；faithfulness/answer_relevancy 照算；需 reference 的指标在空 reference 上报错被吞成 None、不污染均值）。本任务**不改 harness 逻辑**，只：(1) 加 explain golden 样例文件；(2) 加一条回归测试把"explain 行得分不被短路、不计入分类"这条口径焊死，防日后误改 REFUSE 逻辑时把 explain 也误伤。

**Files:**
- Create: `eval/dataset/golden.explain.jsonl`
- Test: `tests/test_eval_run.py`（追加一个测试）

**Interfaces:**
- Consumes：`eval.harness.run_eval.score_row`、`eval.harness.metrics.MetricSpec`、`eval.harness.sut.RagOutput`（测试文件顶部已 import）。
- Produces：无新代码接口（数据 + 测试）。

- [ ] **Step 1: 写 explain golden 样例文件**

Create `eval/dataset/golden.explain.jsonl`（每行一条 JSON，**无 `category` 字段**=explain 行标记；`reference` 留空——faithfulness/answer_relevancy 不需参考答案）：

```jsonl
{"user_input": "讲讲MySQL的InnoDB存储引擎是什么、有哪些核心组成", "scope": null, "reference": ""}
{"user_input": "讲懂MySQL的Buffer Pool：它是什么、解决什么问题、由哪些部分构成", "scope": null, "reference": ""}
{"user_input": "请把MVCC的实现原理讲透", "scope": null, "reference": ""}
{"user_input": "什么是聚簇索引？请讲清楚它的定义、作用和与二级索引的关系", "scope": null, "reference": ""}
```

> 宽题（前两条，停在 是什么/作用/组成 高度）+ 具体深题（第三条，聚焦 原理）+ 中等题（第四条，是什么/作用/关系）。`scope: null` 跟现有 golden 一致（不限定单书 → explain 的 TOC 提示走"无干净目录"分支，正好覆盖该路径）。

- [ ] **Step 2: 写回归测试（失败）**

在 `tests/test_eval_run.py` 末尾追加（`score_row` / `MetricSpec` / `RagOutput` / `_FakeMetric` / `_FakeSUT` 已在文件内定义）：

```python
async def test_score_row_explain_row_scores_and_is_not_refuse_skipped():
    # explain 金标准行：无 category 字段、无 reference；answered。
    # 应照常算 faithfulness/answer_relevancy（不被 REFUSE 短路），且 category 留空
    # → aggregate 不把它计入分类准确率（已由 test_aggregate_skips_blank_category 覆盖）。
    out = RagOutput(response="教学体答案", retrieved_contexts=["c"], outcome="answered")
    specs = [
        MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {}),
        MetricSpec("answer_relevancy", _FakeMetric(0.8), lambda r, o: {}),
    ]
    res = await score_row(
        {"user_input": "讲讲MVCC是什么", "reference": ""}, _FakeSUT(out), specs
    )
    assert res["outcome"] == "answered"
    assert res["expected_category"] == ""        # 无金标准 category
    assert res["faithfulness"] == 0.9            # 未被 REFUSE 短路
    assert res["answer_relevancy"] == 0.8


def test_aggregate_explain_rows_excluded_from_classification():
    # explain 行（category 空）与难度分类行混合：分类准确率只算后者，explain 不被误判。
    rows = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable"},
        {"outcome": "answered", "category": "", "expected_category": ""},   # explain 行
        {"outcome": "answered", "category": "", "expected_category": ""},   # explain 行
    ]
    rep = aggregate(rows)
    assert rep["classification"]["total"] == 1      # 只数难度分类那 1 行
    assert rep["classification"]["accuracy"] == 1.0
```

- [ ] **Step 3: 跑测试确认通过（行为已存在，测试应直接绿）**

Run: `PYTHONUTF8=1 python -m pytest tests/test_eval_run.py -q`
Expected: 全绿。

> 这是"刻画现有正确行为"的回归测试（characterization test），不是先红后绿——`score_row`/`aggregate` 已实现该口径。若意外 FAIL，说明对 harness 行为的判断有误，**停下来**读 `eval/harness/run_eval.py` 的 `score_row`（REFUSE 短路条件 line 79）与 `aggregate`（`if exp and cat` line 103）确认，而非改测试将就。

- [ ] **Step 4: 校验 golden 文件可被 loader 解析**

Run:
```bash
PYTHONUTF8=1 python -c "from eval.harness.run_eval import load_testset; rows=load_testset('eval/dataset/golden.explain.jsonl'); print(len(rows), 'rows'); assert all('category' not in r for r in rows); print('ok: 全部无 category（explain 行）')"
```
Expected: `4 rows` 然后 `ok: 全部无 category（explain 行）`

- [ ] **Step 5: 提交**

```bash
git add eval/dataset/golden.explain.jsonl tests/test_eval_run.py
git commit -m "test(eval): explain golden samples + lock metric routing for explain rows

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 收尾（人工，不在自动执行步骤内）

合并前的真实冷烟（需 `.env` 的 `DEEPSEEK_API_KEY` + 已建索引）——这些靠人读，机器只跑不下结论：

1. **explain 质量冷烟**：用单书 explain 跑 `golden.explain.jsonl` 前两条宽题，肉眼确认产出是"是什么/作用/组成"高度的连贯讲解，**不再**是 `FIL_PAGE_UNDO_LOG` 之类碎片罗列；第三条具体题确认下沉到原理。
   命令参考：`PYTHONUTF8=1 python -m eval.harness.run_eval --testset eval/dataset/golden.explain.jsonl --detail eval/results/explain_smoke.csv`，逐条读 response 列。
2. **faithfulness 守线**：看上面 run 的 faithfulness 列——slice 2 丢了结构性隔离防幻觉，靠它盯整合写作有没有飘；若明显偏低，回 spec §8.4 调讲师 prompt 的 grounding 措辞。
3. **降级阶梯仍通**：构造一条召回必空的 explain 题，确认 `EmptySkeleton → agent → 单轮` 阶梯仍兜得住（不崩、有答案）。

冷烟通过后，用 `superpowers:finishing-a-development-branch` 收束分支（合并 `explain-slice2-lecturer` → master）。

---

## Self-Review

**1. Spec coverage**（对 `2026-06-21-explain-lecturer-synthesis-design.md` 逐条核）：
- AnswerOutliner 改造（召回锚定→教学维度化、出 `list[Dimension]`、"组成"吃 TOC、按 query 定深浅）→ Task 2 Part A ✓
- qa.explain 合成段改造（逐节孤立→一次整合教学写作 `_teach_synthesize`、教案脚手架、grounding 铁律、做减法、轻分节+开场/收束）→ Task 1（`_teach_synthesize`）+ Task 2 Part B（接线）✓
- eval（explain golden 样例 + faithfulness/answer_relevancy 评测侧、不计入分类准确率、不被 null 误伤）→ Task 3 ✓
- 保留 slice 1 覆盖半截（宽 hybrid 召回 / `_retrieve_all` / `_merge_pool`）与降级阶梯（`EmptySkeleton → agent → 单轮`）→ Task 2 explain 体保留 `_explain_recall`/`_retrieve_all`/`_merge_pool`，`EmptySkeleton` 仍抛、`explain_branch` 不动 ✓
- 固定维度词表 + 高度由 query 定 + 两处守高度（outliner 词表过滤 / teach prompt 做减法）→ `_ALLOWED_LABELS` 过滤（Task 2）+ `_TEACH_PROMPT` 做减法指令（Task 1）✓
- Non-goals（运行时不拦 faithfulness / 不做 TOC 结构性推组成 / 不做自定义教学指标 / explain 多跳仍走 agent / 不上 Graph-RAG）→ 计划均未触碰，符合 ✓
- 决策锁定 1-4（faithfulness 评测侧 / 固定词表 / TOC 强提示退 schema-only / 轻分节+开场收束一次写）→ 分别落在 Task 3、Task 2 outliner、Task 2 outliner `toc_text` 退化分支、Task 1 `_TEACH_PROMPT` ✓
- 命名（复用 `Dimension`、新方法 `_teach_synthesize`、组件仍叫 `AnswerOutliner`）→ Global Constraints + Task 2 ✓

**2. Placeholder scan**：无 TBD/TODO；每个改码步骤都给了完整代码（prompt 全文、方法全文、测试全文）；无"类似 Task N"占位。✓

**3. Type consistency**：
- `AnswerOutliner.run(query, passages, toc_hint=None, max_items=8) -> list[Dimension]`：Task 2 定义，explain 调用处传 `(query, passages, toc_hint)` 一致；测试 stub 签名 `(query, passages, toc_hint=None, max_items=8)` 一致 ✓
- `Dimension(label, query)`：Task 1/2/3 全程同一来源 `core.workflow.query_dimension` ✓
- `_teach_synthesize(ctx, query, outline, pool) -> str`：Task 1 定义，Task 2 explain 调用 `(ctx, query, outline, pool)` 一致，测试 stub 同参 ✓
- `_teach_tokens(prompt)` async gen / `astream_complete` → `chunk.delta`：Task 1 实现与 `FakeStreamLLM` 替身一致 ✓
- `DimensionSet`（含 `dimensions: list[Dimension]`）：Task 2 import 自 `query_dimension`，与该文件现有定义一致 ✓
- `rerank_candidate_k` / `reranker` / `_merge_pool` / `_explain_recall` / `_retrieve_all` / `_book_chapters`：均为 `QaCapability` 现有成员，explain 重写沿用同名同签 ✓
