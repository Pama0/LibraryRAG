# 可答性前置闸 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `out_of_scope` / `missing_info` 这条"能不能答"的轴从难度六分类器里抽出来，做成共享判定单元 `Admitter`，让 `explain` 和 `other` 两条路都先过它——修掉"库外/信息不足的 explain 问题溜进 agent → 脑补讲一顿"的根因。

**Architecture:** 四处改动。(1) 新增 `Admitter` 决策单元（`core/workflow/admitter.py`），吃 query + 召回片段，判 `ok / missing_info / out_of_scope`，判据从六分类器原样搬。(2) `qa.classify` 内部改成 `probe → admit → (ok 才跑瘦身分类器)`；非 ok 短路返回该类，对外契约仍可出 6 类。(3) `qa.explain` 宽召回后插 `admit`，非 ok 抛 `OutOfScope` / `MissingInfo`（镜像 `EmptySkeleton`，放 `qa_capability.py`），由 `explain_branch` 接住。(4) `QueryPreprocessor` 瘦身 6→4（删 `out_of_scope`/`missing_info` 两段判据 + 枚举）；拒答话术抽共享常量 `REFUSAL_TEXT` / `REFUSAL_FALLBACK`，库外分支与 explain 拒答共用。降级方向=放行（ok），靠后续 `QaAgent` 库外拒答补丁做防御纵深（另开一刀，不在本计划）。

**Tech Stack:** Python 3.12 / asyncio、LlamaIndex（`LLM.acomplete` + `response_format={"type":"json_object"}`、`Context.write_event_to_stream`）、Pydantic（`llama_index.core.bridge.pydantic`）、pytest（`pytest-asyncio`，`asyncio_mode=auto`）。

## Global Constraints

- **工作目录**：项目根 = `C:\Users\11394\PycharmProjects\llmaLearn`。所有命令从项目根运行；子模块内用相对导入、根脚本用绝对导入（见 `CLAUDE.md`）。
- **提交粒度**：用显式 `git add <文件路径>`，**绝不** `git add -A` / `git add .`（仓库里有大量本刀无关的未提交改动）。不跳 hooks、不跳签名。提交信息结尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **当前分支**：开工前先开 feature 分支（见 Task 1 Step 0）。
- **测试运行**：Windows 下中文 prompt 需 UTF-8，PowerShell 用 `$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest ...`——`$env:` 前缀单独设环境变量（bash 的 `VAR=val cmd` 前缀语法 PowerShell 5.1 不吃）；**必须用 `.venv\Scripts\python.exe`**，环境里裸 `python` 指向另一个项目（gpt-researcher）的 venv、缺 `llama_index`，会 ImportError 误导新 agent 瞎改。
- **所有 I/O 用 `async/await`**；函数签名加类型注解（中文注释可接受）。
- **core 不依赖 api**（守卫 `.venv\Scripts\python.exe scripts\check_layering.py`）。
- **决策单元约定（沿用，不是缺陷）**：注入 LLM、对外只暴露 `run`、`response_format={"type":"json_object"}`、Pydantic 校验、失败优雅降级、`_strip_fences` 每模块各自带一份副本——照抄，不要"消重"。
- **降级方向=放行（ok）**：`Admitter` 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）→ 默认 `ok`（放行去作答），与现有 `QueryPreprocessor` 失败降 `retrievable`、`gate` 失败降 `other` 同一哲学：判定器坏了不该误拒正常问题。残留风险由另一刀 `QaAgent` 库外拒答补丁接住。
- **不重排热路径**：`probe` / explain 宽召回的位置与 retriever 装配**一律不动**，只在证据产生处加一次 `admit` 调用。
- **`ambiguous` 不迁**：`ambiguous`（角度不定）是"按哪个角度答"、判错能优雅降级，属意图/答案形状轴，**留在难度分类器**，不进可答性闸。
- **命名（spec 评审已定）**：组件 `Admitter`；verdict 枚举 `ok / missing_info / out_of_scope`；异常 `OutOfScope` / `MissingInfo`（置于 `qa_capability.py`，镜像 `EmptySkeleton`）；共享拒答常量 `REFUSAL_TEXT` / `REFUSAL_FALLBACK`（也置 `qa_capability.py`，与异常同处）。
- **拒答话术单一来源**：`REFUSAL_TEXT` 取现写在 `out_of_scope_branch` 里的终结句原样抽出；库外分支与 explain `OutOfScope` catch 都引用，避免分叉。`REFUSAL_FALLBACK` 为 missing_info 缺 `clarify_question` 时的兜底反问。
- **probe 不统一/前移（明确不做）**：explain 宽召回（hybrid 大 top_k 求覆盖）与 classify probe（vector 不重排求章节 spread）取向相反，两路各带各的证据喂 `Admitter`，强行合并会压扁信号。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `core/workflow/admitter.py` | 新建 | `Admitter` 决策单元 + `AdmitVerdict` schema + `_ADMIT_PROMPT`（判据从六分类器原样搬）+ `_strip_fences` 副本 |
| `core/workflow/qa_capability.py` | 修改 | 加 `OutOfScope`/`MissingInfo` 异常 + `REFUSAL_TEXT`/`REFUSAL_FALLBACK` 常量（Task 2）；`__init__` 持 `admitter`（Task 3）；`classify` 内嵌 admit（Task 3）；`explain` 宽召回后插 admit + 抛异常（Task 5） |
| `core/workflow/query_preprocess.py` | 修改 | 瘦身 6→4：删 `out_of_scope`/`missing_info` 两段判据 + 枚举值 + 优先级段更新（Task 4） |
| `core/workflow/doc_workflow.py` | 修改 | import 加 `OutOfScope`/`MissingInfo`/`REFUSAL_TEXT`/`REFUSAL_FALLBACK`；`explain_branch` 接两异常 + 写 `category`（Task 6）；`out_of_scope_branch` 改用 `REFUSAL_TEXT` 常量（Task 6） |
| `tests/test_admitter.py` | 新建 | `Admitter` 单测：解析 ok/missing_info/out_of_scope、证据进 prompt、解析失败降级 ok |
| `tests/test_qa_capability.py` | 修改 | `classify` admit 短路测试（Task 3，含更新现有 classify 测试 stub admitter=ok）；`explain` admit 抛异常测试（Task 5） |
| `tests/test_query_preprocess.py` | 修改 | 删 out_of_scope/missing_info 旧用例（迁 test_admitter）；加"瘦身后这两类被枚举拒→降 retrievable"测试（Task 4） |
| `tests/test_doc_workflow.py` | 修改 | `explain_branch` catch OutOfScope→拒答+category / catch MissingInfo→反问+category / EmptySkeleton 仍落 agent（Task 6）；`out_of_scope_branch` 用 REFUSAL_TEXT 回归（Task 6） |

`core/workflow/doc_workflow.py` 的 `preprocess` step 图、`ExplainEvent`、`OutOfScopeEvent`、`ClarifyEvent`、finalize 的 metadata 结构**不动**——`qa.classify` 对外契约仍返回 6 类之一，workflow step 图与话术零改动。

---

## Task 1: 新建 `Admitter` 决策单元

纯加法：新建 `admitter.py` + 单测，不接进任何调用方。任务结束后测试全绿、运行时行为不变。

**Files:**
- Create: `core/workflow/admitter.py`
- Test: `tests/test_admitter.py`

**Interfaces:**
- Consumes：`llama_index.core.llms.LLM`（注入）、`llama_index.core.bridge.pydantic.BaseModel`/`Field`。
- Produces（Task 3/5 依赖）：
  - `AdmitVerdict(BaseModel)`：`verdict: Literal["ok","missing_info","out_of_scope"]`（默认 `"ok"`）、`reason: str = ""`、`clarify_question: str = ""`。
  - `Admitter(llm: LLM)`，`async def run(self, query: str, passages: list[str]) -> AdmitVerdict`。证据由调用方喂，不自检索。任何失败 → `AdmitVerdict(verdict="ok")`（放行）。

- [ ] **Step 0: 开 feature 分支**

Run:
```powershell
git checkout -b answerability-pregate
```
Expected: `Switched to a new branch 'answerability-pregate'`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_admitter.py`：

```python
"""Admitter（可答性判定单元）单测：mock LLM 控返回，验解析/降级/证据进 prompt。"""
from core.workflow.admitter import Admitter, AdmitVerdict


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


def _adm(llm):
    return Admitter(llm)


async def test_run_parses_ok():
    llm = FakeLLM(['{"verdict":"ok","reason":"主体在库且相关"}'])
    v = await _adm(llm).run("MySQL有哪些锁", ["片段A"])
    assert isinstance(v, AdmitVerdict)
    assert v.verdict == "ok"
    assert v.reason == "主体在库且相关"
    assert v.clarify_question == ""


async def test_run_parses_missing_info_with_clarify():
    llm = FakeLLM([
        '{"verdict":"missing_info","reason":"指代不明","clarify_question":"你说的「这个索引」指哪一个？B+树还是全文索引？"}'
    ])
    v = await _adm(llm).run("这个索引的应用场景", ["片段A"])
    assert v.verdict == "missing_info"
    assert v.clarify_question == "你说的「这个索引」指哪一个？B+树还是全文索引？"


async def test_run_parses_out_of_scope():
    llm = FakeLLM([
        '{"verdict":"out_of_scope","reason":"PostgreSQL 不在库，召回全是 MySQL"}'
    ])
    v = await _adm(llm).run("PostgreSQL的MVCC", ["MySQL 片段"])
    assert v.verdict == "out_of_scope"
    assert v.reason == "PostgreSQL 不在库，召回全是 MySQL"


async def test_run_injects_passages_into_prompt():
    llm = FakeLLM(['{"verdict":"ok"}'])
    await _adm(llm).run("openclaw 是什么", ["片段甲", "片段乙"])
    assert "片段甲" in llm.prompts[0]
    assert "片段乙" in llm.prompts[0]
    assert "openclaw 是什么" in llm.prompts[0]
    assert "json_object" not in llm.prompts[0]   # 不进 prompt 正文


async def test_run_empty_passages_still_works():
    llm = FakeLLM(['{"verdict":"out_of_scope","reason":"召回空，主体缺席"}'])
    v = await _adm(llm).run("Cassandra分片", [])
    assert v.verdict == "out_of_scope"


async def test_run_parse_failure_degrades_to_ok():
    llm = FakeLLM(["这不是JSON"])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"            # 失败 → 放行，不误拒


async def test_run_empty_content_degrades_to_ok():
    llm = FakeLLM([""])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"


async def test_run_invalid_verdict_rejected_to_ok():
    # 枚举外的 verdict 应被 Pydantic 拒 → 降级 ok
    llm = FakeLLM(['{"verdict":"maybe"}'])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"


async def test_run_strips_fenced_json():
    llm = FakeLLM(['```json\n{"verdict":"ok"}\n```'])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_admitter.py -v
```
Expected: 全部 FAIL / ERROR —— `ModuleNotFoundError: No module named 'core.workflow.admitter'`。

- [ ] **Step 3: 实现 `Admitter`**

新建 `core/workflow/admitter.py`：

```python
"""Admitter：可答性判定单元（正交于难度分类器）。

把"能不能答"这条轴从难度六分类器里抽出来：吃 query + 召回片段，只判
ok / missing_info / out_of_scope。判据原样搬自 QueryPreprocessor 的那两段
（含"只看主体实体在不在库""深度/角度不匹配≠库外"等已调细的铁律）。

- 证据由调用方喂，不自检索（explain 喂宽召回片段、classify 喂 probe 格式化证据）。
- 沿用决策单元约定：注入 LLM、只暴露 run、json_object + Pydantic 校验、失败降级 ok、
  自带 _strip_fences 副本。
- 降级方向=放行（ok）：判定器坏了不该误拒正常问题。残留风险由 QaAgent 库外拒答补丁接住。

设计见 docs/superpowers/specs/2026-06-21-answerability-pregate-design.md。
"""
import logging
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
# 判据原样搬自 QueryPreprocessor._JUDGE_PROMPT 的 out_of_scope / missing_info 两段
# + 铁律 preamble + 优先级段（只保留可答性轴，删掉难度分类器的 retrievable/ambiguous/
# pending_split/other 判据——那些不归本单元）。
_ADMIT_PROMPT = """你是知识库问答的【可答性判定器】。下面给你一个问题和一批从知识库召回到的片段。你只判一件事：这个问题能不能基于这些片段答——能答(ok)、信息不足要反问(missing_info)、还是库外(out_of_scope)。

【铁律·必读】判定必须以末尾的【召回片段】为准，【绝不以你是否认识问题中的词为准】。知识库里全是你训练时没见过的专有名词（书名/工具名/项目名），其含义由检索决定、不由你的世界知识判断。绝不要因为"我不认识这个词"就判 missing_info 或 out_of_scope——只要召回到与问题相关且集中的内容，就是 ok。

判据（三选一）：

- ok：召回片段与问题相关（主体实体在库），且信息足够作答；或片段虽不完全覆盖但与问题主题一致、不算库外也不缺关键限定。其余皆 ok。
  返回 {"verdict":"ok","reason":"判定理由"}

- missing_info（信息不足）：**末尾【召回片段】到了与问题相关的主题**，但缺了检索必需的关键限定/指代不明，补充后才能精确命中。多为指代不明。
  如「这个索引的应用场景是什么」——库里有索引内容，但"这个索引"指代不明（全文索引？B+树索引？其他？），补充后即可检索。
  注意：若召回到了相关内容，即便问题里有你不认识的专名，也不是 missing_info。
  返回 {"verdict":"missing_info","reason":"需澄清的原因，如'这个索引'指代不明","clarify_question":"一句自然、面向用户的反问，点明不明之处并引导补充，能列候选就列，如'你说的「这个索引」具体指哪一个？是 B+树索引、全文索引，还是其他？'"}

- out_of_scope（库外）：**问题的主体技术实体根本不在知识库里**——末尾【召回片段】里找不到该实体的任何内容（召回片段讲的全是另一套系统）。判据【只看主体实体在不在库】，与问题是否完整、是否缺限定无关；因为库里没有的内容，反问也补不出来。
  判断以问题的【主体技术实体】为准（如 PostgreSQL、MongoDB、Oracle、Cassandra 这类系统/产品名）：若召回片段讲的是另一套系统，即便与问题里的通用术语（如"一致性""分片""架构""集群""事务"）字面重合，也属主体实体缺席 → out_of_scope。
  【铁律·别把"不匹配"误当库外】只要召回里出现了问题的主体实体，就【绝不是】库外——哪怕召回的【深度/角度/粒度/广度】跟用户想要的不一致（用户要"入门概念"、库里是"高阶内核细节"；用户问得很宽、库里是细节散在多章；用户要某个角度、库里是另一角度）。这类深度/角度/广度/完整性不匹配【一律不判库外】，判 ok。
  特征：召回片段讲的全是另一个系统、主体实体缺席。如「PostgreSQL的MVCC怎么实现」「MongoDB分片」「Oracle RAC」「Cassandra的一致性级别」——本库召回到的都是别的系统（如 MySQL）。
  反例（不是库外）：「给我讲懂MySQL的核心概念」——MySQL 在库，只是问得宽、要得浅，召回是高阶内核细节也无妨，判 ok（绝不判库外）。
  返回 {"verdict":"out_of_scope","reason":"库外原因，如'Cassandra 不在本库主题范围，召回片段是 MySQL 内容、主体实体缺席'"}

【判据优先级】**最先看末尾【召回片段】里问题的主体技术实体在不在库——只有主体实体根本缺席（召回全是另一套系统）才判 out_of_scope（最优先，无论问题是否完整、是否缺限定）；若主体实体在库、只是召回的深度/角度/广度与用户所求不符，不算库外，判 ok；在召回相关的前提下，再判断信息是否不足(missing_info)；其余皆 ok**。

只返回 JSON，不要其它任何内容：
{"verdict":"ok / missing_info / out_of_scope","reason":"判定理由","clarify_question":"missing_info 专用：面向用户的反问句；其余为空字符串"}

【召回片段】
{passages}

问题：{query}"""


class AdmitVerdict(BaseModel):
    """LLM 判定目标 schema（代码侧 Pydantic 校验）。

    verdict 用 Literal 锁枚举，非法值会在 model_validate 阶段被拒 → 降级 ok。
    默认 ok：构造失败兜底时直接用 AdmitVerdict() 即放行。
    """

    verdict: Literal["ok", "missing_info", "out_of_scope"] = "ok"
    reason: str = Field(default="", description="判定理由（日志/调试）")
    clarify_question: str = Field(
        default="", description="missing_info 专用：面向用户的自然反问句"
    )


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class Admitter:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, query: str, passages: list[str]) -> AdmitVerdict:
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
            logger.info("admit: verdict=%s reason=%s", verdict.verdict, verdict.reason)
            return verdict
        except Exception as exc:
            # 任何失败 → 放行（ok），绝不阻塞；判定器坏了不该误拒正常问题
            logger.warning("admit 解析失败，降级 ok（放行）：%s", exc)
            return AdmitVerdict(verdict="ok")
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_admitter.py -v
```
Expected: 全部 PASS。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿（纯加法，不应触碰任何现有测试）。

- [ ] **Step 6: 提交**

```powershell
git add core/workflow/admitter.py tests/test_admitter.py
git commit -m "feat: add Admitter decision unit for answerability pre-gate

把 out_of_scope / missing_info 这条可答性轴从难度六分类器里抽出来做成共享
判定单元，吃 query + 召回片段判 ok/missing_info/out_of_scope，失败降级 ok。
本刀纯加法，尚未接进 classify/explain。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 加 `OutOfScope`/`MissingInfo` 异常 + `REFUSAL_TEXT`/`REFUSAL_FALLBACK` 常量

纯加法：在 `qa_capability.py` 顶部加两个异常类 + 两个常量，暂不使用。为 Task 5/6 的接线备好符号。

**Files:**
- Modify: `core/workflow/qa_capability.py`（在 `EmptySkeleton` 定义之后追加异常 + 常量）
- Test: `tests/test_qa_capability.py`（末尾加最小可导入/常量值测试）

**Interfaces:**
- Consumes：无。
- Produces（Task 5/6 依赖）：
  - `OutOfScope(Exception)`：构造 `OutOfScope(query)`，携带 query 供日志。
  - `MissingInfo(Exception)`：构造 `MissingInfo(clarify_question)`，`.clarify_question` 属性供 explain_branch 取用。
  - `REFUSAL_TEXT: str`：库外拒答终结句（原样取自现 `out_of_scope_branch`）。
  - `REFUSAL_FALLBACK: str`：missing_info 缺 clarify_question 时的兜底反问。

- [ ] **Step 1: 写失败测试**

在 `tests/test_qa_capability.py` **末尾**追加（文件顶部已 `from core.workflow.qa_capability import QaCapability, AnswerDeltaEvent`、已 `import pytest`、已 `from core.workflow.qa_capability import EmptySkeleton`；此处补 import 新符号）：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "refusal_text or refusal_fallback or out_of_scope_exception or missing_info_exception" -v
```
Expected: FAIL —— `ImportError: cannot import name 'OutOfScope'`。

- [ ] **Step 3: 加异常 + 常量**

`core/workflow/qa_capability.py` 在 `class EmptySkeleton(Exception):` 那个类定义之后（约第 64 行之后、`# ── 流式专用事件` 注释之前）插入：

```python
class OutOfScope(Exception):
    """explain 路 admit 判库外 → 由 explain_branch 接住拒答。镜像 EmptySkeleton 的异常驱动控制流。"""


class MissingInfo(Exception):
    """explain 路 admit 判信息不足 → 由 explain_branch 接住反问。

    clarify_question 由 Admitter 产；缺时 explain_branch 用 REFUSAL_FALLBACK 兜底。
    """

    def __init__(self, clarify_question: str = ""):
        super().__init__(clarify_question or "")
        self.clarify_question = clarify_question or ""


# ── 拒答话术共享常量（库外分支与 explain OutOfScope catch 共用，避免分叉）──
REFUSAL_TEXT = (
    "这个问题知识库里暂未收录相关内容，我没法基于现有资料回答。"
    "你可以换个已入库主题问我，或把问题换个角度再试试～"
)
# missing_info 缺 clarify_question 时的兜底反问
REFUSAL_FALLBACK = "为了更准确地回答，能不能把问题再说具体一点？"
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "refusal_text or refusal_fallback or out_of_scope_exception or missing_info_exception" -v
```
Expected: 全部 PASS。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```powershell
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: add OutOfScope/MissingInfo exceptions + REFUSAL constants

为可答性前置闸接线备好符号：异常镜像 EmptySkeleton 放 qa_capability.py；
REFUSAL_TEXT 原样抽自 out_of_scope_branch 终结句，供库外分支与 explain
OutOfScope catch 共用；REFUSAL_FALLBACK 为 missing_info 缺反问句时的兜底。
纯加法，尚未使用。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 把 `Admitter` 接进 `qa.classify`

`qa.classify` 内部改成 `probe → admit → (ok 才跑 preprocessor)`。非 ok 短路返回该类（带 `clarify_question`）；ok → 走原 preprocessor。**对外契约不变**（仍返回 `PreprocessResult`，仍可出 6 类）。preprocessor prompt 此时**不动**（Task 4 才瘦），故 admit 与 preprocessor 暂时对 out_of_scope/missing_info 有冗余覆盖——这是过渡态，不破坏正确性。

**Files:**
- Modify: `core/workflow/qa_capability.py`（`__init__` 加 `self.admitter`；`classify` 方法体改）
- Test: `tests/test_qa_capability.py`（加 admit 短路测试；更新现有 classify 测试 stub `admitter.run=ok`）

**Interfaces:**
- Consumes：Task 1 的 `Admitter` + `AdmitVerdict`；现有 `PreprocessResult`。
- Produces：`qa.classify` 行为扩展——admit 非 ok 时短路返回 `PreprocessResult("out_of_scope"|"missing_info", reason, clarify_question)`，不调 preprocessor；admit ok 时调 preprocessor（行为同前）。`QaCapability.__init__` 新增 `self.admitter = Admitter(llm)`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_qa_capability.py` **末尾**追加（顶部需补 import：在已有的 `from core.workflow.qa_capability import ...` 行里加 `OutOfScope`、`MissingInfo`——但本任务只测 classify，不涉及异常，故仅补 `AdmitVerdict` 用不到；直接用 `PreprocessResult` 即可。文件已 `from core.workflow.query_preprocess import PreprocessResult` 出现在多处测试函数内部 local import，沿用该模式）：

```python
# ── classify：admit 短路 / ok 落 preprocessor（Task 3）──────────────────
from core.workflow.query_preprocess import PreprocessResult
from core.workflow.admitter import Admitter, AdmitVerdict


def _ok_admitter(qa):
    """给 qa 装一个恒 ok 的 admitter，复用现有 classify 测试。"""
    async def _ok(query, passages):
        return AdmitVerdict(verdict="ok")
    qa.admitter.run = _ok


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
```

同时**更新现有 classify 测试**让它们 stub `admitter.run=ok`（否则 admit 会真调 `FakeLLM.acomplete`，而本文件 `FakeLLM.acomplete` raise `AssertionError`）。在 `_run_classify` helper 里加一行；在 `test_classify_probes_then_passes_context_to_preprocessor`、`test_classify_degrades_when_probe_fails` 里各自加一行。

改 `_run_classify`（约第 468 行）：

```python
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
```

`test_classify_probes_then_passes_context_to_preprocessor`（约第 320 行）在 `qa.preprocessor.run = fake_run` 之后加：

```python
    async def fake_admit(query, passages):
        from core.workflow.admitter import AdmitVerdict
        return AdmitVerdict(verdict="ok")
    qa.admitter.run = fake_admit
```

`test_classify_degrades_when_probe_fails`（约第 335 行）同样在 `qa.preprocessor.run = fake_run` 之后加同上 `fake_admit` 三行。

- [ ] **Step 1b: 更新 `test_doc_workflow.py` 现有靠 classify 真跑的测试（改 stub classify 整体）**

Task 3 给 classify 内嵌 admit 后，调用顺序变成 `front_door(LLM) → admit(LLM) → preprocessor(LLM)`，但现有这些测试的 FakeLLM 只准备了两轮（front_door + 原 preprocessor）。admit 吃掉第二轮 → 默认 ok 放行 → preprocessor 吃第三轮 → IndexError 降 retrievable → 走 `RetrieveAgentEvent`，导致 missing_info/out_of_scope 测试 retrieve 被调、other 测试走错分支、`test_router_parse_failure` 的 `llm.calls` 断言失败。

这些测试本就测 workflow 分支接线（不是 classify 内部），改成 stub `wf.qa.classify` 整体返回 `PreprocessResult` 一次到位，Task 4 瘦身也不影响它们。

**统一改法**：每个测试的 `FakeLLM([...])` 只留 front_door 那一轮（删掉原 `{"category":...}` 那轮）；在 `wf.qa.retrieve`/`wf.qa_agent.run` 等 stub 之后加：

```python
    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("<CATEGORY>", reason="<REASON>", clarify_question="<CLARIFY>")
    wf.qa.classify = fake_classify
```

**各测试的 CATEGORY / REASON / CLARIFY 与额外断言调整**：

| 测试函数（行号） | CATEGORY | REASON | CLARIFY | 额外调整 |
|---|---|---|---|---|
| `test_qa_intent_feeds_clean_query_and_scope_to_answer` (71) | `retrievable` | | | 保留 `fake_retrieve` + `captured` 断言不变 |
| `test_route_passes_selected_books_to_router` (96) | `retrievable` | | | 保留 `"openclaw" in llm.prompts[0]` 断言（front_door prompt） |
| `test_qa_preprocess_consumes_clean_query_not_original` (113) | `retrievable` | | | **改断言**：原 `assert "MySQL索引有哪些" in llm.prompts[1]` 改为 `fake_classify` 捕获入参——在 `fake_classify` 里 `captured["clean"] = clean_query`，断言 `captured["clean"] == "MySQL索引有哪些"` 且 `"它有哪些" not in captured["clean"]`；删掉 `llm.prompts[1]` 相关断言（preprocessor 不再被调） |
| `test_router_parse_failure_defaults_to_qa_path` (130) | `retrievable` | | | **改断言**：`assert llm.calls == 2` 改为 `assert llm.calls == 1`（只 front_door 一轮；classify 被 stub 不调 LLM）；保留 `captured["query"] == "B+树索引"` |
| `test_missing_info_clarifies_without_retrieval` (150) | `missing_info` | `指代不明` | | 保留 `called["retrieve"] is False` + `"指代不明" in str(result.response)` |
| `test_missing_info_uses_natural_clarify_question` (171) | `missing_info` | | `你说的「这个索引」指哪一个？B+树还是全文索引？` | 保留 `"你说的「这个索引」指哪一个" in str(result.response)` |
| `test_other_category_answers_via_dedicated_branch` (181) | `other` | `开放设计题` | | 保留 `fake_retrieve` + `captured["query"]` + `str(result.response) == "复杂问题答案"`（走 OtherEvent → other_branch → agent 未 stub 抛错 → except → fake_retrieve，与原行为一致） |
| `test_missing_info_budget_exhausted_assumes_and_answers` (203) | `missing_info` | `指代不明` | | 保留 `allow_clarify=False` + `fake_retrieve` 捕获 `preamble` + `"按最可能的解读作答" in captured["preamble"]` |
| `test_other_dispatches_to_bounded_agent` (227) | `other` | `开放权衡` | | 保留 `fake_agent_run` stub + `captured` 断言（走 OtherEvent → other_branch → `qa_agent.run=fake_agent_run`） |
| `test_other_falls_back_to_single_retrieve_when_agent_raises` (250) | `other` | `开放设计` | | 保留 `boom` agent + `fake_retrieve` + caplog `"other agent 失败"` 断言 |
| `test_flags_off_degrade_branches_to_single_retrieve` (299) | `pending_split` | `需罗列` | | 保留 `split_enabled=False` + `boom_split` + `used["retrieve"]` + `str(result.response) == "单轮答案"`（走 SplitEvent → split_branch flag off → fake_retrieve） |
| `test_finalize_exposes_category_in_metadata` (326) | `retrievable` | | | 保留 `result.metadata.get("category") == "retrievable"` |
| `test_out_of_scope_responds_without_retrieval_or_clarify` (367) | `out_of_scope` | `库外，召回片段均不相关` | | 保留 `called["retrieve"] is False` + response 话术断言 + `metadata.category == "out_of_scope"` |

**完整示例**（`test_missing_info_clarifies_without_retrieval`，其余照此模式）：

```python
async def test_missing_info_clarifies_without_retrieval():
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}'])
    wf = _wf(llm)

    called = {"retrieve": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("missing_info", reason="指代不明")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert called["retrieve"] is False            # 反问，不检索
    assert "指代不明" in str(result.response)
```

注意：`_wf(llm)` 里 `wf.qa.gate = _echo_other_gate`（stub gate 为 other）保持不变——这些测试不测 gate，gate stub 避开 explain 路径。`test_preprocess_passes_book_titles_to_classify` (275) 和 `test_explain_intent_routes_to_explain_branch` (392) **不改**——前者已 stub `wf.qa.classify`，后者 stub 了 gate+explain+classify。

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "classify_admit or classify_probes or classify_degrades or classify_probe_uses" -v
```
Expected: 新 `classify_admit_*` FAIL（`AttributeError: 'QaCapability' object has no attribute 'admitter'`）；现有 classify 测试也 FAIL（同因）。`test_doc_workflow.py` 的改动在 Step 1b 已就位，本步不跑它（Step 6 一起跑）。

- [ ] **Step 3: 给 `QaCapability` 加 `admitter` 字段**

`core/workflow/qa_capability.py` 顶部 import 段加：

```python
from core.workflow.admitter import Admitter
```

`QaCapability.__init__` 里在 `self._gate = QueryGate(llm)` 那行之后加：

```python
        self.admitter = Admitter(llm)
```

- [ ] **Step 4: 改 `classify` 方法体**

`core/workflow/qa_capability.py` 的 `classify` 方法（约第 130–148 行）整体替换为：

```python
    async def classify(
        self,
        clean_query: str,
        book_titles: Optional[list[str]] = None,
        probe: bool = True,
    ):
        """先用 clean_query 探测召回，把召回信号喂给 admit + judge，堵住「盲判」。

        probe=False（ablation baseline）→ 不探测、纯文本判定；probe 失败亦容错为空。
        可答性闸：admit 吃 probe 证据判 ok/missing_info/out_of_scope；非 ok 短路返回该类，
        ok 才跑瘦身分类器（4 类）。对外契约仍可出 6 类，workflow step 图不变。
        """
        retrieval_context = ""
        if probe:
            try:
                located = await self._probe_retrieve(clean_query, book_titles)
                retrieval_context = self._format_probe(located, book_titles)
            except Exception as exc:
                logger.warning("classify probe 探测失败，退回纯文本判定：%s", exc)
        verdict = await self.admitter.run(clean_query, [retrieval_context])
        if verdict.verdict == "out_of_scope":
            return PreprocessResult("out_of_scope", verdict.reason)
        if verdict.verdict == "missing_info":
            return PreprocessResult(
                "missing_info", verdict.reason, verdict.clarify_question
            )
        return await self.preprocessor.run(clean_query, retrieval_context)
```

需在文件顶部 import 段加 `PreprocessResult`（若尚未引入）：

```python
from core.workflow.query_preprocess import QueryPreprocessor, PreprocessResult
```

（现有 import 是 `from core.workflow.query_preprocess import QueryPreprocessor`，补上 `, PreprocessResult`。）

- [ ] **Step 5: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "classify" -v
```
Expected: 全部 PASS（新 admit 短路 + 现有 probe 测试均绿）。

- [ ] **Step 6: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。`test_doc_workflow.py` 的 13 个测试已由 Step 1b 改为 stub `wf.qa.classify` 整体返回 `PreprocessResult`，不受 classify 内嵌 admit 影响；`test_query_preprocess.py` 测的是 `QueryPreprocessor` 本身，也不受影响。

- [ ] **Step 7: 提交**

```powershell
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: wire Admitter into qa.classify (probe → admit → preprocessor)

admit 吃 probe 证据判可答性；非 ok 短路返回 out_of_scope/missing_info，
ok 才跑瘦身分类器。对外契约仍可出 6 类，workflow step 图不变。
preprocessor prompt 本刀不动（Task 4 才瘦），过渡态有冗余覆盖但不破坏正确性。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 瘦身 `QueryPreprocessor` 6→4

删 `out_of_scope` / `missing_info` 两段判据 + 枚举值，preprocessor 只判检索结构/难度 4 类。admit 已在 Task 3 接管可答性轴，故瘦身不丢能力。失败仍降 `retrievable`（不变）。

**Files:**
- Modify: `core/workflow/query_preprocess.py`（prompt `_JUDGE_PROMPT`、`QueryJudgment.category` 枚举、模块 docstring）
- Test: `tests/test_query_preprocess.py`（删迁移到 test_admitter 的旧用例；加"瘦身后两类被枚举拒→降 retrievable"测试）

**Interfaces:**
- Consumes：Task 3 的 admit 短路（保证瘦身不丢 out_of_scope/missing_info 能力）。
- Produces：`QueryJudgment.category` 的 `Literal` 从 6 值缩到 4 值（`retrievable / pending_split / ambiguous / other`）；`PreprocessResult` 签名不变（仍可被 admit 短路填 out_of_scope/missing_info）。

- [ ] **Step 1: 写失败测试（先加瘦身后新约束）**

在 `tests/test_query_preprocess.py` 末尾追加：

```python
async def test_run_slim_rejects_out_of_scope_after_extract():
    # 瘦身后 out_of_scope 不在枚举内 → Pydantic 拒 → 降级 retrievable
    llm = FakeLLM(['{"category": "out_of_scope", "rewritten_query": "PostgreSQL的MVCC"}'])
    result = await _pp(llm).run("PostgreSQL的MVCC")
    assert result.category == "retrievable"


async def test_run_slim_rejects_missing_info_after_extract():
    # 瘦身后 missing_info 不在枚举内 → Pydantic 拒 → 降级 retrievable
    llm = FakeLLM(['{"category": "missing_info", "rewritten_query": "这个索引"}'])
    result = await _pp(llm).run("这个索引")
    assert result.category == "retrievable"


async def test_run_prompt_slim_has_no_out_of_scope_section():
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "MySQL锁"}'])
    await _pp(llm).run("MySQL锁")
    p = llm.prompts[0]
    assert "out_of_scope（库外）" not in p          # 类定义段被删
    assert "missing_info（信息不足）" not in p      # 类定义段被删
    # 4 类仍在
    assert "retrievable" in p and "pending_split" in p
    assert "ambiguous" in p and "other" in p


async def test_run_prompt_slim_enum_line_lists_four_classes():
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "MySQL锁"}'])
    await _pp(llm).run("MySQL锁")
    p = llm.prompts[0]
    # 末尾枚举约束行应只列 4 类
    assert "retrievable|pending_split|ambiguous|other" in p
    assert "out_of_scope" not in p and "missing_info" not in p
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_query_preprocess.py -k "slim" -v
```
Expected: 全部 FAIL——`out_of_scope`/`missing_info` 仍在枚举内，不被拒；prompt 仍含那两段。

- [ ] **Step 3: 瘦身 `_JUDGE_PROMPT`**

`core/workflow/query_preprocess.py` 的 `_JUDGE_PROMPT` 整体替换为下面的瘦身后版本（删 missing_info 段、删 out_of_scope 段、更新优先级段、更新对照例、更新末尾枚举约束为 4 类；保留 retrievable / ambiguous / pending_split / other 四段 + 铁律 preamble + 降噪步）：

```python
_JUDGE_PROMPT = """你是检索 query 处理器。下面的 query 已经过净化（指代已消解、错别字已纠正、形式已规范），你只需做两步：先降噪，再判定该 query 的检索结构难度。
要求：如果问题已经足够清晰适合检索，降噪可不改动，不要强行改写。

第一步 降噪（去除口语化、礼貌性、请求词、无信息量的词，保留关键词语、实体、技术名词）
如（原始问题：小E,我想请问一下,MySQL有哪些锁啊?
改写后的查询：MySQL有哪些锁）
降噪只删与检索无关的冗余措辞，严禁删除任何技术限定词、修饰语、实体、版本号或承载意图的词（如"聚簇""行级""全文""有哪些""区别""第3章"）。判据：一个词删掉后若会改变检索命中，就必须保留。
反例：「MySQL聚簇索引和二级索引的区别」不可降成「MySQL 索引」——删掉"聚簇""二级""区别"会毁掉意图。

第二步 判定该 query 的检索结构难度（基于降噪后的 query）。

【铁律·必读】判定必须以末尾的【知识库探测召回】为准，【绝不以你是否认识问题中的词为准】。知识库里全是你训练时没见过的专有名词（书名/工具名/项目名），其含义由检索决定、不由你的世界知识判断。绝不要因为"我不认识这个词"或"这个词可能多义/很难"就判 other——只要召回到与问题相关且集中的内容，就是 retrievable。

【可答性由前置闸判定，本步不再判】本步只判检索结构难度。若召回片段与问题主体实体明显不相关（库外）或缺关键限定（信息不足），那已由前置可答性闸处理，本步不重复判，只关注召回相关的前提下"怎么查"。

【可以】retrievable：召回片段与问题相关，且**一条检索 query 就能集中命中**——单一信息需求，哪怕答案正文要拼几段、要枚举若干项，只要这些内容集中在同一片区域即可。
特征：仅检索问题即可。**枚举集中也算这类**——如「MySQL 有哪些锁」，锁都列在同一节，一次命中即可；枚举本身不等于要拆分。
旁证：参考末尾【知识库探测召回】的「跨 N 个章节」——命中集中在 1 个章节、有明显主导章 → 倾向 retrievable。
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}

【不可以】归入以下三类之一：

- ambiguous（角度不定）：话题已具体、能集中命中，但用户想要的维度/立场未给，有多个合理答法不知道选哪个。
  特征：答案就一个主题，但有几种角度/立场可选。
  如「Vue和React哪个好」(缺选型维度)「Redis做缓存好吗」(缺评判角度)
  返回 {"category":"ambiguous","rewritten_query": "处理后的 query","reason": "角度不定的原因，比如vue和React哪个好缺少评价维度"}

- pending_split（需要拆分）：判据=**一条检索 query 覆盖不全，必须扇出多个彼此独立的子查询**（这些子查询在**一轮粗召回定位后**就能一次性规划全、并行检索，彼此不依赖对方的检索结果）。触发于以下二者之一：
  · 多主体（**只看问题文本，与 probe 形状无关**）：问题显式含 ≥2 个并列主体、且带比较/对比/区别/异同意图（「A和B的区别」「A和B有什么不同」「A、B、C分别…」），需各自检索再综合。**即便 probe 召回看着集中在一处，这类结构也判 pending_split**——扇出各检索一侧再综合，覆盖比单轮 top-k 全，单轮易只命中泛化的上位概念而丢掉某一侧。如「Vue和React的区别」「聚簇索引和二级索引的区别」。
  · 广度分散（看 probe 形状）：单一大主题铺成多个互不重叠的子领域，单轮 top-k 覆盖不全，且末尾【知识库探测召回】显示命中**跨多个章节、无明显主导章**佐证。如「怎么优化MySQL」（索引/查询/配置/架构散在多章）。
  特征：答案需罗列/综合多个并列子项才完整，且子查询互相独立、不依赖彼此的检索结果。
  与 other 的边界：若子查询之间有**依赖**（后一个要等前一个**检索回的答案**才写得出来，即多跳），不归这里，归 other。
  返回 {"category":"pending_split","rewritten_query": "处理后的 query","reason": "需要拆分的原因，如'MySQL优化跨多章需扇出'、'Vue和React两主体需分别检索'"}

- other（高难度/开放复杂问题）：**召回到了相关内容，但**需要【多跳依赖检索、跨主题综合多步推理，或开放设计/权衡比较】，单轮、甚至一次性并行扇出都答不全，必须多轮检索+推理逐步求解。
  特征二者之一：
  · 多跳依赖：子查询之间有依赖，后一跳的 query 要等前一跳**检索回的答案**才写得出来（一轮定位后规划不完，得边检索边定下一步）。如「MySQL 默认隔离级别会有哪些并发问题」——先查出默认级别是 RR，才能去查 RR 的并发问题。
  · 开放综合/权衡：要综合多处证据分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  与 pending_split 的边界：一轮定位后就能一次产出全部子查询、彼此独立可并行（不依赖中间检索结果）→ pending_split；做不到（下一步查什么要看上一步检索的答案，或步骤集合都预定不了）→ other。
  铁律：other 看的是【问题结构是否需多跳/多步综合】，不是【你认不认识其中的词】。「X是什么 / 讲讲X / 讲明白X」这类即便 X 是你不认识的专名，只要召回到相关内容，就归 retrievable（单一概念）或 pending_split（X 是大主题需罗列），**绝不因不认识 X 而判 other**。
  返回 {"category":"other","rewritten_query": "处理后的 query", "reason":"判为高难度的原因，如'多跳依赖：需先定位默认级别再查其并发问题'、'需跨主题综合+权衡比较'"}

【不可以】归类的优先级：在召回相关的前提下，先判断是否角度不定(ambiguous)，再判断是否需扇出独立子查询(pending_split)；若以上都不是、但问题需要多跳依赖/跨主题综合/开放权衡，则判 other（积极）；其余一条 query 能集中命中的归 retrievable。

对照：
  「怎么优化MySQL」→ pending_split（优化是一整片：索引/查询/配置/架构）
  「给我讲懂MySQL的核心概念」→ pending_split（主体在库、问得宽，概念散在多章需扇出）
  「MySQL大表查询慢怎么优化」→ ambiguous（场景已具体，仍有索引/分区/改SQL几个角度）
  「Vue和React哪个好」→ ambiguous（缺"好"的维度，虽然两个实体，但仍为ambiguous）
  「Vue和React的区别」→ pending_split（不缺维度，两主体需分别检索再比，子查询独立可并行）
  「MySQL默认隔离级别会有哪些并发问题」→ other（多跳依赖：先查出默认级别，才能查该级别的并发问题，子查询有先后依赖）

category 仅为[retrievable|pending_split|ambiguous|other]不允许有其他词，rewritten_query 始终返回处理后的 query，reason返回对应的原因，结果只返回 JSON，不要其他任何内容。

系统已用该 query 在知识库做了一次探测检索：
【知识库探测召回】
{retrieval}

query：{query}"""
```

- [ ] **Step 4: 瘦身 `QueryJudgment.category` 枚举**

`core/workflow/query_preprocess.py` 的 `QueryJudgment` 类（约第 106 行）：

```python
    category: Literal[
        "retrievable", "pending_split", "missing_info", "ambiguous", "other", "out_of_scope"
    ]
```

改为：

```python
    category: Literal[
        "retrievable", "pending_split", "ambiguous", "other"
    ]
```

- [ ] **Step 5: 瘦身模块 docstring**

`core/workflow/query_preprocess.py` 顶部模块 docstring 第 5 行：

```
- 难度分类 → retrievable / pending_split / missing_info / ambiguous / other / out_of_scope
```

改为：

```
- 难度分类 → retrievable / pending_split / ambiguous / other（可答性轴 out_of_scope/missing_info 已上移到 Admitter 前置闸）
```

- [ ] **Step 6: 删迁移到 test_admitter 的旧用例**

`tests/test_query_preprocess.py` 删除以下四个测试函数（判据已迁 `Admitter`，本文件不再覆盖）：

- `test_run_classifies_missing_info`（约第 54 行）
- `test_run_missing_info_carries_clarify_question`（约第 93 行）
- `test_run_classifies_out_of_scope`（约第 133 行）
- `test_run_accepts_out_of_scope_in_schema`（约第 143 行）

- [ ] **Step 7: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_query_preprocess.py -v
```
Expected: 全部 PASS。新 `slim` 测试绿；保留的 retrievable/pending_split/ambiguous/other/fallback/no-history/signature/retrieval-context 测试绿。

- [ ] **Step 8: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。`test_doc_workflow.py` 的 `test_out_of_scope_responds_without_retrieval_or_clarify`、`test_missing_info_clarifies_without_retrieval` 等通过 `wf.qa.classify = fake_classify` 整体 stub classify，不受 preprocessor 瘦身影响；`test_qa_capability.py` 的 classify 测试通过 stub `preprocessor.run` + `admitter.run`，也不受影响。

- [ ] **Step 9: 提交**

```powershell
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "refactor: slim QueryPreprocessor 6→4 (move out_of_scope/missing_info to Admitter)

删 out_of_scope/missing_info 两段判据 + 枚举值，preprocessor 只判检索结构/难度
4 类（retrievable/pending_split/ambiguous/other）。可答性轴已由 Admitter 前置闸
接管（Task 3），瘦身不丢能力。失败仍降 retrievable（不变）。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 把 `Admitter` 接进 `qa.explain`（宽召回后插 admit，非 ok 抛异常）

`qa.explain` 在宽召回后、出教案前插一次 `admit`。非 ok → 抛 `OutOfScope` / `MissingInfo`（镜像 `EmptySkeleton` 的异常驱动控制流，由 Task 6 的 `explain_branch` 接住）；ok → 进 outline（原逻辑不变）。`Admitter` 失败降级 ok（放行），不阻塞。

**Files:**
- Modify: `core/workflow/qa_capability.py`（`explain` 方法体在宽召回之后加 admit 段）
- Test: `tests/test_qa_capability.py`（加 explain admit 抛异常测试）

**Interfaces:**
- Consumes：Task 1 的 `Admitter`；Task 2 的 `OutOfScope` / `MissingInfo`。
- Produces：`qa.explain` 可抛 `OutOfScope(query)` / `MissingInfo(clarify_question)`（Task 6 的 `explain_branch` catch）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_qa_capability.py` **末尾**追加（文件已 `from core.workflow.qa_capability import EmptySkeleton`、已 `import pytest`；此处补 import `OutOfScope`、`MissingInfo`、`AdmitVerdict`——`AdmitVerdict` 已在 Task 3 顶部 import 过，`OutOfScope`/`MissingInfo` 已在 Task 2 顶部 import 过，无需重复）：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "explain_admit" -v
```
Expected: 全部 FAIL——`explain` 现在未调 `admitter`，`fake_admit` 不会被调用，`OutOfScope`/`MissingInfo` 不抛（前两个测试 fail）；后两个走到 outline 但 `boom_admit` 未被调用（测试逻辑仍能 raise EmptySkeleton，但断言意图是验证 admit 被调——靠 Step 1 的 `fake_admit`/`boom_admit` 是否被调间接验证。为更精确，先看 FAIL 形态：前两个 fail 因为不抛预期异常；后两个可能 pass 因为 EmptySkeleton 仍会被 raise。若后两个 pass，不影响——重点在前两个 fail 驱动实现）。

- [ ] **Step 3: 改 `explain` 方法体**

`core/workflow/qa_capability.py` 的 `explain` 方法（约第 238–276 行）。在宽召回 + passages 抽取之后、出教案（`toc_hint` / `outliner.run`）之前插 admit 段。

把这段（约第 246–256 行）：

```python
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
```

改为：

```python
        # 1. 宽覆盖召回（内部，不发流事件——空教案时要静默落 agent，别先污染 UI）
        located = await self._explain_recall(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 1.5 可答性闸：吃宽召回片段判 ok/missing_info/out_of_scope；非 ok 抛异常
        # 由 explain_branch 接住拒答/反问。admit 失败降级 ok（放行），不阻塞。
        verdict = await self.admitter.run(query, passages)
        if verdict.verdict == "out_of_scope":
            raise OutOfScope(query)
        if verdict.verdict == "missing_info":
            raise MissingInfo(verdict.clarify_question)

        # 2. 出教案：教学维度词表 + 书的 TOC 提示（单书才有，多书/未选 → []）
        toc_hint = self._book_chapters(book_titles)
        outline = await self.outliner.run(query, passages, toc_hint)
        if not outline:
            raise EmptySkeleton(query)
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k "explain" -v
```
Expected: 全部 PASS（新 explain_admit 测试 + 现有 explain outline/empty/teach 测试均绿——现有 explain 测试 stub 了 `_explain_recall` / `outliner.run` / `_teach_synthesize`，但没 stub `admitter.run`；`admitter` 默认用 `FakeLLM`（本文件 `_qa()` 用的 `FakeLLM.acomplete` raise `AssertionError`），admit 会抛错→降级 ok→放行→走 outline。需确认现有 explain 测试的 `FakeLLM` 行为：本文件顶部 `FakeLLM.acomplete` 是 `raise AssertionError("不应被调用")`。admit 调它 → RuntimeError → except 捕获 → 降级 ok → 放行。所以现有 explain 测试仍绿，因为 admit 失败降级 ok 不影响后续 outline 流程。✓）。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```powershell
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: wire Admitter into qa.explain (admit raises OutOfScope/MissingInfo)

explain 宽召回后、出教案前插一次 admit；非 ok 抛 OutOfScope/MissingInfo
（镜像 EmptySkeleton 异常驱动控制流，由 explain_branch 接住）。admit 失败
降级 ok（放行），不阻塞。修掉"库外/信息不足的 explain 问题溜进 agent 脑补"
的根因——explain 路至此也被可答性闸覆盖。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `doc_workflow` 分支接线（explain_branch 接异常 + out_of_scope_branch 用常量）

`explain_branch` catch `OutOfScope` → 拒答 finalize + 写 `category=out_of_scope`；catch `MissingInfo` → 反问 finalize + 写 `category=missing_info`（缺 clarify_question 用 `REFUSAL_FALLBACK`）；`EmptySkeleton` 仍落 agent（原样保留）。`out_of_scope_branch` 改用 `REFUSAL_TEXT` 常量（去内联话术）。写 `category` 入 ctx 让 finalize metadata 带上（供评测算分类准确率）。

**Files:**
- Modify: `core/workflow/doc_workflow.py`（import 加 4 符号；`explain_branch` 改；`out_of_scope_branch` 改）
- Test: `tests/test_doc_workflow.py`（加 explain_branch 三异常分支测试 + out_of_scope_branch 用 REFUSAL_TEXT 回归）

**Interfaces:**
- Consumes：Task 2 的 `OutOfScope` / `MissingInfo` / `REFUSAL_TEXT` / `REFUSAL_FALLBACK`；Task 5 的 `qa.explain` 抛异常行为。
- Produces：`explain_branch` 可终结于拒答/反问（带 `category` metadata）；`out_of_scope_branch` 话术单一来源 `REFUSAL_TEXT`。workflow 对外行为：explain 路库外/信息不足不再落 agent 脑补，而是拒答/反问。

- [ ] **Step 1: 写失败测试**

在 `tests/test_doc_workflow.py` **末尾**追加（文件已 `from core.workflow.doc_workflow import DocQueryWorkflow`；补 import 异常/常量）：

```python
# ── explain_branch：catch OutOfScope/MissingInfo + EmptySkeleton 仍落 agent（Task 6）──
from core.workflow.qa_capability import (
    EmptySkeleton, OutOfScope, MissingInfo, REFUSAL_TEXT, REFUSAL_FALLBACK,
)


async def test_explain_out_of_scope_refuses_with_category():
    # explain admit 判库外 → 抛 OutOfScope → explain_branch 拒答 + category=out_of_scope
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "PostgreSQL的MVCC"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise OutOfScope(query)

    async def fake_agent(ctx, query, book_titles):
        raise AssertionError("agent 不应被调用（库外应直接拒答）")

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("retrieve 不应被调用")

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent
    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="PostgreSQL的MVCC", memory=FakeMemory())
    assert str(result.response) == REFUSAL_TEXT
    assert result.source_nodes == []
    assert result.metadata.get("category") == "out_of_scope"
    assert result.metadata.get("intent") == "explain"


async def test_explain_missing_info_clarifies_with_category():
    # explain admit 判信息不足 → 抛 MissingInfo(反问) → explain_branch 反问 + category=missing_info
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "这个索引的应用场景"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise MissingInfo("你说的「这个索引」指哪一个？B+树还是全文索引？")

    async def fake_agent(ctx, query, book_titles):
        raise AssertionError("agent 不应被调用（信息不足应直接反问）")

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("retrieve 不应被调用")

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent
    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert str(result.response) == "你说的「这个索引」指哪一个？B+树还是全文索引？"
    assert result.source_nodes == []
    assert result.metadata.get("category") == "missing_info"


async def test_explain_missing_info_without_clarify_uses_fallback():
    # MissingInfo 缺 clarify_question → 用 REFUSAL_FALLBACK 兜底反问
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "这个索引"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise MissingInfo("")                      # 缺反问句

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain

    result = await wf.run(query="这个索引", memory=FakeMemory())
    assert str(result.response) == REFUSAL_FALLBACK
    assert result.metadata.get("category") == "missing_info"


async def test_explain_empty_skeleton_still_falls_to_agent():
    # 回归：EmptySkeleton 不被 OutOfScope/MissingInfo catch 截胡，仍落 agent 兜底
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "讲讲X"}'])
    wf = _wf(llm)

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        raise EmptySkeleton(query)

    agent_called = {"v": False}

    async def fake_agent(ctx, query, book_titles):
        agent_called["v"] = True
        return "agent 兜底答案", ["n1"]

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa_agent.run = fake_agent

    result = await wf.run(query="讲讲X", memory=FakeMemory())
    assert agent_called["v"] is True
    assert str(result.response) == "agent 兜底答案"
    assert result.source_nodes == ["n1"]


async def test_out_of_scope_branch_uses_refusal_text_constant():
    # 回归：other 路库外分支话术 = REFUSAL_TEXT（单一来源，不另写一句）
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "MongoDB分片"}'])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("库外不应检索")

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("out_of_scope", reason="库外")
    wf.qa.classify = fake_classify

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="MongoDB分片", memory=FakeMemory())
    assert str(result.response) == REFUSAL_TEXT
    assert result.metadata.get("category") == "out_of_scope"
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -k "explain_out_of_scope or explain_missing_info or explain_empty_skeleton or out_of_scope_branch_uses" -v
```
Expected: FAIL——`explain_branch` 现在只 catch `EmptySkeleton`，`OutOfScope`/`MissingInfo` 会冒泡到 workflow 框架（不被任何 step 接住 → 报错）；`out_of_scope_branch` 仍用内联话术，与 `REFUSAL_TEXT` 常量值相同可能巧合 pass，但 `explain_*` 测试会 fail 驱动实现。

- [ ] **Step 3: 更新 import**

`core/workflow/doc_workflow.py` 第 41–47 行的 import 块：

```python
from core.workflow.qa_capability import (  # noqa: F401  (事件类 re-export 供 api 层 import)
    AnswerDeltaEvent,
    EmptySkeleton,
    QaCapability,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
```

改为：

```python
from core.workflow.qa_capability import (  # noqa: F401  (事件类 re-export 供 api 层 import)
    AnswerDeltaEvent,
    EmptySkeleton,
    MissingInfo,
    OutOfScope,
    QaCapability,
    REFUSAL_FALLBACK,
    REFUSAL_TEXT,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
```

- [ ] **Step 4: 改 `out_of_scope_branch` 用常量**

`core/workflow/doc_workflow.py` 的 `out_of_scope_branch`（约第 269–278 行）：

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

改为：

```python
    @step
    async def out_of_scope_branch(self, ctx: Context, ev: OutOfScopeEvent) -> FinalizeEvent:
        # 库外：探测召回片段与问题主题无关 → 共享拒答话术（与 explain OutOfScope catch 同源）。
        return FinalizeEvent(answer=REFUSAL_TEXT, source_nodes=[])
```

- [ ] **Step 5: 改 `explain_branch` 接两异常**

`core/workflow/doc_workflow.py` 的 `explain_branch`（约第 280–294 行）：

```python
    @step
    async def explain_branch(self, ctx: Context, ev: ExplainEvent) -> FinalizeEvent:
        # explain：讲清楚精修工作流。空骨架 → 落有界 agent 多轮探索 → agent 再失败 → 单轮兜底。
        rewritten = await ctx.store.get("rewritten_query")
        book_titles = await ctx.store.get("book_titles")
        try:
            answer, nodes = await self.qa.explain(ctx, rewritten, book_titles)
        except EmptySkeleton:
            logger.info("explain: 空骨架，落 agent 兜底")
            try:
                answer, nodes = await self.qa_agent.run(ctx, rewritten, book_titles)
            except Exception as exc:
                logger.warning("explain agent 兜底失败，降级单轮：%s", exc)
                answer, nodes = await self.qa.retrieve(ctx, rewritten, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

改为：

```python
    @step
    async def explain_branch(self, ctx: Context, ev: ExplainEvent) -> FinalizeEvent:
        # explain：讲清楚精修工作流。
        # - admit 判库外 → OutOfScope → 拒答 finalize + 写 category（供评测算分类准确率）
        # - admit 判信息不足 → MissingInfo → 反问 finalize + 写 category
        # - 空骨架 → 落有界 agent 多轮探索 → agent 再失败 → 单轮兜底
        rewritten = await ctx.store.get("rewritten_query")
        book_titles = await ctx.store.get("book_titles")
        try:
            answer, nodes = await self.qa.explain(ctx, rewritten, book_titles)
        except OutOfScope:
            logger.info("explain: admit 判库外，拒答")
            await ctx.store.set("category", "out_of_scope")
            return FinalizeEvent(answer=REFUSAL_TEXT, source_nodes=[])
        except MissingInfo as e:
            logger.info("explain: admit 判信息不足，反问")
            await ctx.store.set("category", "missing_info")
            return FinalizeEvent(
                answer=e.clarify_question or REFUSAL_FALLBACK, source_nodes=[]
            )
        except EmptySkeleton:
            logger.info("explain: 空骨架，落 agent 兜底")
            try:
                answer, nodes = await self.qa_agent.run(ctx, rewritten, book_titles)
            except Exception as exc:
                logger.warning("explain agent 兜底失败，降级单轮：%s", exc)
                answer, nodes = await self.qa.retrieve(ctx, rewritten, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

- [ ] **Step 6: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -v
```
Expected: 全部 PASS。新 explain_branch 异常分支测试绿；`test_out_of_scope_branch_uses_refusal_text_constant` 绿；现有 `test_explain_intent_routes_to_explain_branch`（stub `qa.explain` 成功返回）绿；现有 `test_out_of_scope_responds_without_retrieval_or_clarify` 绿（话术值不变，只是来源改成常量）。

- [ ] **Step 7: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。`scripts/check_layering.py` 守卫不受影响（core 内部改动，未引入 api 反向依赖）。

- [ ] **Step 8: 提交**

```powershell
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat: wire explain_branch to catch OutOfScope/MissingInfo + use REFUSAL_TEXT

explain_branch catch OutOfScope→拒答finalize+写category=out_of_scope、
catch MissingInfo→反问finalize+写category=missing_info（缺clarify用REFUSAL_FALLBACK），
EmptySkeleton 仍落 agent（原样保留）。out_of_scope_branch 改用 REFUSAL_TEXT 常量
（话术单一来源，与 explain OutOfScope catch 同源）。写 category 入 ctx 让 finalize
metadata 带上，供评测算分类准确率。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

### 1. Spec coverage

| spec 条目 | 落在哪个 Task |
|---|---|
| `Admitter`（新）决策单元 | Task 1 |
| `Admitter.run(query, passages) -> AdmitVerdict` 接口 | Task 1 |
| `AdmitVerdict` schema（verdict/reason/clarify_question） | Task 1 |
| prompt 判据原样搬（含铁律/优先级/反例） | Task 1（`_ADMIT_PROMPT`） |
| `qa.classify` 内嵌 admit（probe→admit→preprocessor） | Task 3 |
| `qa.explain` 宽召回后插 admit、非 ok 抛异常 | Task 5 |
| `QueryPreprocessor` 瘦身 6→4 | Task 4 |
| `explain_branch` 接 `OutOfScope`/`MissingInfo` + 写 category | Task 6 |
| `out_of_scope_branch` 拒答话术抽共享常量 | Task 2（常量）+ Task 6（引用） |
| `OutOfScope`/`MissingInfo` 异常放 `qa_capability.py` | Task 2 |
| `REFUSAL_TEXT`/`REFUSAL_FALLBACK` 共享常量 | Task 2 |
| 降级=放行（admit 失败→ok） | Task 1（`Admitter.run` except） |
| 降级=放行（explain admit ok 但空骨架→agent） | Task 6（`EmptySkeleton` 保留） |
| 降级=放行（preprocessor 失败→retrievable，不变） | Task 4（保留现有 except） |
| 话术归一：clarify_question 出自 Admitter | Task 1（`AdmitVerdict.clarify_question`）+ Task 3/5（透传）+ Task 6（MissingInfo catch 取用） |
| 测试：mock LLM 验解析/接线/降级 | Task 1（Admitter）/ Task 3（classify）/ Task 5（explain）/ Task 6（branches）/ Task 4（preprocessor 瘦身） |
| 测试：other 路 workflow 边界仍出 6 类 | Task 6 Step 6（现有 `test_out_of_scope_responds_*`、`test_missing_info_clarifies_*` 回归绿） |
| 不重排热路径（probe/宽召回位置不动） | Task 3/5（只在证据产生处加 admit 调用） |
| `ambiguous` 不迁 | Task 4（保留在 4 类枚举内） |
| Non-goal：不统一/前移 probe | Global Constraints（probe 不统一/前移） |
| Non-goal：不运行时再生成/多轮校验 | Task 1（admit 一次判定） |
| Non-goal：QaAgent 库外拒答补丁 | 未出现在任何 Task（明确留后续，见"已知缺口"） |

无遗漏。

### 2. Placeholder scan

- 无 "TBD" / "TODO" / "implement later" / "add appropriate error handling"。
- 每个步骤都给了完整代码块或精确的"改为"前后对照。
- 测试代码都是可直接运行的具体用例，无 "Write tests for the above" 占位。
- 异常/常量/方法签名在定义 Task 与消费 Task 间一致（见下）。

### 3. Type consistency

- `AdmitVerdict.verdict: Literal["ok","missing_info","out_of_scope"]` —— Task 1 定义，Task 3/5 消费（`verdict.verdict == "out_of_scope"` / `"missing_info"`），一致。
- `Admitter.run(query: str, passages: list[str]) -> AdmitVerdict` —— Task 1 定义，Task 3 调 `self.admitter.run(clean_query, [retrieval_context])`，Task 5 调 `self.admitter.run(query, passages)`，签名一致。
- `OutOfScope(query)` —— Task 2 定义 `class OutOfScope(Exception)`（通过 `Exception` 基类 `args` 携带 query），Task 5 `raise OutOfScope(query)`，Task 6 `except OutOfScope:`（不取 `e`，只用 `ctx.store.set("category",...)` + `REFUSAL_TEXT`），一致。
- `MissingInfo(clarify_question)` —— Task 2 定义 `.clarify_question` 属性，Task 5 `raise MissingInfo(verdict.clarify_question)`，Task 6 `except MissingInfo as e: ... e.clarify_question`，属性名一致。
- `REFUSAL_TEXT` / `REFUSAL_FALLBACK` —— Task 2 定义为模块级 `str` 常量，Task 6 import 并在 `out_of_scope_branch` / `explain_branch` 引用，一致。
- `PreprocessResult(category, reason, clarify_question)` —— Task 3 `qa.classify` 短路时构造 `PreprocessResult("out_of_scope", verdict.reason)` / `PreprocessResult("missing_info", verdict.reason, verdict.clarify_question)`，与现有 `PreprocessResult` dataclass 字段顺序一致（`category, reason="", clarify_question=""`）。
- `QaCapability.admitter` —— Task 3 `__init__` 加 `self.admitter = Admitter(llm)`，Task 5 `explain` 用 `self.admitter.run(...)`，一致。
- `qa.classify` 对外签名 `(clean_query, book_titles=None, probe=True) -> PreprocessResult` —— Task 3 改方法体但签名不变，`doc_workflow.preprocess` 调用处 `result = await self.qa.classify(denoised_query, book_titles, probe=self._probe)` 不受影响，一致。
- `qa.explain` 对外签名 `(ctx, query, book_titles) -> tuple[str, list]` —— Task 5 改方法体但签名不变，`doc_workflow.explain_branch` 调用处 `await self.qa.explain(ctx, rewritten, book_titles)` 不受影响，一致。

无类型/签名漂移。

### 4. 已知缺口（本计划不覆盖，spec 明确留后续）

- **QaAgent 库外拒答补丁**：本刀的防御纵深后手，另开一刀（最小：镜像 `AutoAgent` 那段收场搬进 `QA_AGENT_SYSTEM_PROMPT`）。本计划不实现。
- **可答性闸的真实冷烟**：库外 4 类经 explain 路判 out_of_scope、且"讲懂MySQL概念"不被误判——需 `DEEPSEEK_API_KEY` + 索引，人读，不在 mock LLM 单测范围。
- **missing_info 的文本可判半**：指代不明那半其实文本就能判，本刀仍统一走证据判（一致优先）；将来若要省一次检索再分。
