# Book RAG 评测体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 抛弃现有手写 `eval/`，参考 `legacy/evals` 的 ragas 0.4.3 地道用法，重建一套通用 RAG 质量评测体系：TestsetGenerator 自动生成 book 测试集 + collections 五指标 + `@experiment` runner，端到端评测 `BookRagWorkflow`（SUT 接口预留 Agent）。

**Architecture:** 评测侧 LLM/embedding（ragas 原生 `llm_factory`→智谱 GLM + `HuggingFaceEmbeddings`）与被测系统（项目 DeepSeek + `BookRagWorkflow`）解耦。被测系统藏在 `RagSystem` 协议后，`map_workflow_result` 把 workflow 返回值（`Response`/`ClarifyResult`）归一成 `RagOutput(response, retrieved_contexts, outcome)`。指标字段映射是纯函数，逐行打分 `score_row` 与聚合 `aggregate` 也是纯逻辑——这些全部 TDD；真实模型构造、`@experiment` 装配、测试集生成是集成脚本，靠 smoke 手动验证。

**Tech Stack:** Python 3.12 / ragas 0.4.3（`metrics.collections`、`experiment`、`Dataset`、`LocalCSVBackend`、`TestsetGenerator`）/ llama-index（SUT）/ pytest + pytest-asyncio（`asyncio_mode=auto`）。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `eval/__init__.py` | 包标记（置空） |
| `eval/config.py` | 评测侧 LLM(`make_eval_llm`)、embedding(`make_eval_embeddings`)、路径常量 |
| `eval/sut.py` | `RagOutput`、`RagSystem` 协议、`map_workflow_result`（纯）、`BookRagWorkflowSystem` |
| `eval/metrics.py` | 5 个指标的字段映射纯函数 + `MetricSpec` + `build_metric_specs` |
| `eval/run_eval.py` | `load_testset`、`_row_to_dict`、`score_row`、`aggregate`（纯）、`main`（`@experiment` 装配） |
| `eval/generate_testset.py` | `chunks_to_langchain`（纯）+ 生成脚本 main |
| `eval/dataset/.gitkeep` | 测试集目录占位（`testset.draft.jsonl` 生成、`testset.jsonl` 人工校验后提交） |
| `tests/test_eval_sut.py` | `map_workflow_result` 单测 |
| `tests/test_eval_metrics.py` | 字段映射函数 + `build_metric_specs` 装配单测 |
| `tests/test_eval_run.py` | `aggregate`、`score_row`、`_row_to_dict` 单测 |
| `tests/test_eval_generate.py` | `chunks_to_langchain` 单测 |

**删除**（均为未提交的 `?? eval/` 文件）：`eval/ablation.py`、`eval/questions.json`、`eval/ablation_report.md`、`eval/ablation_results.json`。

---

## Task 0: 清理旧 eval + 包骨架

**Files:**
- Delete: `eval/ablation.py`, `eval/questions.json`, `eval/ablation_report.md`, `eval/ablation_results.json`
- Create: `eval/__init__.py`（置空）, `eval/dataset/.gitkeep`

- [ ] **Step 1: 删除旧文件并建目录**

Run:
```bash
git rm -f --ignore-unmatch eval/ablation.py eval/questions.json eval/ablation_report.md eval/ablation_results.json 2>$null
rm -f eval/ablation.py eval/questions.json eval/ablation_report.md eval/ablation_results.json
mkdir -p eval/dataset
```
（`eval/` 当前未跟踪，`git rm` 可能无匹配——`rm -f` 兜底删除即可。）

- [ ] **Step 2: 重置包标记 + 目录占位**

把 `eval/__init__.py` 内容清空（写入空字符串）。新建 `eval/dataset/.gitkeep` 内容为空。

- [ ] **Step 3: 确认旧文件已不在**

Run: `ls eval`
Expected: 只剩 `__init__.py`、`dataset/`（无 `ablation.py`/`questions.json`）。

- [ ] **Step 4: Commit**

```bash
git add -A eval/
git commit -m "chore(eval): 清理旧 ablation 实现，重置 eval 包骨架"
```

---

## Task 1: config.py — 评测侧模型与路径

**Files:**
- Create: `eval/config.py`
- Test: 无单测（工厂依赖网络/模型加载，留给集成 smoke）。仅做 import 冒烟。

- [ ] **Step 1: 写 config.py**

```python
"""评测侧配置：judge LLM / embedding / 路径。

评测侧用 ragas 原生 llm_factory（instructor 结构化输出，collections 指标必需），
沿用 legacy 的智谱 GLM，与被测系统（项目 DeepSeek）解耦。
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(EVAL_DIR, "dataset")
TESTSET_PATH = os.path.join(DATASET_DIR, "testset.jsonl")
TESTSET_DRAFT_PATH = os.path.join(DATASET_DIR, "testset.draft.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# ── 评测侧模型 ────────────────────────────────────────
EVAL_LLM_MODEL = "glm-4-flash"
EVAL_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
EVAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def make_eval_llm():
    """评测 judge LLM：llm_factory + 智谱 OpenAI 兼容端点。"""
    import openai
    from ragas.llms import llm_factory

    client = openai.AsyncOpenAI(
        base_url=EVAL_LLM_BASE_URL,
        api_key=os.getenv("ZHIPU_API_KEY"),
    )
    return llm_factory(EVAL_LLM_MODEL, client=client)


def make_eval_embeddings():
    """评测 embedding：ragas HuggingFaceEmbeddings。"""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=EVAL_EMBED_MODEL)
```

- [ ] **Step 2: import 冒烟**

Run: `python -c "from eval.config import make_eval_llm, make_eval_embeddings, TESTSET_PATH, RESULTS_DIR; print('ok', TESTSET_PATH)"`
Expected: 打印 `ok` + testset 路径，无 ImportError。

- [ ] **Step 3: Commit**

```bash
git add eval/config.py
git commit -m "feat(eval): config.py 评测侧 LLM/embedding 与路径常量"
```

---

## Task 2: sut.py — 被测系统抽象

**Files:**
- Create: `eval/sut.py`
- Test: `tests/test_eval_sut.py`

- [ ] **Step 1: 写 map_workflow_result 的失败测试**

`tests/test_eval_sut.py`：

```python
from eval.sut import map_workflow_result, RagOutput


class _Node:
    def __init__(self, text): self._t = text
    def get_content(self): return self._t


class _NodeWithScore:
    def __init__(self, text): self.node = _Node(text)


class _Response:
    """模拟 llama-index Response。"""
    def __init__(self, response, source_nodes):
        self.response = response
        self.source_nodes = source_nodes


class _ClarifyResult:
    """类名须为 ClarifyResult 以触发分流分支。"""
    def __init__(self, query, clarify_reason):
        self.query = query
        self.clarify_reason = clarify_reason


# 让伪类的类名匹配映射逻辑
_ClarifyResult.__name__ = "ClarifyResult"


def test_answered_extracts_text_and_contexts():
    resp = _Response("MVCC 通过 undo log 实现", [_NodeWithScore("片段A"), _NodeWithScore("片段B")])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "answered"
    assert out.response == "MVCC 通过 undo log 实现"
    assert out.retrieved_contexts == ["片段A", "片段B"]


def test_empty_when_no_nodes():
    resp = _Response("", [])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "empty"
    assert out.retrieved_contexts == []


def test_clarify_branch():
    cr = _ClarifyResult("这个索引", "指代不明")
    out = map_workflow_result(cr, response_cls=_Response)
    assert out.outcome == "clarify"
    assert out.response == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_eval_sut.py -v`
Expected: FAIL（`ImportError: cannot import name 'map_workflow_result'`）。

- [ ] **Step 3: 写 sut.py**

```python
"""被测系统（SUT）抽象：协议 + BookRagWorkflow 适配器。

map_workflow_result 把 workflow 返回值（Response / ClarifyResult）归一成 RagOutput，
是纯函数便于单测；BookRagWorkflowSystem 负责实际运行与异常兜底。
"""
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | clarify | split | empty | error


@runtime_checkable
class RagSystem(Protocol):
    async def answer(self, query: str) -> RagOutput: ...


def map_workflow_result(result, response_cls=None) -> RagOutput:
    """把 BookRagWorkflow.run() 的返回值映射为 RagOutput。

    response_cls 仅供测试注入伪 Response；生产默认用 llama-index Response。
    """
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813

    # clarify / split 分支统一返回 ClarifyResult
    if result.__class__.__name__ == "ClarifyResult":
        return RagOutput(response="", retrieved_contexts=[], outcome="clarify")

    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(response=text, retrieved_contexts=[], outcome="empty")
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(response=text, retrieved_contexts=contexts, outcome="answered")

    return RagOutput(response=str(result), retrieved_contexts=[], outcome="empty")


class BookRagWorkflowSystem:
    """包装 core.workflow.book_rag.BookRagWorkflow，实现 RagSystem 协议。"""

    def __init__(self, index_manager, llm, similarity_top_k: int = 5, timeout: float = 120.0):
        from core.workflow.book_rag import BookRagWorkflow

        self._workflow = BookRagWorkflow(
            index_manager=index_manager,
            llm=llm,
            similarity_top_k=similarity_top_k,
            timeout=timeout,
        )

    async def answer(self, query: str) -> RagOutput:
        try:
            result = await self._workflow.run(query=query)
        except Exception as e:  # noqa: BLE001 — eval 需吞掉单条异常，记 error 不中断
            return RagOutput(response=f"{type(e).__name__}: {e}",
                             retrieved_contexts=[], outcome="error")
        return map_workflow_result(result)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_eval_sut.py -v`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add eval/sut.py tests/test_eval_sut.py
git commit -m "feat(eval): SUT 抽象 + map_workflow_result 归一化"
```

---

## Task 3: metrics.py — 指标字段映射与装配

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

注：5 个 `collections` 指标的 `ascore` 参数已核实——
`Faithfulness(user_input, response, retrieved_contexts)`、
`AnswerRelevancy(user_input, response)`、
`ContextPrecisionWithReference(user_input, reference, retrieved_contexts)`、
`ContextRecall(user_input, retrieved_contexts, reference)`、
`FactualCorrectness(response, reference)`。

- [ ] **Step 1: 写映射函数与装配的失败测试**

`tests/test_eval_metrics.py`：

```python
from eval.metrics import METRIC_KWARGS, MetricSpec
from eval.sut import RagOutput


def _row():
    return {"user_input": "Q", "reference": "REF"}


def _out():
    return RagOutput(response="ANS", retrieved_contexts=["c1", "c2"], outcome="answered")


def test_faithfulness_kwargs():
    kw = METRIC_KWARGS["faithfulness"](_row(), _out())
    assert kw == {"user_input": "Q", "response": "ANS", "retrieved_contexts": ["c1", "c2"]}


def test_answer_relevancy_kwargs_omits_contexts():
    kw = METRIC_KWARGS["answer_relevancy"](_row(), _out())
    assert kw == {"user_input": "Q", "response": "ANS"}


def test_context_precision_kwargs():
    kw = METRIC_KWARGS["context_precision"](_row(), _out())
    assert kw == {"user_input": "Q", "reference": "REF", "retrieved_contexts": ["c1", "c2"]}


def test_context_recall_kwargs():
    kw = METRIC_KWARGS["context_recall"](_row(), _out())
    assert kw == {"user_input": "Q", "retrieved_contexts": ["c1", "c2"], "reference": "REF"}


def test_factual_correctness_kwargs():
    kw = METRIC_KWARGS["factual_correctness"](_row(), _out())
    assert kw == {"response": "ANS", "reference": "REF"}


def test_metric_spec_dataclass():
    spec = MetricSpec(name="x", metric=object(), kwargs=lambda r, o: {})
    assert spec.name == "x"
    assert callable(spec.kwargs)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: FAIL（`ImportError: cannot import name 'METRIC_KWARGS'`）。

- [ ] **Step 3: 写 metrics.py**

```python
"""5 个 ragas collections 指标：字段映射（纯函数）+ MetricSpec 装配。

字段映射与指标构造分离——映射可离线单测，构造需真实 InstructorLLM（集成 smoke）。
"""
from dataclasses import dataclass
from typing import Callable


# ── 字段映射：row(dict) + RagOutput → ascore kwargs ──
def _faithfulness_kwargs(row, out):
    return {"user_input": row["user_input"], "response": out.response,
            "retrieved_contexts": out.retrieved_contexts}


def _answer_relevancy_kwargs(row, out):
    return {"user_input": row["user_input"], "response": out.response}


def _context_precision_kwargs(row, out):
    return {"user_input": row["user_input"], "reference": row["reference"],
            "retrieved_contexts": out.retrieved_contexts}


def _context_recall_kwargs(row, out):
    return {"user_input": row["user_input"], "retrieved_contexts": out.retrieved_contexts,
            "reference": row["reference"]}


def _factual_correctness_kwargs(row, out):
    return {"response": out.response, "reference": row["reference"]}


METRIC_KWARGS: dict[str, Callable] = {
    "faithfulness": _faithfulness_kwargs,
    "answer_relevancy": _answer_relevancy_kwargs,
    "context_precision": _context_precision_kwargs,
    "context_recall": _context_recall_kwargs,
    "factual_correctness": _factual_correctness_kwargs,
}

# 指标均值聚合的固定顺序
METRIC_NAMES = list(METRIC_KWARGS.keys())


@dataclass
class MetricSpec:
    name: str
    metric: object
    kwargs: Callable  # (row: dict, out: RagOutput) -> dict


def build_metric_specs(llm, embeddings) -> list[MetricSpec]:
    """构造 5 个 collections 指标（llm/embeddings 须为真实 ragas 对象）。"""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
    )

    return [
        MetricSpec("faithfulness", Faithfulness(llm=llm), METRIC_KWARGS["faithfulness"]),
        MetricSpec("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=embeddings),
                   METRIC_KWARGS["answer_relevancy"]),
        MetricSpec("context_precision", ContextPrecisionWithReference(llm=llm),
                   METRIC_KWARGS["context_precision"]),
        MetricSpec("context_recall", ContextRecall(llm=llm), METRIC_KWARGS["context_recall"]),
        MetricSpec("factual_correctness", FactualCorrectness(llm=llm, mode="f1"),
                   METRIC_KWARGS["factual_correctness"]),
    ]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): 指标字段映射纯函数 + MetricSpec 装配"
```

---

## Task 4: run_eval.py 核心 — 打分与聚合（纯逻辑 TDD）

**Files:**
- Create: `eval/run_eval.py`（本任务只实现 `load_testset`/`_row_to_dict`/`score_row`/`aggregate`，`main` 留到 Task 5）
- Test: `tests/test_eval_run.py`

注：`ascore` 返回 `MetricResult`，分值在 `.value`。

- [ ] **Step 1: 写失败测试**

`tests/test_eval_run.py`：

```python
from dataclasses import dataclass

from eval.metrics import MetricSpec
from eval.run_eval import aggregate, score_row, _row_to_dict
from eval.sut import RagOutput


# ── _row_to_dict ──
class _AttrRow:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def test_row_to_dict_from_dict():
    assert _row_to_dict({"user_input": "Q", "reference": "R"})["user_input"] == "Q"


def test_row_to_dict_from_attr_object():
    row = _AttrRow(user_input="Q", reference="R")
    d = _row_to_dict(row)
    assert d["user_input"] == "Q" and d["reference"] == "R"


# ── aggregate ──
def test_aggregate_means_only_over_answered():
    rows = [
        {"outcome": "answered", "faithfulness": 1.0, "answer_relevancy": 0.8,
         "context_precision": 1.0, "context_recall": 0.5, "factual_correctness": 0.6},
        {"outcome": "answered", "faithfulness": 0.0, "answer_relevancy": 0.6,
         "context_precision": 0.0, "context_recall": 0.5, "factual_correctness": 0.4},
        {"outcome": "clarify"},
    ]
    rep = aggregate(rows)
    assert rep["total"] == 3
    assert rep["answered"] == 2
    assert rep["outcome_distribution"] == {"answered": 2, "clarify": 1}
    assert rep["metric_means"]["faithfulness"] == 0.5
    assert rep["metric_means"]["answer_relevancy"] == 0.7


def test_aggregate_ignores_none_scores():
    rows = [
        {"outcome": "answered", "faithfulness": None, "answer_relevancy": 0.4,
         "context_precision": None, "context_recall": None, "factual_correctness": None},
    ]
    rep = aggregate(rows)
    assert rep["metric_means"]["faithfulness"] is None
    assert rep["metric_means"]["answer_relevancy"] == 0.4


# ── score_row ──
@dataclass
class _MetricResult:
    value: float


class _FakeMetric:
    def __init__(self, value): self._v = value
    async def ascore(self, **kw): return _MetricResult(self._v)


class _FakeSUT:
    def __init__(self, out): self._out = out
    async def answer(self, query): return self._out


async def test_score_row_answered_scores_all_metrics():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "answered"
    assert res["faithfulness"] == 0.9
    assert res["response"] == "A"


async def test_score_row_non_answered_skips_metrics():
    out = RagOutput(response="", retrieved_contexts=[], outcome="clarify")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "clarify"
    assert "faithfulness" not in res
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_eval_run.py -v`
Expected: FAIL（`ImportError: cannot import name 'aggregate'`）。

- [ ] **Step 3: 写 run_eval.py 的核心部分**

```python
"""@experiment runner：逐行跑 SUT → 打分 → 聚合。

本文件上半部（load_testset / _row_to_dict / score_row / aggregate）是纯逻辑，
已 TDD；下半部 main 是 @experiment + Dataset 装配，靠集成 smoke 验证。
"""
import json

from eval.metrics import METRIC_NAMES, MetricSpec
from eval.sut import RagOutput, RagSystem


def load_testset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_dict(row) -> dict:
    """把 ragas Dataset 行（可能是 dict / pydantic / 带属性对象）归一成 dict。"""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    # 兜底：按已知字段取属性
    keys = ["user_input", "reference", "reference_contexts"]
    return {k: getattr(row, k, None) for k in keys}


async def score_row(row: dict, sut: RagSystem, metric_specs: list[MetricSpec]) -> dict:
    out: RagOutput = await sut.answer(row["user_input"])
    base = {
        "user_input": row["user_input"],
        "reference": row.get("reference", ""),
        "response": out.response,
        "outcome": out.outcome,
        "num_contexts": len(out.retrieved_contexts),
    }
    if out.outcome != "answered":
        return base
    for spec in metric_specs:
        try:
            result = await spec.metric.ascore(**spec.kwargs(row, out))
            base[spec.name] = result.value
        except Exception as e:  # noqa: BLE001 — 单指标失败不影响其他指标
            base[spec.name] = None
            base[f"{spec.name}_error"] = f"{type(e).__name__}: {e}"
    return base


def aggregate(rows: list[dict]) -> dict:
    """指标均值（仅 answered 行、忽略 None）+ outcome 分布。"""
    outcomes: dict[str, int] = {}
    for r in rows:
        oc = r.get("outcome", "error")
        outcomes[oc] = outcomes.get(oc, 0) + 1
    answered = [r for r in rows if r.get("outcome") == "answered"]
    metric_means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in answered if r.get(name) is not None]
        metric_means[name] = (sum(vals) / len(vals)) if vals else None
    return {
        "total": len(rows),
        "answered": len(answered),
        "outcome_distribution": outcomes,
        "metric_means": metric_means,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_eval_run.py -v`
Expected: 7 passed。

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py tests/test_eval_run.py
git commit -m "feat(eval): 逐行打分 score_row + 聚合 aggregate（纯逻辑 TDD）"
```

---

## Task 5: run_eval.py main — @experiment 装配与 CLI

**Files:**
- Modify: `eval/run_eval.py`（追加 imports 与 `main`）
- Test: 无单测（需真实模型/索引），靠 Step 3 集成 smoke。

注：`@experiment()` 装饰的异步函数通过 `.arun(dataset, name=, **kwargs)` 运行，
`dataset` 须为 `ragas.Dataset(name, backend, data=list[dict])`，额外 kwargs 转发给逐行函数；
`LocalCSVBackend(root_dir)` 负责落盘。

- [ ] **Step 1: 在 run_eval.py 顶部追加 imports**

在现有 `import json` 下方追加：

```python
import argparse
import asyncio
import os
```

- [ ] **Step 2: 在 run_eval.py 末尾追加 main**

```python
async def _run(testset_path: str, limit: int | None) -> dict:
    from ragas import Dataset, experiment
    from ragas.backends import LocalCSVBackend

    from eval.config import RESULTS_DIR, make_eval_embeddings, make_eval_llm
    from eval.metrics import build_metric_specs
    from eval.sut import BookRagWorkflowSystem
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]

    eval_llm = make_eval_llm()
    eval_emb = make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)

    sut = BookRagWorkflowSystem(index_manager=RAGIndexManager(), llm=configure_llm())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    backend = LocalCSVBackend(root_dir=RESULTS_DIR)
    dataset = Dataset(name="book_testset", backend=backend, data=rows)

    @experiment()
    async def book_rag_experiment(row, sut, metric_specs):
        return await score_row(_row_to_dict(row), sut, metric_specs)

    exp = await book_rag_experiment.arun(
        dataset, name="book_rag", sut=sut, metric_specs=metric_specs,
    )
    result_rows = [_row_to_dict(r) for r in exp.to_pandas().to_dict("records")]
    return aggregate(result_rows)


def main():
    parser = argparse.ArgumentParser(description="Book RAG ragas 评测")
    parser.add_argument("--testset", default=None, help="测试集 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    args = parser.parse_args()

    from eval.config import TESTSET_PATH
    path = args.testset or TESTSET_PATH
    report = asyncio.run(_run(path, args.limit))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 集成 smoke（依赖真实索引/网络，前置：book_rag.py 可运行 + chroma_db 有 book 数据 + 一条临时测试集）**

先造一条临时测试集（绕过尚未生成的正式集）：
```bash
mkdir -p eval/dataset
python -c "import json; open('eval/dataset/_smoke.jsonl','w',encoding='utf-8').write(json.dumps({'user_input':'InnoDB 的 MVCC 是怎么实现的','reference':'MVCC 通过 undo log 多版本实现','reference_contexts':['MVCC 通过 undo log...']}, ensure_ascii=False))"
```
Run: `python -m eval.run_eval --testset eval/dataset/_smoke.jsonl --limit 1`
Expected: 打印 JSON 报告，含 `metric_means` 五项与 `outcome_distribution`；`results/` 下出现实验 CSV。
失败排查：若 `outcome` 为 `error`，多半是 `core/workflow/book_rag.py` 半成品（见 Task 7 前置）或 chroma 无数据。

清理：`rm -f eval/dataset/_smoke.jsonl`

- [ ] **Step 4: Commit**

```bash
git add eval/run_eval.py
git commit -m "feat(eval): @experiment runner 装配 + CLI（main）"
```

---

## Task 6: generate_testset.py — 测试集生成（改编 legacy）

**Files:**
- Create: `eval/generate_testset.py`
- Test: `tests/test_eval_generate.py`（只测纯函数 `chunks_to_langchain`）

注：走"复用 chroma 已切块"路线——`TestsetGenerator.generate_with_chunks(chunks=LangChain Document 列表, ...)`。
chroma 配置：`RAGIndexManager(persist_dir="./chroma_db", collection_name="book_knowledge")`，
chunk 元数据含 `book_title/chapter/page/file_path`。

- [ ] **Step 1: 写 chunks_to_langchain 的失败测试**

`tests/test_eval_generate.py`：

```python
from eval.generate_testset import chunks_to_langchain


def test_chunks_to_langchain_wraps_text_and_metadata():
    docs = ["正文1", "正文2"]
    metas = [{"book_title": "MySQL", "chapter": "3"}, {"book_title": "MySQL", "chapter": "4"}]
    out = chunks_to_langchain(docs, metas)
    assert len(out) == 2
    assert out[0].page_content == "正文1"
    assert out[0].metadata["book_title"] == "MySQL"
    assert out[1].metadata["chapter"] == "4"


def test_chunks_to_langchain_skips_empty_text():
    docs = ["正文", "", "  "]
    metas = [{"book_title": "X"}, {"book_title": "X"}, {"book_title": "X"}]
    out = chunks_to_langchain(docs, metas)
    assert len(out) == 1
    assert out[0].page_content == "正文"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_eval_generate.py -v`
Expected: FAIL（`ImportError`）。

- [ ] **Step 3: 写 generate_testset.py**

```python
"""从 book chroma 切片用 ragas TestsetGenerator 生成测试集草稿。

走"复用已切块"路线：把 chroma 片段包成 LangChain Document，喂 generate_with_chunks。
产出 dataset/testset.draft.jsonl，人工校验后另存为 testset.jsonl 供 run_eval 使用。
运行（项目根目录）：python -m eval.generate_testset --size 50
"""
import argparse
import asyncio
import json
import os

from langchain_core.documents import Document as LCDocument


def chunks_to_langchain(documents: list[str], metadatas: list[dict]) -> list[LCDocument]:
    """把 chroma 的 (正文, 元数据) 逐条包成 LangChain Document，跳过空文本。"""
    out: list[LCDocument] = []
    for text, meta in zip(documents, metadatas):
        if not text or not text.strip():
            continue
        out.append(LCDocument(
            page_content=text,
            metadata={
                "book_title": (meta or {}).get("book_title", ""),
                "chapter": (meta or {}).get("chapter", ""),
                "page": (meta or {}).get("page", ""),
                "file_path": (meta or {}).get("file_path", ""),
            },
        ))
    return out


def load_book_chunks() -> list[LCDocument]:
    """从项目 chroma 全量取出 book 切片并转 LangChain Document。"""
    from core.rag.data_loader import RAGIndexManager

    mgr = RAGIndexManager()  # persist_dir=./chroma_db, collection=book_knowledge
    data = mgr.chroma_collection.get(include=["documents", "metadatas"])
    chunks = chunks_to_langchain(data["documents"], data["metadatas"])
    print(f"从 chroma 加载 {len(chunks)} 条 book 切片")
    return chunks


async def generate(size: int) -> None:
    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

    from eval.config import DATASET_DIR, TESTSET_DRAFT_PATH, make_eval_embeddings, make_eval_llm

    chunks = load_book_chunks()
    if not chunks:
        raise SystemExit("chroma 无 book 切片，先入库（python main.py 入库流程）再生成测试集")

    gen_llm = make_eval_llm()
    gen_emb = make_eval_embeddings()

    personas = [
        Persona(
            name="tech_reader",
            role_description="正在阅读技术书的工程师，针对书中具体的技术概念、机制、章节提出有据可查的问题",
        ),
    ]

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb, persona_list=personas)

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=gen_llm), 0.6),
        (MultiHopSpecificQuerySynthesizer(llm=gen_llm), 0.4),
    ]
    # 中文 prompt 适配
    for query, _ in distribution:
        prompts = await query.adapt_prompts("chinese", llm=gen_llm)
        query.set_prompts(**prompts)

    print(f"开始生成测试集（{size} 条）……")
    dataset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=size,
        query_distribution=distribution,
    )
    eval_dataset = dataset.to_evaluation_dataset()

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(TESTSET_DRAFT_PATH, "w", encoding="utf-8") as f:
        for sample in eval_dataset.to_list():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"草稿已写入 {TESTSET_DRAFT_PATH}（共 {len(eval_dataset.to_list())} 条）")
    print("⚠️ 人工校验后另存为 testset.jsonl，再跑 run_eval。")


def main():
    parser = argparse.ArgumentParser(description="生成 book RAG 测试集草稿")
    parser.add_argument("--size", type=int, default=50, help="测试集条数")
    args = parser.parse_args()
    asyncio.run(generate(args.size))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_eval_generate.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add eval/generate_testset.py tests/test_eval_generate.py
git commit -m "feat(eval): TestsetGenerator 测试集生成（复用 chroma 切块）"
```

---

## Task 7: 全量回归 + 文档收尾

**Files:**
- Modify: `eval/__init__.py`（可加简短模块 docstring，可选）
- 可选 Modify: `CLAUDE.md`（修正"GLM"为实际 DeepSeek，并补一行 eval 用法）

- [ ] **Step 1: 跑全部 eval 单测**

Run: `pytest tests/test_eval_sut.py tests/test_eval_metrics.py tests/test_eval_run.py tests/test_eval_generate.py -v`
Expected: 全绿（3+6+7+2 = 18 passed）。

- [ ] **Step 2: 跑整套测试确认未破坏既有用例**

Run: `pytest -q`
Expected: 既有用例不因本次改动失败（`eval/` 旧文件已删，不应有引用残留）。

- [ ] **Step 3: 记录前置条件到计划/README（人读）**

确认以下前置仍成立，供后续真实评测：
- `core/workflow/book_rag.py` 当前为半成品（`assume` 空体、`split` 引用不存在的 `ev.clarify_reason`、脏 import `from sympy.strategies.core import switch`）。**真实端到端评测前需先修复使其可 import/run**——本计划不含此修复。
- chroma_db 需已入库至少一本 book。
- `.env` 需含 `ZHIPU_API_KEY`（评测侧）与 `DEEPSEEK_API_KEY`（SUT 侧）。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(eval): 全量回归 + 前置条件收尾"
```

---

## 完成标准

- 4 个单测文件全绿（18 passed），不破坏既有用例。
- `python -m eval.generate_testset --size N` 能产出 `testset.draft.jsonl`（需 chroma 有数据）。
- 人工校验得到 `testset.jsonl` 后，`python -m eval.run_eval` 打印含五指标均值 + outcome 分布的报告，并在 `results/` 落盘实验 CSV（需 `book_rag.py` 可运行）。
