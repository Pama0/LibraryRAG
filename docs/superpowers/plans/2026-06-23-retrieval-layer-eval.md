# 检索层评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `eval/retrieval/` 子包搭一套确定性、零生成/裁判 LLM 的检索层评测，做 vector vs hybrid 的 Recall@k / nDCG@k 对照，并当后续检索调参的尺子。

**Architecture:** 两阶段。阶段一 `label.py` 离线对 golden 可答类 query 做 pooling（vector∪bm25 候选池）+ DeepSeek 二元判定，冻结成 `golden.retrieval.jsonl`（调 LLM，一次性）。阶段二 `run.py` 读冻结标注，逐 retriever 绕过 front_door 检索、用 `metrics.py` 纯函数算指标、出对比表（零 LLM，反复跑）。

**Tech Stack:** Python 3.12 async；复用 `core.retrieval.make_retriever`、`core.rag.data_loader.RAGIndexManager`、`configs.llm/embedding`；judge 用 `openai.AsyncOpenAI`（DeepSeek 端点，`thinking:disabled`，JSON 输出，镜像 `eval/datagen/fill_reference.py`）；测试用 pytest。

## Global Constraints

- 从项目根目录运行，模块用 `python -m eval.retrieval.xxx`（子模块内相对导入，跨包绝对导入）—— 见 CLAUDE.md Gotchas。
- 所有 I/O 用 `async/await`，函数签名加类型注解（项目 Code Style）。
- 检索层评测自成 `eval/retrieval/` 子包，**数据/结果/代码不混进** `eval/harness`、`eval/datagen`、`eval/dataset`、`eval/results`。
- chunk 标识统一用 `node.node_id`（dense 经 chroma、BM25 经 `TextNode(id_=chroma_id)` 都等于 chroma id，可跨阶段对齐）。
- 常量：`POOL_N = 30`（每路候选深度）、`JUDGE_BATCH = 10`（每批判定 chunk 数）、`CHUNK_TRUNC = 600`（judge 时 chunk 正文截断字数）、`K_VALUES = (1, 3, 5, 10)`。
- 单测放项目根 `tests/`，沿用现有约定（如 `tests/test_eval_compare.py`）。

---

## File Structure

- `eval/retrieval/__init__.py` — 空包标记。
- `eval/retrieval/metrics.py` — 纯函数 `recall_at_k / precision_at_k / mrr / ndcg_at_k` + 每条/聚合辅助。**Task 1**。
- `eval/retrieval/label.py` — 阶段一：pooling + LLM 判定 → `dataset/golden.retrieval.jsonl`。纯辅助 `merge_pool / parse_judgement` 可单测，`main()` 为集成 smoke。**Task 2**。
- `eval/retrieval/run.py` — 阶段二：跑 retriever → 算 → 出 console 表 + CSV。纯辅助 `aggregate` 可单测，检索循环为集成 smoke。**Task 3**。
- `eval/retrieval/dataset/.gitkeep`、`eval/retrieval/results/.gitkeep` — 占位目录。
- `tests/test_retrieval_metrics.py` — Task 1 的纯函数单测。
- `tests/test_retrieval_label_helpers.py` — Task 2 纯辅助单测。
- `tests/test_retrieval_aggregate.py` — Task 3 聚合辅助单测。

---

## Task 1: metrics.py 指标纯函数

**Files:**
- Create: `eval/retrieval/__init__.py`
- Create: `eval/retrieval/metrics.py`
- Create: `eval/retrieval/dataset/.gitkeep`
- Create: `eval/retrieval/results/.gitkeep`
- Test: `tests/test_retrieval_metrics.py`

**Interfaces:**
- Produces:
  - `K_VALUES: tuple[int, ...] = (1, 3, 5, 10)`
  - `recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None`（relevant 为空→None）
  - `precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None`（k<=0→None）
  - `mrr(retrieved: list[str], relevant: set[str]) -> float`（无命中→0.0）
  - `ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None`（relevant 为空→None）

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval_metrics.py`:
```python
import math

import pytest

from eval.retrieval.metrics import (
    K_VALUES,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_partial_and_full():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "x"}
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)   # 命中 a
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)   # 命中 a,c


def test_recall_empty_relevant_is_none():
    assert recall_at_k(["a"], set(), 5) is None


def test_precision_divides_by_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(1 / 2)
    assert precision_at_k(retrieved, relevant, 0) is None


def test_mrr_rank_and_miss():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_single_hit_at_rank2():
    # b 在 index1（rank2）：dcg=1/log2(3)；理想命中数 1：idcg=1/log2(2)=1
    val = ndcg_at_k(["a", "b"], {"b"}, 2)
    assert val == pytest.approx(1 / math.log2(3))


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_empty_relevant_is_none():
    assert ndcg_at_k(["a"], set(), 5) is None


def test_k_values_constant():
    assert K_VALUES == (1, 3, 5, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.retrieval'`

- [ ] **Step 3: Create package + empty dirs**

`eval/retrieval/__init__.py`:
```python
"""检索层评测子包：确定性、零生成/裁判 LLM，做 vector vs hybrid 的 Recall@k/nDCG 对照。"""
```
`eval/retrieval/dataset/.gitkeep`：空文件。
`eval/retrieval/results/.gitkeep`：空文件。

- [ ] **Step 4: Write metrics.py**

`eval/retrieval/metrics.py`:
```python
"""检索层指标（纯函数，零依赖，可离线单测）。

retrieved：按检索序的 chunk_id 列表；relevant：相关 chunk_id 集合。
relevant 为空的 query 不该进来（评测层已过滤），各函数对空 relevant 返回 None。
"""
import math

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def _hit_count(retrieved: list[str], relevant: set[str], k: int) -> int:
    return sum(1 for rid in retrieved[:k] if rid in relevant)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """命中相关数 / 相关总数。relevant 为空 → None（不计入均值）。"""
    if not relevant:
        return None
    return _hit_count(retrieved, relevant, k) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """命中相关数 / k（标准 P@k，分母恒为 k）。k<=0 → None。"""
    if k <= 0:
        return None
    return _hit_count(retrieved, relevant, k) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """第一个相关命中的 1/rank；无命中 → 0.0。"""
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """二元增益 nDCG@k = DCG / IDCG。relevant 为空 → None。"""
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved[:k])
        if rid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval_metrics.py -v`
Expected: PASS（8 passed）

- [ ] **Step 6: Commit**

```bash
git add eval/retrieval/__init__.py eval/retrieval/metrics.py eval/retrieval/dataset/.gitkeep eval/retrieval/results/.gitkeep tests/test_retrieval_metrics.py
git commit -m "feat(eval): 检索层指标纯函数 + retrieval 子包骨架"
```

---

## Task 2: label.py 标注脚本（pooling + LLM 判定）

**Files:**
- Create: `eval/retrieval/label.py`
- Test: `tests/test_retrieval_label_helpers.py`

**Interfaces:**
- Consumes:
  - `core.retrieval.make_retriever(name) -> Retriever`；`VectorRetriever.retrieve(query, *, index_manager, book_titles, top_k)`；`HybridRetriever._ensure_bm25(index_manager)` + `._bm25_search(query, book_titles, top_k)`（pooling 取 bm25-only 候选，复用其内部 BM25 路径）。
  - `core.rag.data_loader.RAGIndexManager(persist_dir, collection_name)`，属性 `chroma_collection`、方法 `get_index()`。
  - `eval.config.CHROMA_DIR`、`CHROMA_COLLECTION`、`DATASET_DIR`。
  - `configs.llm.configure_llm/deepseek_api_key`、`configs.embedding.configure_embedding`。
- Produces:
  - `merge_pool(ranked_lists: list[list]) -> tuple[list[str], dict[str, object]]`：各路有序 NodeWithScore → (保序去重 id 列表, id→node)。
  - `parse_judgement(text: str, idx_to_id: dict[int, str]) -> set[str]`：LLM JSON `{局部序号: 0|1}` → 判 1 且在范围内的 chunk_id 集合，容错（去 ```json 围栏 / 非法键忽略）。
  - `LABEL_OUT`（`eval/retrieval/dataset/golden.retrieval.jsonl` 绝对路径）。
  - `main()` async 入口，CLI `python -m eval.retrieval.label`。
  - 输出每行 schema：`{"user_input": str, "category": str, "relevant_chunk_ids": list[str], "skipped": bool}`。

- [ ] **Step 1: Write the failing test (pure helpers)**

`tests/test_retrieval_label_helpers.py`:
```python
from types import SimpleNamespace

from eval.retrieval.label import merge_pool, parse_judgement


def _nws(node_id: str):
    return SimpleNamespace(node=SimpleNamespace(node_id=node_id))


def test_merge_pool_dedup_preserves_order():
    dense = [_nws("a"), _nws("b")]
    sparse = [_nws("b"), _nws("c")]
    ids, id2node = merge_pool([dense, sparse])
    assert ids == ["a", "b", "c"]              # 保序、去重
    assert set(id2node) == {"a", "b", "c"}
    assert id2node["a"].node_id == "a"


def test_parse_judgement_maps_local_index_to_id():
    idx_to_id = {0: "a", 1: "b", 2: "c"}
    assert parse_judgement('{"0": 1, "1": 0, "2": 1}', idx_to_id) == {"a", "c"}


def test_parse_judgement_tolerates_fence_and_bad_keys():
    idx_to_id = {0: "a", 1: "b"}
    text = '```json\n{"0": 1, "9": 1, "x": 1, "1": "1"}\n```'
    # 9/x 超范围或非法忽略；"1":"1" 视为相关
    assert parse_judgement(text, idx_to_id) == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval_label_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'merge_pool'`

- [ ] **Step 3: Write label.py**

`eval/retrieval/label.py`:
```python
"""阶段一：给 golden 可答类 query 标相关 chunk（pooling + LLM 二元判定），冻结到
eval/retrieval/dataset/golden.retrieval.jsonl。调 LLM，一次性；人工抽检后即用。

只标 retrievable/pending_split/other/ambiguous；missing_info/out_of_scope 跳过
（本无相关 chunk）。pooling = vector top-N ∪ bm25 top-N，judge 分批 0/1 判定。

运行（项目根）：python -m eval.retrieval.label
"""
import asyncio
import json
import os

from openai import AsyncOpenAI

from configs.embedding import configure_embedding
from configs.llm import configure_llm, deepseek_api_key
from core.rag.data_loader import RAGIndexManager
from core.retrieval.retrieve import make_retriever
from eval.config import CHROMA_COLLECTION, CHROMA_DIR, DATASET_DIR

POOL_N = 30          # 每路候选深度
JUDGE_BATCH = 10     # 每批判定 chunk 数
CHUNK_TRUNC = 600    # judge 时 chunk 正文截断字数

GOLDEN = os.path.join(DATASET_DIR, "golden.jsonl")
LABEL_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "golden.retrieval.jsonl"
)
LABEL_CATEGORIES = {"retrievable", "pending_split", "other", "ambiguous"}

_JUDGE_PROMPT = """判断每个检索片段是否与问题【直接相关】（能用于回答问题）。

问题：{question}

片段（按序号）：
{chunks}

只输出 JSON：键为片段序号(字符串)，值为 1(相关)或 0(不相关)，覆盖全部序号。
不要任何解释。例：{{"0": 1, "1": 0}}"""


def _chunk_text(node) -> str:
    return (node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")) or ""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def merge_pool(ranked_lists: list[list]) -> tuple[list[str], dict[str, object]]:
    """各路有序 NodeWithScore 列表 → (保序去重的 node_id 列表, node_id→node)。"""
    ordered_ids: list[str] = []
    id2node: dict[str, object] = {}
    for nodes in ranked_lists:
        for nws in nodes:
            nid = nws.node.node_id
            if nid not in id2node:
                id2node[nid] = nws.node
                ordered_ids.append(nid)
    return ordered_ids, id2node


def parse_judgement(text: str, idx_to_id: dict[int, str]) -> set[str]:
    """LLM JSON {局部序号: 0|1} → 判 1 且序号在范围内的 chunk_id 集合。容错。"""
    data = json.loads(_strip_fences(text))
    out: set[str] = set()
    for key, val in data.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if idx in idx_to_id and int(val) == 1:
            out.add(idx_to_id[idx])
    return out


async def _judge_batch(gen, question, batch_ids, id2node) -> set[str]:
    """对一批 chunk 调 LLM 判 0/1。batch_ids：本批 chunk_id 列表。"""
    idx_to_id = {i: cid for i, cid in enumerate(batch_ids)}
    chunks = "\n".join(
        f"[{i}] {_chunk_text(id2node[cid])[:CHUNK_TRUNC]}" for i, cid in idx_to_id.items()
    )
    prompt = _JUDGE_PROMPT.format(question=question, chunks=chunks)
    resp = await gen.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    return parse_judgement(resp.choices[0].message.content, idx_to_id)


async def _build_pool(query, idx, hybrid) -> tuple[list[str], dict[str, object]]:
    """vector top-N ∪ bm25 top-N 候选池（bm25 复用 HybridRetriever 内部路径）。"""
    dense = await make_retriever("vector").retrieve(
        query, index_manager=idx, book_titles=None, top_k=POOL_N
    )
    await hybrid._ensure_bm25(idx)
    sparse = hybrid._bm25_search(query, None, POOL_N)
    return merge_pool([dense, sparse])


async def _label_one(gen, query, idx, hybrid) -> list[str]:
    ids, id2node = await _build_pool(query, idx, hybrid)
    relevant: set[str] = set()
    for start in range(0, len(ids), JUDGE_BATCH):
        batch = ids[start:start + JUDGE_BATCH]
        relevant |= await _judge_batch(gen, query, batch, id2node)
    return [cid for cid in ids if cid in relevant]   # 保 pooling 序


async def main() -> None:
    with open(GOLDEN, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    hybrid = make_retriever("hybrid")
    gen = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)

    out_rows: list[dict] = []
    zero_hit: list[str] = []
    for r in rows:
        q = r["user_input"]
        cat = r.get("category", "")
        if cat not in LABEL_CATEGORIES:
            out_rows.append({"user_input": q, "category": cat,
                             "relevant_chunk_ids": [], "skipped": True})
            continue
        try:
            rel = await _label_one(gen, q, idx, hybrid)
        except Exception as exc:  # noqa: BLE001 — 单条失败不中断，标 skipped
            print(f"[warn] 判定失败，跳过：{q[:30]} | {type(exc).__name__}: {exc}")
            out_rows.append({"user_input": q, "category": cat,
                             "relevant_chunk_ids": [], "skipped": True})
            continue
        skipped = not rel
        if skipped:
            zero_hit.append(q)
        out_rows.append({"user_input": q, "category": cat,
                         "relevant_chunk_ids": rel, "skipped": skipped})
        print(f"[{cat}] {q[:40]} → {len(rel)} 相关")

    os.makedirs(os.path.dirname(LABEL_OUT), exist_ok=True)
    with open(LABEL_OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labeled = sum(1 for r in out_rows if not r["skipped"])
    print(f"\n已写 {LABEL_OUT}：{labeled}/{len(out_rows)} 条有相关标注")
    if zero_hit:
        print(f"[抽检] {len(zero_hit)} 条零命中（已标 skipped），建议人工核对：")
        for q in zero_hit:
            print(f"  - {q[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval_label_helpers.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add eval/retrieval/label.py tests/test_retrieval_label_helpers.py
git commit -m "feat(eval): 检索层标注脚本(pooling+LLM二元判定)"
```

- [ ] **Step 6: 集成 smoke（手动，需 .env + 已入库 chroma_db）**

Run: `python -m eval.retrieval.label`
Expected: 逐条打印 `[category] query → N 相关`，末尾写出 `eval/retrieval/dataset/golden.retrieval.jsonl` 并列出零命中条目。**人工抽检几条相关 chunk 是否合理**后再用于评测。
（无 `.env` / 空库时此步跳过，不阻塞 Task 3 的纯函数部分。）

---

## Task 3: run.py 评测运行器

**Files:**
- Create: `eval/retrieval/run.py`
- Test: `tests/test_retrieval_aggregate.py`

**Interfaces:**
- Consumes:
  - `eval.retrieval.metrics`：`K_VALUES`、`recall_at_k`、`precision_at_k`、`mrr`、`ndcg_at_k`。
  - `core.retrieval.make_retriever`；`RAGIndexManager`；`eval.config`；`configs.embedding.configure_embedding`。
  - `eval.retrieval.label.LABEL_OUT`（读标注文件路径）。
- Produces:
  - `per_query_metrics(retrieved: list[str], relevant: set[str], k_values=K_VALUES) -> dict[str, float | None]`：键如 `recall@5`、`precision@3`、`ndcg@10`、`mrr`。
  - `aggregate(rows: list[dict]) -> dict[str, float | None]`：逐键求均值，忽略 None；空入参 → `{}`。
  - `RESULT_CSV`（`eval/retrieval/results/retrieval_eval.csv` 绝对路径）。
  - `main()` async 入口，CLI `python -m eval.retrieval.run --retrievers vector hybrid`。

- [ ] **Step 1: Write the failing test (pure helpers)**

`tests/test_retrieval_aggregate.py`:
```python
import pytest

from eval.retrieval.run import aggregate, per_query_metrics


def test_per_query_metrics_keys_and_values():
    row = per_query_metrics(["a", "b", "c"], {"a", "c"}, k_values=(1, 3))
    assert row["recall@1"] == pytest.approx(1 / 2)   # top1 命中 a
    assert row["recall@3"] == pytest.approx(1.0)
    assert row["precision@1"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(1.0)
    assert "ndcg@3" in row


def test_aggregate_ignores_none():
    rows = [
        {"recall@1": 1.0, "mrr": 0.5},
        {"recall@1": None, "mrr": 1.0},   # recall@1=None 不计入
    ]
    agg = aggregate(rows)
    assert agg["recall@1"] == pytest.approx(1.0)     # 只 (1.0)/1
    assert agg["mrr"] == pytest.approx(0.75)


def test_aggregate_empty_is_empty_dict():
    assert aggregate([]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'aggregate'`

- [ ] **Step 3: Write run.py**

`eval/retrieval/run.py`:
```python
"""阶段二：读冻结标注，逐 retriever 绕过 front_door 检索、算指标、出对比表。
零 LLM（dense 路调 embedding），可反复跑——改 retrieve.py 后只重跑此脚本。

运行（项目根）：python -m eval.retrieval.run --retrievers vector hybrid
"""
import argparse
import asyncio
import csv
import json
import os

from configs.embedding import configure_embedding
from core.rag.data_loader import RAGIndexManager
from core.retrieval.retrieve import make_retriever
from eval.config import CHROMA_COLLECTION, CHROMA_DIR
from eval.retrieval.label import LABEL_OUT
from eval.retrieval.metrics import (
    K_VALUES,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RESULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "retrieval_eval.csv"
)


def per_query_metrics(
    retrieved: list[str], relevant: set[str], k_values: tuple[int, ...] = K_VALUES
) -> dict[str, float | None]:
    """一条 query 的全部指标（recall/precision/ndcg@k + mrr）。"""
    row: dict[str, float | None] = {"mrr": mrr(retrieved, relevant)}
    for k in k_values:
        row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        row[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return row


def aggregate(rows: list[dict]) -> dict[str, float | None]:
    """逐键求均值，忽略 None。空入参 → {}。"""
    if not rows:
        return {}
    out: dict[str, float | None] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def _metric_cols(k_values: tuple[int, ...] = K_VALUES) -> list[str]:
    cols = ["mrr"]
    for k in k_values:
        cols += [f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
    return cols


def _load_labels() -> list[dict]:
    if not os.path.exists(LABEL_OUT):
        raise SystemExit(f"标注文件不存在：{LABEL_OUT}\n先跑：python -m eval.retrieval.label")
    with open(LABEL_OUT, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    labeled = [r for r in rows if not r.get("skipped")]
    if not labeled:
        raise SystemExit("标注全为 skipped，无可评测样本；先核对 label 产物。")
    return labeled


async def _eval_retriever(name: str, labels: list[dict], idx) -> tuple[dict, list[dict]]:
    """跑一个 retriever：返回 (聚合均值, 每条明细)。"""
    retriever = make_retriever(name)
    top_k = max(K_VALUES)
    per_rows: list[dict] = []
    detail: list[dict] = []
    for r in labels:
        q = r["user_input"]
        relevant = set(r["relevant_chunk_ids"])
        try:
            nodes = await retriever.retrieve(
                q, index_manager=idx, book_titles=None, top_k=top_k
            )
            retrieved = [n.node.node_id for n in nodes]
        except Exception as exc:  # noqa: BLE001 — 单条异常不计入，不中断
            print(f"[warn] {name} 检索失败：{q[:30]} | {type(exc).__name__}: {exc}")
            continue
        m = per_query_metrics(retrieved, relevant)
        per_rows.append(m)
        detail.append({"variant": name, "user_input": q,
                       "category": r.get("category", ""), **m})
    return aggregate(per_rows), detail


def _render_table(results: list[tuple[str, dict]]) -> str:
    """results: [(name, 聚合均值)]。→ Markdown 对比表。"""
    cols = _metric_cols()
    header = "| retriever | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name, agg in results:
        cells = [f"{agg.get(c):.3f}" if agg.get(c) is not None else "—" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_csv(all_detail: list[dict], path: str) -> None:
    cols = ["variant", "user_input", "category"] + _metric_cols()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in all_detail:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in d.items()})


async def main(retrievers: list[str]) -> None:
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    labels = _load_labels()
    print(f"评测 {len(labels)} 条有标注样本 | retrievers={retrievers}\n")

    results: list[tuple[str, dict]] = []
    all_detail: list[dict] = []
    for name in retrievers:
        agg, detail = await _eval_retriever(name, labels, idx)
        results.append((name, agg))
        all_detail += detail

    print(_render_table(results))
    _write_csv(all_detail, RESULT_CSV)
    print(f"\n明细已写 {RESULT_CSV}")


def _parse_args():
    p = argparse.ArgumentParser(description="检索层评测：vector vs hybrid 的 Recall@k/nDCG")
    p.add_argument("--retrievers", nargs="+", default=["vector", "hybrid"],
                   help="被测 retriever 名（make_retriever 注册名），默认 vector hybrid")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.retrievers))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval_aggregate.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Run full retrieval test suite**

Run: `python -m pytest tests/test_retrieval_metrics.py tests/test_retrieval_label_helpers.py tests/test_retrieval_aggregate.py -v`
Expected: all PASS（14 passed）

- [ ] **Step 6: Commit**

```bash
git add eval/retrieval/run.py tests/test_retrieval_aggregate.py
git commit -m "feat(eval): 检索层评测运行器(vector vs hybrid 指标对比表+CSV)"
```

- [ ] **Step 7: 集成 smoke（手动，需 Task 2 已产标注 + chroma_db）**

Run: `python -m eval.retrieval.run --retrievers vector hybrid`
Expected: console 打印 Markdown 对比表（两行 vector/hybrid，各 k 的 recall/precision/ndcg + mrr），写出 `eval/retrieval/results/retrieval_eval.csv`。确认 hybrid 各指标相对 vector 的高低符合预期。

---

## Self-Review

- **Spec coverage**：
  - 三个新文件 `label.py`/`metrics.py`/`run.py` + 单测 → Task 1/2/3 全覆盖；目录隔离（`eval/retrieval/dataset`、`results`）→ Task 1 建占位。
  - 标注来源 pooling+LLM 二元、绕过 front_door、只标可答类、skipped 不计入、按 category 明细、CSV 输出、常量 POOL_N/JUDGE_BATCH/CHUNK_TRUNC/K_VALUES → 均落到任务步骤。
  - 错误处理（单条失败标 skipped / 记 warning 不中断、空标注报错）→ label.main / run._load_labels / _eval_retriever。
- **Placeholder scan**：无 TODO/TBD；每个代码步骤含完整代码与可运行命令。
- **Type consistency**：`merge_pool`/`parse_judgement`/`per_query_metrics`/`aggregate` 签名在 Interfaces 与实现/测试间一致；指标键名 `recall@k`/`precision@k`/`ndcg@k`/`mrr` 在 metrics→run→csv→test 间统一；`LABEL_OUT` 由 label 定义、run 导入复用。
- **Note**：pooling 取 bm25-only 候选复用了 `HybridRetriever._ensure_bm25/_bm25_search`（私有方法），属评测工具内的有意复用，已在 Task 2 Interfaces 标注；若后续 core 注册独立 `bm25` retriever 可替换。
