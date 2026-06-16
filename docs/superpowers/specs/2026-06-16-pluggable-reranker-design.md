# 可插拔 Reranker：注入式检索组件（第一版）

**日期**：2026-06-16
**状态**：设计已确认，待写实现 plan

## 背景与动机

eval 全量表发现 `context_precision ≈ 0.37`——召回排序偏弱，相关片段没排在前面。
reranker（对召回候选重新打分排序）正是对症的杠杆。

现有 ablation 靠布尔 flag（`probe_then_classify` / `split_enabled` / `assume_enabled`
/ `other_agent_enabled`），适合二元决策。但 reranker 这类「带多种实现 + 配置」的东西
再堆 flag 会组合爆炸。改用**装配时注入的策略组件**：组件按类别划分，工作流写好后在
拼装层选实现——**不传 = 没这步（基线）**，传 A 用 A，传 B 用 B。

本设计是该方向的第一个增量：**只做 Reranker 一个可注入类别**，把注入骨架跑通、用 eval
量出 reranker 的 delta。检索策略 / dedup / filter 等其余类别留待本套验证过后按同样模式加。

## 范围

- ✅ 引入 `Reranker` 协议 + 一个真实现（本地 bge 交叉编码器）+ `make_reranker` 工厂。
- ✅ 在 `QaCapability._retrieve_nodes` 唯一咽喉点接入「过召回 → 重排 → 截断」。
- ✅ `DocQueryWorkflow` / eval `VARIANTS` 透传 reranker 选择，compare 量化增益。
- ❌ 不做检索策略（向量/混合/HyDE）、dedup/filter、top_k 后处理等其余 stage——YAGNI，
   等数据支撑再按同样注入模式加。
- ❌ 第一版只落 bge 一个实现；第二个实现（如 LLM reranker）骨架建好后顺手加，不在本版。

## 设计

### 1. 接缝：唯一咽喉点 `_retrieve_nodes` + 过召回语义

rerank 放进 `QaCapability._retrieve_nodes`——它是当前唯一的检索调用点，
classify 的 probe / retrieve / split 子查询 / assume 维度全走它，于是**统一受益**，
不必在四处分别插。

reranker 只有在**过召回**时才有意义（候选越多越能挑出真正相关的）：

| 配置 | 召回行为 |
|------|---------|
| `reranker is None`（基线） | 直接召回 `similarity_top_k`（=5），与现状完全一致 |
| `reranker` 存在 | 先召回更大候选池 `rerank_candidate_k`（默认 20）→ reranker 打分 → 截回 `similarity_top_k`（=5） |

probe 也统一走重排（probe 召回质量同样受益；保持单咽喉点的一致性）。

### 2. 组件定义（core，新模块 `core/retrieval/rerank.py`）

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Reranker(Protocol):
    """对召回候选重新打分排序，返回前 top_n 个。"""
    async def rerank(self, query: str, nodes: list, top_n: int) -> list: ...


class BgeReranker:
    """本地交叉编码器，包 LlamaIndex SentenceTransformerRerank（bge-reranker-v2-m3）。

    模型同步推理，用 asyncio.to_thread 卸到线程，不堵事件循环。
    """
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        ...
    async def rerank(self, query, nodes, top_n) -> list:
        ...


def make_reranker(name: str | None) -> "Reranker | None":
    """名字 → 实例。None/"" → None（跳过这步）。未知名字 → 抛错（配置错误尽早暴露）。"""
    ...
```

- **纯 DI**：`QaCapability` 只认 `Reranker` 协议对象，单测可塞假 reranker。
- 字符串→对象的解析（`make_reranker`）住在 **core**，eval 只传名字字符串，
  评测概念不漏进 core。
- 分层守卫不破坏：`core/retrieval/` 只依赖 LlamaIndex + configs，不碰 api / eval。

### 3. 装配与注入路径

```
eval VARIANTS: {..., "reranker": "bge-reranker-v2-m3"}
        │  （flags dict 原样 **kwargs 透传）
        ▼
DocQueryWorkflow.__init__: 新增 reranker: str | None = None
        │  make_reranker(name) → Reranker 对象 | None
        ▼
QaCapability.__init__: 新增 reranker: Reranker | None = None,
                              rerank_candidate_k: int = 20
        │
        ▼
_retrieve_nodes(): None → 直召 5 / 有 → 召 20 重排截 5
```

- `DocQueryWorkflow` 与 `QaCapability` 的 `reranker` 默认 `None` → **现有全部行为与
  测试零变化**，基线天然 = 不传。
- `make_reranker` 在 `DocQueryWorkflow.__init__` 调用（core 内解析）。
- eval `sut.py` 已 `**self._flags` 透传，compare 加一条带 `reranker=...` 的变体即可
  自动流通，无需改 sut。
- eval `datagen/*` 直接构造 `QaCapability` 的脚本不受影响（reranker 默认 None）。

### 4. eval 对接

- `eval/harness/compare.py` 的 `VARIANTS` 加一条，如：
  ```python
  "全开+rerank": dict(probe_then_classify=True, split_enabled=True,
                      assume_enabled=True, other_agent_enabled=True,
                      reranker="bge-reranker-v2-m3"),
  ```
- 复用现成 ablation 框架与 delta 表，直接量出 `context_precision`（及其余指标）增益。

### 5. 测试

- 新增 `QaCapability` 单测：注入假 reranker，断言
  - 被调用且 query 正确传入；
  - 候选顺序按假 reranker 返回重排；
  - 结果截到 `top_n = similarity_top_k`；
  - `reranker=None` 时跳过、行为同现状（过召回也不触发）。
- `make_reranker` 单测：None/""→None，已知名字→实例，未知名字→抛错。
- 现有 `tests/test_qa_capability.py` 不受影响（默认 None）。

### 6. 依赖

- `requirements` 增加 sentence-transformers（及 bge reranker 所需包；实现期确认确切
  LlamaIndex postprocessor 包名）。
- bge-reranker-v2-m3 模型（~600MB）首次使用时下载。

## 非目标 / 后续

- 第二个 reranker 实现（LLM reranker 等）：骨架建好后按同样注入模式追加。
- 其余组件类别（Retriever 策略、dedup/filter）：同模式，数据支撑后再做。

## 相关

- 评测体系与已知缺陷：`docs/EVAL_OVERVIEW.md`、记忆 `project_eval_golden_and_oob_bug`。
- 接缝代码：`core/workflow/qa_capability.py`、装配 `core/workflow/doc_workflow.py`、
  eval 对接 `eval/harness/compare.py` + `eval/harness/sut.py`。
