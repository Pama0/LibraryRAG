# 评测加「时延 + token 消耗」统计 · 设计

> 一句话：给评测每条 query 记录 **答案耗时** 与 **被测系统 LLM token 消耗**，汇总成表里两列（时延/条、tokens/条）+ 明细 CSV 四列，让 compare/run_eval 在质量之外也能比成本。

日期：2026-06-18 ｜ 分支：master（不切分支，直接改）

---

## 1. 背景与动机

评测现在只看质量（分类准确率 + 5 ragas）。但不同系统的**成本**差异很大：agent 自主规划路线是多轮真实 FunctionAgent，token 与时延远高于单轮 workflow。要回答「显式路由 vs agent 自主规划谁更划算」「probe/split 各决策加了多少开销」，必须把**时延**和**token 消耗**纳入对比。

底层事实（已探明）：
- SUT 的 LLM 是 `configs/llm.configure_llm()` 那个 DeepSeek 实例（同时 `Settings.llm` 指向它）；评测 judge 是 `eval/config.make_eval_llm()` 的**另一个实例**。→ 把 token 计数器挂在 **SUT llm 实例**上，天然只数被测系统、排除 judge。
- 答案合成走流式，DeepSeek 流式默认不返回 `usage`（`configs/usage_logging.py` 已踩过此坑）。→ 不走「读服务端 raw.usage」，改用 **LlamaIndex `TokenCountingHandler`**（客户端 tokenizer 计数），流式/非流式都从响应文本算，绕开 usage 缺口。

## 2. 范围

**做**：
- 新增 `eval/harness/meter.py`：把 `TokenCountingHandler` 挂到 SUT llm，按行 reset+read。
- `score_row` 加每条计时 + 可选 token 读取。
- `aggregate` 加 `cost` 汇总块。
- `report.py` 表加两列、明细 CSV 加四列。
- compare/run_eval 两处接线 + 对应单测。

**不做**（YAGNI）：
- 不统计 embedding token（只数 LLM；embedding 非主成本，`TokenCountingHandler` 另有字段但不纳入）。
- 不统计评测 judge 的 token（评测自身开销不是被测对象）。
- 不读服务端 raw.usage、不动 `configs/usage_logging.py`（那是另一套缓存命中观测，保持不变）。
- 不做按金额（$）折算、不做缓存命中率列。
- token 为客户端近似值（cl100k 代 DeepSeek 分词），只求跨系统**相对**可比，不追绝对精确。

## 3. 组件设计

### 3.1 `eval/harness/meter.py`（新）

```python
def attach_token_meter(llm) -> "RunMeter":
    """给 SUT llm 挂 TokenCountingHandler（只数这一实例 → SUT-only），返回 RunMeter。"""
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
    handler = TokenCountingHandler()
    llm.callback_manager = CallbackManager([handler])
    return RunMeter(handler)


class RunMeter:
    """按行测 token：reset 清零、read 取 {prompt,completion,total}_tokens。"""
    def __init__(self, handler=None):
        self._handler = handler

    def reset(self) -> None:
        if self._handler is not None:
            self._handler.reset_counts()

    def read(self) -> dict:
        h = self._handler
        if h is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": h.prompt_llm_token_count,
            "completion_tokens": h.completion_llm_token_count,
            "total_tokens": h.total_llm_token_count,
        }
```

`RunMeter` 与 `TokenCountingHandler` 解耦（构造可传 None，便于单测注入假 handler）。

### 3.2 `score_row` 加计时 + token（`eval/harness/run_eval.py`）

签名加可选 `meter`：`async def score_row(row, sut, metric_specs, meter=None)`。

```python
if meter is not None:
    meter.reset()
t0 = perf_counter()
out: RagOutput = await sut.answer(row["user_input"])
base = {... 既有字段 ...}
base["latency_s"] = round(perf_counter() - t0, 3)
if meter is not None:
    base.update(meter.read())   # prompt_tokens / completion_tokens / total_tokens
if out.outcome != "answered":
    return base
... 既有 metric 打分 ...
```

要点：
- `latency_s` 对**所有行**恒有（计时零成本，不依赖 meter）。
- token 读取点在 `answer()` 之后、metric 打分之前——judge 用别的 llm 实例，不挂此 handler，不会污染计数；reset 在 answer 前保证逐行独立。
- `from time import perf_counter` 顶部导入。

### 3.3 `aggregate` 加 cost 汇总（`eval/harness/run_eval.py`）

report 字典追加 `cost` 块（与 `metric_means` 平行）：

```python
latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
token_vals = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
report["cost"] = {
    "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
    "mean_total_tokens": (sum(token_vals) / len(token_vals)) if token_vals else None,
    "total_tokens": sum(token_vals) if token_vals else None,
}
```

- 时延对全行求均值（每行都跑了 answer）。
- token 仅对有 `total_tokens` 的行求均值/求和；没挂 meter（如纯单测）→ None。

### 3.4 `report.py` 加列

`_COLS` 末尾追加两列：

```python
("时延(s/条)", lambda rep: rep.get("cost", {}).get("mean_latency_s")),
("tokens/条", lambda rep: rep.get("cost", {}).get("mean_total_tokens")),
```

`_DETAIL_COLS` 末尾追加：`"latency_s", "prompt_tokens", "completion_tokens", "total_tokens"`。

沿用现有 `_fmt`（None→「—」，非 baseline 带 delta）与 `write_detail_csv`（`extrasaction="ignore"`，缺列自动空）。

**符号语义**：这两列**越低越好**，故 delta 为正＝更贵、为负＝更省，与质量列（越高越好）相反。不在代码里特殊处理，由 `EVAL_OVERVIEW.md` 注明读法。

### 3.5 接线（`compare._run_variants` / `run_eval._run`）

两处都在 `sut_llm = configure_llm()` 之后加：

```python
from eval.harness.meter import attach_token_meter
meter = attach_token_meter(sut_llm)
```

并把 `meter` 透传：`score_row(r, sut, metric_specs, meter=meter)`。

- compare 多变体共用同一 `sut_llm` 实例（循环里复用）；逐行 `reset` 保证每行/每变体计数独立。
- agent 变体的 `AgentSystem`→`QaAgent` 也用这个 `sut_llm` → FunctionAgent 多轮 token 自然累加进同一 handler，逐行 reset 后即该条 agent 的总消耗。

## 4. 数据流

```
score_row(meter):
  meter.reset() → t0 → await sut.answer() → latency_s = now-t0
                → base.update(meter.read())  # {prompt,completion,total}_tokens
  ↓
aggregate: report["cost"] = {mean_latency_s, mean_total_tokens, total_tokens}
  ↓
render_delta_table: 表尾两列「时延(s/条)」「tokens/条」（带 delta，越低越好）
write_detail_csv: 明细尾四列 latency_s / prompt_tokens / completion_tokens / total_tokens
```

## 5. 错误处理

- 未挂 meter（`meter=None`，如单测直接调 `score_row`）：token 三键缺省，`cost.mean_total_tokens=None` → 列显示「—」，不报错。
- `answer()` 抛异常时：compare/run_eval 的 SUT 适配器内部已 try/except 兜成 `outcome="error"`，`answer()` 正常返回 RagOutput → 计时与 token 读取照常（error 行也有 latency，token 为该次失败前的消耗）。
- `TokenCountingHandler` 对某些响应取不到文本：计数为 0 不抛，读到的就是已累计值。

## 6. 测试（TDD）

`tests/test_eval_meter.py`（新）：
- `RunMeter.read` 注入假 handler（带 `prompt_llm_token_count` 等属性）→ 返回三键正确；无 handler → 全零。
- `RunMeter.reset` 调到 handler 的 `reset_counts`（假 handler 记标志）。
- `attach_token_meter` 给假 llm 设了 `callback_manager`，返回的 meter 初始 `read()` 合理（真 handler 初值 0）。

`tests/test_eval_run.py`（追加）：
- `score_row` 无 meter：结果含 `latency_s`（≥0、float），不含 token 键。
- `score_row` 有假 meter：含 `prompt_tokens/completion_tokens/total_tokens`，且 reset 在 answer 前被调用。
- `aggregate`：给定带 `latency_s`/`total_tokens` 的行 → `cost.mean_latency_s`/`mean_total_tokens`/`total_tokens` 算对；无 token 行 → `mean_total_tokens is None`。

`tests/test_eval_report.py`（追加）：
- `render_delta_table` 输出含「时延(s/条)」「tokens/条」表头与对应值；`cost` 缺失时这两列为「—」。

既有 report/compare 渲染测试不回归（只新增列，行内原子串仍在）。

## 7. 影响面与取舍

- 质量评测路径零改动；`score_row` 仅多计时 + 可选 token 读，对未传 meter 的既有调用零行为变化。
- token 是客户端近似（cl100k vs DeepSeek 服务端分词略有出入），仅供**跨系统相对比较**，不当账单。
- 不碰 `configs/usage_logging.py`（缓存命中观测）——两套观测目的不同：那套看缓存命中率（日志），这套看每条评测样本的总消耗（落表）。
- TokenCountingHandler 经 `callback_manager` 对流式/非流式都从响应文本计数，规避 raw.usage 流式缺口。
