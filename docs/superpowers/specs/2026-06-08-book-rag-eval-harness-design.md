# Book RAG 评测体系设计（仿 legacy/evals，基于 ragas 0.4.3）

- 日期：2026-06-08
- 状态：已批准（设计阶段）
- 作者：与用户协作（brainstorming）

## 1. 背景与目标

当前 `eval/` 是手写实现：`ablation.py`（三臂查询理解 ablation）+ `questions.json`（10 道手写 MySQL 问题、无 reference）+ 手写 `_JUDGE_PROMPT`（LLM-as-judge 打 relevance/groundedness）。本次**抛弃整个现有 `eval/`**，参考项目早期 `legacy/evals` 的 ragas 地道用法，为当前 book 项目重建一套**通用 RAG 质量评测体系**。

**核心目标**：一套可复用、可回归对比的 RAG 质量基线——自动生成 book 测试集 + ragas 标准指标 + 端到端评测真实 `BookRagWorkflow`，结果带时间戳留档。

**非目标**（本次不做）：
- 不保留三臂查询理解 ablation（raw/decompose/preprocess 对照）。
- 不修复 `core/workflow/book_rag.py`（当前半成品，仅作为风险标注）。
- 不评测 `BookAgent`（仅在 SUT 接口上预留扩展）。

## 2. 参考来源：legacy/evals 的形态

- `legacy/evals/generate_testset.py`：ragas `TestsetGenerator` 从 ChromaDB 切片自动生成测试集——Persona 驱动、单跳/多跳合成器、中文 prompt 适配，产出 `user_input / reference_contexts / reference` 三件套。
- `legacy/evals/evals.py`：`@experiment()` 装饰器 + `DiscreteMetric`（correctness pass/fail）+ `collections.ContextEntityRecall`，逐行跑 RAG、打分、`@experiment` 框架自动落盘 `results/experiments/<时间戳>_<name>.csv`。

本设计沿用该模式，并在指标上扩展到检索侧 + 生成侧四象限。

## 3. 关键技术约束（ragas 0.4.3，已核实）

- 装的就是当前最新版 **0.4.3**。
- `ragas.metrics.collections` 指标**构造时注入 `llm`**：`Faithfulness(llm)`、`ContextRecall(llm)`、`ContextPrecisionWithReference(llm)`、`FactualCorrectness(llm, mode=...)`；`AnswerRelevancy(llm, embeddings)` 额外需 embeddings。
- 评测侧 LLM 必须用 ragas 原生 `llm_factory(model, client=...)`（instructor 结构化输出），**不能直接塞 llama-index 的 LLM**。embeddings 用 ragas `HuggingFaceEmbeddings`。
- `TestsetGenerator(llm, embedding_model, persona_list, ...)` 吃同一个原生 llm。
- Runner 采用 ragas `@experiment()` 框架（legacy 同款，自带时间戳留档，契合回归对比）。

## 4. 决策记录（用户已确认）

1. 评测目的：**通用 RAG 质量评测（仿 legacy）**，非 ablation。
2. 被测系统：**先 `BookRagWorkflow`，runner/SUT 接口抽象可换**，预留 `BookAgent`。
3. 测试集：**自动生成（TestsetGenerator）+ 人工校验**。
4. 指标：全选——**Faithfulness、AnswerRelevancy、ContextPrecision、ContextRecall、FactualCorrectness**。
5. Runner：**方案 A `@experiment()`**。
6. 删除现有 `eval/` 全部文件。
7. `outcome != answered` 的行**排除出指标均值、单独报占比**。

## 5. 目录结构

```
eval/
  __init__.py
  config.py            # 评测侧 LLM(llm_factory→GLM) + embeddings(HF bge) + 路径常量
  sut.py               # SUT 协议 + BookRagWorkflow 适配器（预留 BookAgent）
  metrics.py           # 构建 5 个 ragas 指标实例（注入评测 llm/embeddings）
  generate_testset.py  # TestsetGenerator 从 book chroma 切片生成 → testset.draft.jsonl
  run_eval.py          # @experiment runner：载入测试集→跑 SUT→打分→结果 CSV + 聚合报告
  dataset/
    testset.draft.jsonl  # 生成的草稿（待人工校验）
    testset.jsonl        # 校验后的最终测试集（committed，run_eval 读它）
  results/experiments/   # @experiment 自动落盘
```

**删除**：`eval/ablation.py`、`eval/questions.json`、`eval/ablation_report.md`、`eval/ablation_results.json`（保留 `eval/__init__.py`，内容重写或置空）。

## 6. 组件职责

### 6.1 config.py — 评测侧配置（与 SUT 自身 LLM 解耦）

- `make_eval_llm()` → `llm_factory("glm-4-flash", client=AsyncOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=os.getenv("ZHIPU_API_KEY")))`。instructor 原生，collections 指标必需。
- `make_eval_embeddings()` → ragas `HuggingFaceEmbeddings(model="BAAI/bge-small-zh-v1.5")`。
- 路径常量：`EVAL_DIR`、`DATASET_DIR`、`TESTSET_PATH`（=`dataset/testset.jsonl`）、`TESTSET_DRAFT_PATH`、`RESULTS_DIR`。

### 6.2 sut.py — 可替换被测系统抽象

```python
@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str          # "answered" | "clarify" | "split" | "empty" | "error"

class RagSystem(Protocol):
    async def answer(self, query: str) -> RagOutput: ...

class BookRagWorkflowSystem:   # 包装 core.workflow.book_rag.BookRagWorkflow
    def __init__(self, index_manager, llm, similarity_top_k=5): ...
    async def answer(self, query) -> RagOutput:
        # 运行 workflow.run(query=query)，按返回类型分流：
        #   llama-index Response（有 response/source_nodes）→ outcome="answered"
        #       response=resp.response；retrieved_contexts=[n.node.get_content() for n in resp.source_nodes]
        #   Response 但 source_nodes 为空 / response 空 → outcome="empty"
        #   ClarifyResult（clarify/split 分支）→ outcome="clarify" 或 "split"，response=""、contexts=[]
        #   异常 → outcome="error"
```

落地"先 Workflow 预留 Agent"：以后加 `BookAgentSystem` 实现同一 `RagSystem` 协议，`run_eval` 不动。

### 6.3 metrics.py

`build_metrics(llm, embeddings) -> list`，返回：
- `Faithfulness(llm=llm)`
- `AnswerRelevancy(llm=llm, embeddings=embeddings)`
- `ContextPrecisionWithReference(llm=llm)`
- `ContextRecall(llm=llm)`
- `FactualCorrectness(llm=llm, mode="f1")`

（均来自 `ragas.metrics.collections`。）

### 6.4 generate_testset.py — 改编 legacy

- 从项目现有 chroma（复用 `core.rag.data_loader.RAGIndexManager` / 项目 chroma 配置）载入 book 切片，转 LangChain `Document`，metadata 带 `book_title` 等。
- book 领域 Persona（如"读技术书、提出具体技术问题的工程师/学习者"）。
- 问题分布：单跳 `SingleHopSpecificQuerySynthesizer` 60% + 多跳 `MultiHopSpecificQuerySynthesizer` 40%。
- 中文 prompt 适配（`adapt_prompts("chinese", llm=...)`）。
- 产出 `dataset/testset.draft.jsonl`（含 `user_input/reference_contexts/reference/...`）。
- **人工校验**：剔除低质题、修正 reference 后另存为 `dataset/testset.jsonl`（run_eval 读这个）。
- 测试集规模、collection 名等参数以项目实际 book 数据为准（实现时确认）。

### 6.5 run_eval.py — @experiment runner

```python
@experiment()
async def evaluate_book_rag(row, sut: RagSystem, metrics) -> dict:
    out = await sut.answer(row.user_input)
    base = {"user_input": row.user_input, "reference": row.reference,
            "response": out.response, "outcome": out.outcome}
    if out.outcome != "answered":
        return base          # 非 answered：不打分，仅记录
    # 对每个指标 .ascore(...)，按指标需要的字段传入：
    #   Faithfulness: response, retrieved_contexts
    #   AnswerRelevancy: user_input, response, retrieved_contexts
    #   ContextPrecisionWithReference: user_input, retrieved_contexts, reference
    #   ContextRecall: user_input, retrieved_contexts, reference
    #   FactualCorrectness: response, reference
    return {**base, **scores}
```

main：
- 载 `dataset/testset.jsonl` → ragas `EvaluationDataset` / Dataset。
- 构建 `eval_llm`、`eval_embeddings`、`metrics`、`sut`。
- 跑 experiment（`@experiment` 自动落盘 `results/experiments/<时间戳>_<name>.csv`）。
- 打印聚合报告：各指标均值（仅 `answered` 行，NaN 忽略）+ 各 `outcome` 占比。

## 7. 数据流

```
testset.jsonl
   └─逐行─> SUT.answer(user_input)
              └─> RagOutput(response, retrieved_contexts, outcome)
                    └─[answered]─> 5×metric.ascore() ─> 行结果
                    └─[其他]──────────────────────────> 行结果（仅记录 outcome）
   └─> 结果 CSV（@experiment 落盘）
   └─> 聚合：指标均值（answered 行）+ outcome 分布
```

## 8. 错误处理

- 逐行 try/except：单行 SUT 或打分失败 → `outcome="error"`，记录不中断全局。
- `outcome != "answered"` 的行排除出指标均值，单独报占比。
- GLM 限流：`RunConfig`（控并发 + 重试）。
- 指标 NaN 聚合时忽略。

## 9. 测试

轻量 smoke：`tests/test_eval_smoke.py`，用桩 `RagSystem`（返回固定 response/contexts/outcome）跑 1–2 行，验证 run_eval 的装配、字段映射与聚合逻辑，**不打真实 LLM**。

## 10. 风险 / 前置条件

- **`core/workflow/book_rag.py` 当前为半成品**（`assume` 空函数体、`split` 引用不存在的 `ev.clarify_reason`、脏 import `from sympy.strategies.core import switch`）：跑**真实**端到端 eval 前需先能 import/run。本设计不负责修它，仅标注为前置。smoke 测试用桩 SUT 不受影响。
- chroma collection 名 / 测试集规模 / Persona 文案以项目实际 book 数据为准，实现时确认。
- 自动生成的测试集质量依赖人工校验环节；未校验前不应作为正式基线。
