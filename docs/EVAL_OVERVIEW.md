# 评测体系总览（eval/）

> 一句话：用 **ragas 指标**，对"决策路由 RAG 系统"做 **workflow vs agent 两路线** 对比，量化每条路线带来多少提升。

本文档梳理整套评测：被测什么、用什么指标、数据从哪来、脚本怎么分工、怎么跑、当前已知问题。

---

## 1. 全景图

```
┌─ 数据集 (eval/dataset/) ──────────────────────────────┐
│  testset.draft.jsonl   ragas 自动生成的草稿（窄事实题）   │
│  golden.jsonl          金标准集（五类齐，带 category 标注） │
└───────────────────────────┬───────────────────────────┘
                            │ 喂入
┌─ Runner ──────────────────▼───────────────────────────┐
│  compare.py   workflow vs agent 两路线对比 delta 表     │ ← 唯一入口
└───────────────────────────┬───────────────────────────┘
              ┌─────────────┴─────────────┐
        被测系统 SUT                   评测打分
   (sut.py 包装 workflow + agent)    (metrics.py: 5 ragas 指标
   逐条 query → 答案                    + compare.aggregate: 均值聚合)
              │                            │
   被测 LLM = DeepSeek            评测 judge LLM = DeepSeek
   (configs/llm.py)               (eval/config.py，刻意解耦)
```

**两套 LLM 解耦**：被测系统用 `configs/llm.py` 的 DeepSeek；评测侧 judge 用 `eval/config.py` 的 `deepseek-v4-flash`（关 thinking）+ `bge-small-zh` embedding。两者独立配置，互不污染。
> 注：`eval/config.py` 与 `CLAUDE.md` 注释里写"沿用智谱 GLM"是历史说法，**当前代码实际用 DeepSeek**。

---

## 2. 被测系统（SUT）：决策路由 RAG

评测对象是 `core/workflow/DocQueryWorkflow`，由 `eval/harness/sut.py` 的 `DocQueryWorkflowSystem` 包装成统一 `answer(query)` 接口。它的核心是**两层决策**：

### Layer 1 · 门口（IntentRouter）
`intent_router.py`：净化 query（消指代 + 纠错 + 规范化）→ 分意图 `qa / study_plan / chitchat`。

### Layer 2 · QA 能力（QaCapability）
`qa_capability.py` + `query_preprocess.py`：对 `qa` 意图的 query，先**探测检索（probe）**，再 judge 判 **category**，按 category 走不同分支：

| category | 含义 | 分支 | 决策 flag |
|----------|------|------|-----------|
| `retrievable` | 单一概念、单轮能命中 | `retrieve()` 单轮检索合成 | （基础） |
| `pending_split` | 多实体/大主题、需罗列子项 | `split()` 拆解→逐项检索→汇总 | `split_enabled` |
| `ambiguous` | 话题具体但缺评判维度 | `assume()` 归纳维度→分节作答 | `assume_enabled` |
| `missing_info` | 信息不足/指代不明（库里有该主题但缺限定） | 反问澄清 clarify | （基础） |
| `out_of_scope` | 库外：问题清晰但库里没有该主题 | 固定话术告知，不检索/不反问 | （基础） |
| `other` | 跨主题综合/多步推理 | other agent（高难度） | `other_agent_enabled` |

**probe（探测检索）** 是横切开关 `probe_then_classify`：开启时 judge 拿"探测召回信号"判 category（更准但偏向 retrievable）；关闭时纯文本判。

### Layer 0 替代 · agent 自主规划路线（对照系）
除上面的 `DocQueryWorkflow`，评测另有第二被测系统 `eval/harness/sut.py` 的
`AgentSystem`：**绕过 IntentRouter + category 分类**，每条 query 直接喂给有界
`AutoAgent`（FunctionAgent + book_search/list_books，自主多轮规划检索）。它**不产
category**，故分类准确率列显示「—」，只在 5 个 ragas 答案质量指标上与 workflow 同台对比。
用途：回答「显式决策路由 vs 让 agent 自己规划」到底谁强。

---

## 3. 评测指标

### 3.1 五个 ragas 指标（`metrics.py`）—— claim 级，非 chunk 比对

| 指标 | 吃什么参数 | 测什么 |
|------|-----------|--------|
| `faithfulness` | user_input, response, **retrieved_contexts** | 答案有没有瞎编（每条声明是否被召回支撑） |
| `answer_relevancy` | user_input, response | 答案切不切题 |
| `context_precision` | user_input, **reference**, retrieved_contexts | 召回的 chunk 排序好不好 |
| `context_recall` | user_input, **reference**, retrieved_contexts | 召回是否盖全参考答案的要点 |
| `factual_correctness` | response, **reference** | 答案 vs 参考答案的事实 F1 |

**关键**：没有一个指标读测试集的 `reference_contexts`。它们要么用 SUT **实际召回**的 `retrieved_contexts`，要么用**参考答案** `reference`。全是 LLM 把文本拆成声明逐条判，**与召回 chunk 条数无关**。
> 推论：标金标准**只需 `user_input` + `category`**；`reference` 仅 `context_recall`/`factual_correctness`/`context_precision` 用，可选；`reference_contexts` 不用填。

### 3.2 分类准确率 —— 已移除

分类准确率指标已随门口路由重构移除（agent 路线不产 category，与 golden 标签不对齐）。现以 ragas 质量 + 成本对比两路线。

### 3.3 成本：时延 + token（`eval/harness/meter.py`）

表里还有两列成本：**时延(s/条)** 和 **tokens/条**（明细 CSV 另含 `latency_s`/`prompt_tokens`/`completion_tokens`/`total_tokens`）。

- token 由 LlamaIndex `TokenCountingHandler` 挂在 **SUT llm 实例**上客户端计数——只数被测系统、排除评测 judge；流式/非流式都从响应文本算，绕开 DeepSeek 流式不返回 usage 的缺口。只数 **LLM token，不含 embedding**；客户端近似（cl100k 代 DeepSeek 分词），仅供**跨系统相对比较**，不当账单。
- ⚠️ **读法**：这两列**越低越好**，故 delta 为正＝更贵、为负＝更省，**与上面质量列（越高越好）符号相反**。看 agent 自主规划路线时尤其明显：质量未必更高，token/时延通常高出单轮 workflow 一截。

---

## 4. 数据集（eval/dataset/）

| 文件 | 是什么 | 怎么来 |
|------|--------|--------|
| `testset.draft.jsonl` | ragas 自动生成草稿，50 条（30 单跳 + 20 多跳） | `generate_testset.py` |
| `testset.draft.csv` | 上者的 CSV，便于人工看 | `jsonl_to_csv.py` |
| `golden.jsonl` | **金标准集**，23 条，五类齐 | `merge_golden.py` 合并候选 |
| `golden.seed.jsonl` | 早期手写种子 | 手工 |
| `split_candidates.jsonl` | split/other/retrievable 候选 + 离散度诊断 | `build_split_candidates.py` |
| `ambiguous_missing_candidates.jsonl` | ambiguous/missing_info 候选 + probe 诊断 | `build_ambiguous_missing.py` |
| `ANNOTATION_CHECKLIST.md` | 金标准标注操作手册（判定树+配额+校验） | 手工 |
| `README.md` | 字段定义 + category 准则 | 手工 |

### golden.jsonl 当前组成（23 条）
| 类别 | 条数 | 来源 |
|------|------|------|
| retrievable | 4 | 窄题 |
| pending_split | 3 | 跨章多实体·列举式 |
| other | 3 | 跨章多实体·综合式 |
| ambiguous | 5 | 在库概念+缺维度问法 |
| missing_info | 4 | 悬空指代 |
| out_of_scope | 4 | 库外（PostgreSQL/MongoDB/Oracle/Cassandra） |

---

## 5. 数据集生成方法学（本次 session 总结）

ragas 自动集**只产窄事实题**（单 chunk 锚定），五类里只覆盖 retrievable。其余四类靠下面手法补：

- **pending_split / other**：跨章多实体题（概念取自不同专章）+ **措辞模板**定向——
  「…**分别起什么作用**」→ pending_split（罗列）；「…**如何协作配合**」→ other（综合）。
- **retrievable**：单一概念窄题。
- **ambiguous**：在库具体概念 + 缺评判维度问法（"X好吗""X和Y哪个好"）。
- **missing_info**：悬空指代（"上面那个怎么配"，单轮无上文）+ 库外内容。

**核心规律（实测）**：
1. `pending_split` 真正触发 = **召回离散** + **措辞列举意图**，不是"题面有几个实体"。
2. `split` vs `other` 边界 = **列举 vs 综合**（措辞一词翻类）。
3. 标签按**构造意图**定最可靠；probe 召回离散度当机器标签太噪（k=10 两轮乱跳、窄题也会散）。

---

## 6. 脚本地图（eval/ 按功能分包）

```
eval/
├── config.py          评测 judge LLM / embedding / 路径（顶层，大家都 import）
├── harness/           评测引擎
│   ├── sut.py           被测系统适配器：DocQueryWorkflowSystem（带决策 flag）、AgentSystem、RagOutput
│   ├── metrics.py       5 个 ragas 指标的字段映射 + 装配
│   ├── report.py        展示+落盘共用：render_delta_table（5 ragas+成本列）/ write_detail_csv / default_result_paths
│   └── compare.py       【唯一入口】workflow vs agent 两路线 delta 表，--out 落盘 / --detail 明细 CSV（共用 report.py）
├── datagen/           测试集 + 金标准生成
│   ├── generate_testset.py        ragas TestsetGenerator 自动生成草稿（A+B 中文约束）
│   ├── build_split_candidates.py  造 split/other/retrievable（离散度筛子 + 措辞模板）
│   ├── build_ambiguous_missing.py 造 ambiguous/missing_info
│   ├── merge_golden.py            合并候选 → golden.jsonl
│   └── fill_reference.py          慷慨检索补 reference
├── poc/               章节摘要法探索性 PoC（样板，非主流程）
│   └── poc_chapter_summary.py / poc_classify_check.py / poc_chapter_loop.py
├── utils/
│   └── jsonl_to_csv.py            testset jsonl → csv 便于人工看
├── dataset/           测试集与金标准数据
└── results/           compare 落盘的跑分表（md）+ 明细（detail.csv）
```

---

## 7. 怎么跑

```powershell
# 冒烟（前 2 条，确认链路通）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 2

# 单路线（只跑 workflow）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --variants workflow

# 全量对比表，落盘到 docs/
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --out docs/compare.md --detail docs/compare_detail.csv

# 其它入口：生成测试集 / 造金标准 / 补 reference
python -m eval.datagen.generate_testset --size 50
python -m eval.datagen.build_split_candidates
python -m eval.datagen.merge_golden
python -m eval.datagen.fill_reference
```

**路线**（`compare.VARIANTS`）：`workflow`（DocQueryWorkflow 默认 flags） / `agent`（AutoAgent 自主规划）。

**注意**：
- 先激活 `.venv`，`.env` 要有 `DEEPSEEK_API_KEY`。
- 花 LLM 调用：每条 × 每变体 = 1 次作答 + 5 指标。冒烟用 `--limit`。
- 金标准没填 `reference` → `context_recall`/`factual_correctness` 列为 `—`。
- `compare` 显示 5 ragas + 2 成本列；`--detail` 可导出每条明细 CSV。

---

## 8. 当前已知问题（评测发现的真实弱点）

1. **库外问题误判 retrievable —— 已修（2026-06-17）**：曾因检索器永远返回 top-k（即便跑题）、judge 缺召回相关性门控，PostgreSQL/MongoDB 等库外题被判 retrievable。已新增独立 `out_of_scope` 分类（判据锚定召回片段与问题主体实体的相关性），与 `missing_info`（信息不足→反问）解耦。详见 `docs/superpowers/specs/2026-06-17-out-of-scope-classification-design.md`。
2. **probe 偏向 retrievable**：probe 提供召回证据会把 judge 推向 retrievable，利于 faithfulness 接地，但对边界 ambiguous 题有害（实测把"自适应哈希索引好用吗"从 ambiguous 翻成 retrievable）。
3. **当前 golden 集对 probe 不公平**：全是 MySQL（judge 本就认识）+ 概念重叠的 OOB，**缺 probe 该救的"在库不认识专名"样本**（如 OpenClaw 书的 SOUL.md/Cron）。导致首张全量表里 +probe 分类准确率不升反降（0.70→0.61），**不能据此判 probe 没用**。

> 已修：旧 `BookRagWorkflow` 及其评测装配（`BookRagWorkflowSystem`）已退役删除。`run_eval` 已并入 `compare`，现仅 `compare` 唯一入口，测 workflow vs agent 两路线。

---

## 9. 待办（让评测更公平/完整）

- [ ] 补 OpenClaw 书"不认识专名"题（probe 的主场），重测 probe 是否翻盘。
- [ ] OOB 题改用零概念重叠话题（如"怎么用 React 写组件"），让 missing_info 标签干净。
- [ ] golden 扩到 30~50 条，补措辞变体，做 OpenClaw 平行样本。
- [ ] 报告改看**分类别准确率**，而非被 missing_info 拖累的总分。
- [ ] 给 golden 的非 missing_info 条目补 `reference`，激活 context_recall/factual_correctness。
- [x] 修库外 bug：已新增 `out_of_scope` 分类（召回相关性按主体实体判定）。

---

*本文档随评测演进更新。最近更新：2026-06-13。*
