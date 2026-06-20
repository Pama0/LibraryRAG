# 设计：explain 意图轴 + 精修工作流（slice 1）

> 把"讲清楚"这条核心竞争力做成一等公民：在 QA 预处理里抽出**答案意图轴**，
> `explain` 走一条独立的"列骨架→逐节检索→教学体合成"精修工作流；非 explain 原样
> 滑入难度分类。
> 架构方向与取舍见 [docs/intent-axis-explain-note.md](../../intent-axis-explain-note.md)。

## Context

线上「MySQL基础知识」被 split 拆成"字符集查看 / engine_cost / EXPLAIN / 小册前言"——
召回噪声里抠出的零散片段。根因两层：

1. **难度六分类偷塞了答案意图**（`pending_split` 含"对比"、`other` 含"开放设计"），把
   "检索结构"和"答案意图"两条正交轴焊进一个 enum。
2. **"讲清楚"无专属路径**：窄概念走 retrievable、宽概念走 pending_split，同一诉求被结构轴
   切散；宽题的章节树拆分（`_book_chapters` 的 `len != 1` 死门槛）一旦结构骨架缺失就退化
   成纯内容主导，被噪声召回逼出垃圾子查询。

本刀抽出**答案意图轴**（v1 二元闸 explain/非explain），给 explain 一条独立精修工作流。

## Goals / Non-goals

**Goals**
- 新增 `IntentClassifier`（Call A）：检索降噪 + 意图二判，取代 `QueryPreprocessor` 的降噪步。
- `QueryPreprocessor`（Call B）瘦身为**只做难度六分类**（分类逻辑不动），收已降噪 query。
- `explain` 走独立工作流：宽 hybrid 覆盖召回 → `AnswerOutliner` 列概念骨架 → 每节点检索+去重
  → 教学体合成（开场全景 / 逐节接地 / 收束）。
- 非 explain **原样**滑入难度分类→现有分支（一行不改逻辑，仅 query 改为上游已降噪）。
- 降级绝不阻塞：意图判错优雅落非explain；骨架失败落有界 agent，再失败落单轮。

**Non-goals（明确不在本刀）**
- **lookup / compare / design 意图**：v1 不识别，全归非explain 滑入难度分类。
- **explain 内的多跳依赖专门处理**：列不出骨架时落 agent 兜底，不单独建多跳探测。
- **显式空节点剪枝**：hybrid 覆盖召回锚定后空节点罕见，略过即可。
- **"讲透彻"评测度量 + golden explain 样例**：另估。
- **章节树拆分的 `len != 1` 缺陷**：explain 不依赖它而自然绕开；非explain 的 split 仍受限，另一刀。

## 架构：三个决策单元

沿用项目既有决策单元模式（注入 LLM、对外只暴露 `run`、`json_object` + Pydantic 校验、
失败降级、`_strip_fences` 按模块自带副本）。

| 组件 | 职责 | 接口 | 来源 |
|---|---|---|---|
| `IntentClassifier`（新，Call A） | 检索降噪 + 意图二判 | `run(clean_query) -> (denoised_query, intent)` | 抽 `QueryPreprocessor` 的降噪步 + 加意图 |
| `QueryPreprocessor`（瘦身，Call B） | 只做难度六分类 | `run(denoised_query, retrieval_ctx) -> (category, reason, clarify_question)` | 去掉降噪步；不再回 `rewritten_query` |
| `AnswerOutliner`（新） | 据宽召回列概念骨架 | `run(denoised_query, passages) -> sub_queries: list[str]` | 全新 |

- `intent` 取值（v1）：`"explain" | "other"`（`other` = 非explain，沿用现有难度分类路径）。
- `IntentClassifier` 判意图**不需检索**（意图是答案形状、从问题本身判）。
- 命名：Call A 定为 `IntentClassifier`（降噪是其第一步，主新职是意图）。

### IntentClassifier 判据（prompt 要点）
- explain：用户要"理解/讲清楚/讲透一个概念或主题"（"什么是X""讲讲X""讲懂X""X的原理"）。
- other（非explain）：查具体事实、对比、设计、操作步骤等——交给难度分类按现有路径处理。
- 降级：解析失败/空 → `("other", clean_query)`（落已验证的存量路径，最安全）。

### AnswerOutliner 判据（prompt 要点）
- 据【宽召回 passages】把问题拆成**概念骨架**：每节点一个子主题，聚焦答案的一个方面。
- 骨架**尺寸自适应**：原子概念 → 1~2 节（退化成一段结构化回答）；宽主题 → 多节。**下限 1 节**，
  不强制最小节数（防原子概念被撑出注水）。
- 骨架对齐【库里实际覆盖】：只列召回里有支撑的子主题，别凭世界知识编库里没有的。
- 降级：解析失败/空列表 → 返回空，由 `qa.explain` 落 agent 兜底。

## 数据流与接线（DocQueryWorkflow）

```
route(front_door) → PreprocessEvent
        ↓
preprocess step:
  1. Call A: qa.gate(clean_query) → (denoised_query, intent)   # IntentClassifier
     ctx.store: rewritten_query = denoised_query                # 取代原 rewritten_query 来源
  2. intent == "explain" → ExplainEvent → explain_branch → qa.explain(...)
  3. else → Call B: qa.classify(denoised_query) → category
            → 现有分支事件(RetrieveAgent/Split/Assume/Clarify/Other/OutOfScope) 原样
```

**新增**：
- `ExplainEvent`（Event）+ `explain_branch`（step，消费 → `FinalizeEvent`，薄委托 `qa.explain`）。
- `qa.gate()`：包 `IntentClassifier`。`qa.explain()`：新能力方法（与 `split/assume/retrieve` 平级）。

**对存量的触碰（本刀最需小心处）**：
- 降噪从 `QueryPreprocessor` 上移到 Call A → **降噪后的检索 query（原 `rewritten_query`）改由
  Call A 产出**，在 preprocess step 第 1 步就 `ctx.store.set("rewritten_query", denoised_query)`，
  下游分支照旧从 ctx 取。`QueryPreprocessor` 只回 `category`/`reason`/`clarify_question`，
  **`PreprocessResult` 去掉 `rewritten_query` 字段**（不保留透传——来源唯一在 Call A，避免双源）。
- 难度的 probe 召回、explain 的宽召回，都在 denoised_query 上做（比现在更干净）。
- 非explain 路径：除 query 变"上游已降噪"，难度分类→分支这段逻辑不改。

## qa.explain() 内部管线

```
qa.explain(ctx, denoised_query, book_titles):
  1. 宽 hybrid 覆盖召回：可插拔 retriever，默认 hybrid(dense+BM25)、不重排、大 top-k
       → passages（"覆盖探针"：求有哪几块，不求精）
  2. 列骨架：AnswerOutliner.run(denoised_query, passages) → sub_queries
       └ 空骨架/失败 → 落 agent 兜底（见降级阶梯）
  3. 每节点检索：_retrieve_all(sub_queries) → 各节点 nodes（复用并发扇出；答案侧 retriever，求精）
       └ 某节点召回空 → 略过该节（无显式剪枝）
  4. 教学体合成（复用 _retrieve_and_concat 分节结构 + 教学 prompt + 头尾框）：
       · 开场全景（一段，串起骨架）
       · 逐节接地（每节点一节，标题=骨架项，正文只来自该节 chunk）
       · 收束（一句）
  5. return answer, _merge_pool(各节点 nodes)   # 去重合并供 source_nodes
```

要点：
- **两次取的解耦**：宽召回（hybrid/覆盖/列骨架）vs 节点检索（答案侧/求精/填答案），同构 probe vs 答案。
- **流式**：沿用 `RetrievalStartEvent/RetrievalDoneEvent/AnswerDeltaEvent`，前端零改动。
- **grounding 红线**：逐节正文只喂该节 chunk；开场/收束是串场框架，不引入 chunk 外事实。
- explain **跳过难度分类**：其广度由骨架节点数自然涌现（1 节=单轮、N 节=扇出），不预分类。

## 降级阶梯（绝不阻塞）

| 触发 | 落点 |
|---|---|
| `IntentClassifier` 解析失败/空 | 默认 `other` + 原 query → 难度分类（存量路径） |
| `AnswerOutliner` 空骨架/失败 | **`qa_agent` 多轮探索**（复用 other_branch 的有界 QaAgent，记日志） |
| `qa_agent` 再抛错 | 单轮检索 + 教学体合成整句（最终地板） |
| 某骨架节点召回空 | 略过该节 |

- agent 兜底顺手覆盖 explain 的多跳残留（列不出骨架的多跳题掉进 agent 探索）。
- **加日志**：落 agent 时记一条，供 eval 盯"骨架失败率"，别让 outliner 老失败被静默掩盖。

## 测试（mock LLM，沿用现有模式，只验解析/接线/降级，不验真 LLM 判断质量）

- `IntentClassifier`：降噪+意图解析；explain / other 出口；降级默认 `other` + 原 query。
- `AnswerOutliner`：骨架解析（含尺寸 1 节 / 多节）；空列表/解析失败 → 空（降级信号）。
- `QueryPreprocessor`：瘦身后仍按 denoised_query 出六分类；不再回 rewritten_query（更新相关断言）。
- `qa.explain`：stub outliner+检索，验"宽召回→列骨架→扇出→教学体三段"接线；空骨架→落 agent；
  agent 失败→单轮兜底；空节点略过。
- `doc_workflow`：explain 意图 → ExplainEvent → explain_branch，**不进难度分类**；
  非explain → 难度分类→分支原样；rewritten_query 来源改 Call A 后现有用例仍绿。

## 已知缺口（标记，留以后）

- explain 内**非空但错的骨架**（多跳题列出浅骨架）：agent 兜底只接空骨架，接不住"错骨架"。留观察。
- **意图 taxonomy 封口**：v1 二元；将来长 lookup/compare/design 时按答案形状 MECE 切、有默认。
- **教学体合成的 synthesize 整合模式**：v1 逐节分节；需跨节整合的概念暂靠开场全景串，不做专门整合。
- **评测**：golden 加 explain 样例 + "是否真讲透彻"的度量。

## 命名待评审时定

- Call A：`IntentClassifier`（倾向）。若嫌"只是 intent"名不副实（它也降噪），备选 `QueryGate`。
- 新能力方法 `qa.explain` / 新事件 `ExplainEvent` / 新组件 `AnswerOutliner`：随评审。
