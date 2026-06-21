# 设计：可答性前置闸 —— 把 out_of_scope / missing_info 抽成共享判定单元

> 把"能不能答"这条轴从难度六分类里**抽出来**，做成一个共享判定单元 `Admitter`，
> 让 **explain 和 other 两条路都先过它**。explain 至此也被库外/信息不足覆盖，
> 修掉"库外问题溜进 explain → 落 agent → 脑补讲一顿"的根因。
> 起因诊断见本文 Context；与意图轴/难度轴的关系见 [routing-architecture-note.md](../../routing-architecture-note.md)。

## Context

线上观察：让系统"讲东西"时，**不管资料里有没有，它都讲一顿**。根因（代码层已定位）：

- 产品主路 `DocQueryWorkflow`：gate（降噪+意图）→ explain / other 分流。
- `out_of_scope` / `missing_info` 这两条"能不能答"的判定，**只活在 other 路的难度六分类器**（`QueryPreprocessor`，靠 probe 探测召回判）里。
- **explain 路整段跳过难度分类**（gate→explain→outline），于是库外/信息不足的 explain 问题**完全不被筛**：宽召回捞到一堆不相关片段 → outliner 列不出干净骨架 → `EmptySkeleton` → 落 `QaAgent` 兜底 → `QaAgent` 没有库外拒答收场 → 用世界知识脑补一整篇。
- 修 `AutoAgent`（评测 agent）库外拒答时，曾用"QaAgent 走 router、上游已分类"为由跳过 QaAgent——**这个前提被 explain 打破了**：explain 不经分类就把库外送进了 agent。

**正交轴诊断**：`out_of_scope` / `missing_info` 既不是"怎么取"（检索结构轴），也不是"怎么写"（答案意图轴），而是**可答性轴**——逻辑上**先于**意图与难度。它被埋在难度分类器里，等于这条轴只在 other 分支生效。**把它提出来当前置闸，两条路一起筛**，是 routing-note"正交轴压扁"在此处的同源修复。

**为什么不能盲判**：`out_of_scope` 无法只凭 query 文本判（"PostgreSQL 是库外吗"要检索后看"只捞到 MySQL 片段"才知道）。现有六分类器的铁律整段建立在"以末尾【知识库探测召回】为准"。所以可答性判定**必须吃检索证据**——这决定了它埋在"证据已产生的地方"。

## Goals / Non-goals

**Goals**
- 新增 `Admitter`（可答性判定单元）：吃 query + 检索证据，判 `ok / missing_info / out_of_scope`。判据从六分类器**原样搬**那两段（含"只看主体实体在不在库""深度/角度不匹配≠库外"等已调细的铁律）。
- **explain 路**：在其已有宽召回之后插一句 `admit`，库外/信息不足 → 抛信号（`OutOfScope`/`MissingInfo`）由 `explain_branch` 接住 → 拒答/反问。
- **other 路**：`qa.classify` 内部改成 `probe → admit → (ok 才跑瘦身分类器)`；**workflow 边界不变**（对外仍可能返回 6 类中任意一类）。
- **难度分类器瘦身**：enum 从 6 类删成 4 类（`retrievable / pending_split / ambiguous / other`），prompt 删掉 `out_of_scope` / `missing_info` 两段。
- **话术归一**：反问句单一来源（`Admitter` 产）；拒答话术抽成共享常量，库外分支与 explain 拒答都引用。
- 不重排热路径：probe / 宽召回的位置与 retriever 装配**一律不动**，只在证据产生处加一次 `admit` 调用。

**Non-goals（明确不做）**
- **统一/前移 probe**（方案 3）：不把 probe 提到分流前共享。explain 宽召回（hybrid 大 top_k 求覆盖）与 classify probe（vector 不重排求章节 spread）**取向相反**，强行合并会压扁 `pending_split` 的 spread 信号或削掉 explain 覆盖。两路各带各的证据。
- **运行时再生成/多轮校验**：`Admitter` 一次判定，失败优雅降级，不加回路。
- **`ambiguous` 迁移**：`ambiguous`（角度不定）是"按哪个角度答"、判错能优雅降级，属意图/答案形状轴，**留在难度分类器**，不进可答性闸。
- **QaAgent 库外拒答补丁**：那是**另一刀**（防御纵深的最后一道）。本刀是前置闸（真·根因）；两者不互斥，见"降级"。

## 架构：组件（新增 + 改造）

| 组件 | 变化 | 职责 |
|---|---|---|
| `Admitter`（新） | 全新决策单元 | `run(query, evidence_passages) -> AdmitVerdict`；只判可答性轴 |
| `qa.classify`（改） | 内嵌 admit | `probe → admit`；非 ok 短路返回该类；ok → 瘦身分类器。对外契约不变 |
| `qa.explain`（改） | 宽召回后插 admit | admit 非 ok → 抛 `OutOfScope`/`MissingInfo`；ok → 进 outline |
| `QueryPreprocessor`（瘦身） | 6 类 → 4 类 | 删 `out_of_scope`/`missing_info` 判据，只判检索结构/难度 |
| `explain_branch`（改） | 接两异常 | catch `OutOfScope`→拒答 finalize、`MissingInfo`→反问 finalize，并写 `category` 入 ctx |
| 拒答话术（抽常量） | 去重 | 库外终结句抽成共享常量，库外分支与 explain 拒答共用 |

**复用（不动）**：probe（`_probe_retrieve`）/ explain 宽召回（`_explain_recall`）/ 现成 `ClarifyEvent`·clarify 分支 / 库外终结分支 / gate / front_door —— 全留。`Admitter` 沿用决策单元约定（注入 LLM、只暴露 `run`、`json_object`+Pydantic 校验、失败降级、自带 `_strip_fences` 副本）。

### Admitter 接口与 schema

```
Admitter.run(query: str, passages: list[str]) -> AdmitVerdict

AdmitVerdict (Pydantic, 代码侧校验):
  verdict: Literal["ok", "missing_info", "out_of_scope"]   # 默认 "ok"
  reason: str = ""             # 判定理由（日志/调试）
  clarify_question: str = ""   # missing_info 专用：面向用户的自然反问句
```

- prompt：把六分类器里 `out_of_scope` / `missing_info` 两段判据**原样搬入**（含优先级"先看主体实体在不在库"、"深度/角度/广度不匹配≠库外"反例、missing_info 多为指代不明等），加一句"其余皆 `ok`"。
- 证据由调用方喂，`Admitter` 不自检索。

## 数据流

### other 路（qa.classify 内部，对外契约不变）

```
qa.classify(clean_query, book_titles, probe=True):
  located = _probe_retrieve(clean_query, book_titles)          # 不变
  evidence = _format_probe(located, book_titles)               # 不变
  verdict = await admitter.run(clean_query, [evidence])        # 新增
  if verdict.verdict == "out_of_scope":
      return PreprocessResult("out_of_scope", verdict.reason)
  if verdict.verdict == "missing_info":
      return PreprocessResult("missing_info", verdict.reason, verdict.clarify_question)
  return await preprocessor.run(clean_query, evidence)         # 瘦身分类器，只出 4 类
```

> workflow `preprocess` step 仍调 `qa.classify`、仍按返回的 category 发现有分支事件（含 OutOfScope/Clarify）——**other 路 step 图与话术零改动**。

### explain 路（qa.explain 内部 + explain_branch 接异常）

```
qa.explain(ctx, query, book_titles):
  located = _explain_recall(query, book_titles)                # 不变
  passages = [n.text[:500] for n in located]
  verdict = await admitter.run(query, passages)                # 新增
  if verdict.verdict == "out_of_scope":
      raise OutOfScope(query)
  if verdict.verdict == "missing_info":
      raise MissingInfo(verdict.clarify_question)
  ...（toc_hint → outline → 检索扇出 → _teach_synthesize，均不变）

explain_branch(ctx, ev):
  try:
      answer, nodes = await qa.explain(ctx, rewritten, book_titles)
  except OutOfScope:
      await ctx.store.set("category", "out_of_scope")          # 供 eval 计分
      return FinalizeEvent(answer=REFUSAL_TEXT, source_nodes=[])
  except MissingInfo as e:
      await ctx.store.set("category", "missing_info")
      return FinalizeEvent(answer=e.clarify_question or REFUSAL_FALLBACK, source_nodes=[])
  except EmptySkeleton:
      ...（落 agent 兜底，原样保留）
```

- `OutOfScope` / `MissingInfo` 镜像现成 `EmptySkeleton` 的异常驱动控制流，放 `qa_capability.py`。
- explain 拒答/反问终结前**写 `category` 入 ctx**，让 finalize 的 metadata 带上（否则评测里这些行 category 空、算不进分类准确率）。

## 话术归一

- **反问句**：唯一来源 `Admitter.clarify_question`。other 路经 `PreprocessResult.clarify_question` 透传给现成 `ClarifyEvent`；explain 路经 `MissingInfo` 异常带回。
- **拒答话术**：现写在 workflow 库外分支里的终结句（"……你可以换个已入库主题问我，或把问题换个角度再试试～"）抽成**共享常量** `REFUSAL_TEXT`，库外分支与 explain `OutOfScope` catch 都引用，避免 explain 另写一句又分叉。`REFUSAL_FALLBACK` 为 missing_info 缺 clarify_question 时的兜底反问。

## 降级（绝不阻塞）

| 触发 | 落点 |
|---|---|
| `Admitter` 解析失败/空 | 默认 `ok`（放行去作答） |
| explain admit=ok 但后续空骨架 | 原样 `EmptySkeleton → agent → 单轮` |
| other admit=ok | 跑瘦身分类器；其失败仍降级 `retrievable`（不变） |

- **降级方向=放行**，与现有 `QueryPreprocessor` 失败降 `retrievable`、`gate` 失败降 `other` 同一哲学：**判定器坏了不该误拒正常问题**。
- 残留风险（admit 失败时库外问题溜过去脑补）由**另一刀 `QaAgent` 库外拒答补丁**当最后兜底接住——两层防御纵深。本刀落地后，那个补丁退化成纯保险（仍建议做）。

## 测试（mock LLM，验解析/接线/降级，不验真 LLM 判断质量）

- `Admitter`：解析 `ok` / `missing_info`（带 clarify_question）/ `out_of_scope`；证据进 prompt；解析失败/空 → `ok`。
- `qa.classify`：admit=out_of_scope/missing_info → 短路返回该类、**不**调瘦身分类器；admit=ok → 调瘦身分类器返回其 4 类之一（stub admit + preprocessor）。
- `qa.explain`：admit=out_of_scope → 抛 `OutOfScope`；missing_info → 抛 `MissingInfo`（带 clarify）；ok → 进 outline（stub `_explain_recall` + admit + outliner）。
- `doc_workflow.explain_branch`：catch `OutOfScope` → 拒答 finalize + ctx.category=out_of_scope；catch `MissingInfo` → 反问 finalize + ctx.category=missing_info；`EmptySkeleton` 仍落 agent。
- `QueryPreprocessor` 瘦身：4 类解析正常；不再产 out_of_scope/missing_info（迁移/删两类旧用例）；非法/空仍降 retrievable。
- other 路 workflow：边界仍可出 6 类、分支事件与话术不变（回归）。

## 决策锁定（评审依据）

1. **抽共享判定单元（方案 1）**，不统一/前移 probe（方案 3）——两路各带各的证据，热路径不重排。
2. **埋入式编排**：admit 调用放在证据产生处（classify 内 / explain 内），不提成独立 workflow step；explain 走异常信号，与 `EmptySkeleton` 同构。
3. **降级方向=放行（ok）**，靠 `QaAgent` 库外拒答补丁做防御纵深兜底。
4. **`ambiguous` 不迁**，留难度分类器。
5. **话术单一来源**：clarify_question 出自 Admitter；拒答句抽共享常量。

## 已知缺口（留后续）

- **QaAgent 库外拒答补丁**：本刀的防御纵深后手，另开一刀（最小：镜像 AutoAgent 那段收场搬进 `QA_AGENT_SYSTEM_PROMPT`）。
- **可答性闸的真实冷烟**：库外 4 类（PG/Mongo/Oracle/Cassandra）经 explain 路仍判 out_of_scope、且"讲懂MySQL概念"不被误判——需 DEEPSEEK_API_KEY + 索引，人读。
- **missing_info 的文本可判半**：指代不明那半其实文本就能判，本刀仍统一走证据判（一致优先）；将来若要省一次检索再分。

## 命名（评审时定）

- 组件 `Admitter`（备选 `AnswerabilityGate`）；verdict 枚举 `ok / missing_info / out_of_scope`；异常 `OutOfScope` / `MissingInfo`（镜像现成 `EmptySkeleton`，置于 `qa_capability.py`）；共享拒答常量 `REFUSAL_TEXT` / `REFUSAL_FALLBACK`。
