# core/workflow/ —— 文档问答 workflow 包

book 问答走 **`DocQueryWorkflow`**（顶层编排）+ **`FrontDoor`**（门口三步）+ **`QaCapability`**
（QA 检索/合成实质），由 **`DocQueryService`** 在装配层组装，`api/main.py` 与根 `main.py` 各自注入。

> 历史：早期 `book_rag.py`（`BookRagWorkflow`）+ `core/tools/book_tools.py` 已退役删除；
> 单体门口 `IntentRouter` 已拆为下面的 `FrontDoor` 三步。

## 整体流程

`DocQueryWorkflow`（`doc_workflow.py`）是顶层 step 图，自身不持检索/合成实质，只编排 + 薄委托：

```
start → clean_question → split_query → route ─┬─ qa_branch          → finalize
                                              ├─ direct_reply_branch → finalize
                                              └─ study_plan_branch   → finalize
```

- **门口 `FrontDoor`**（`components/new_front_door.py`）：三步各一次独立 LLM 调用、各自降级——
  1. `clean`：读会话记忆消指代 + 规范化 → `clean_query`；指代无法消解则标 `is_missing_info`，直接反问。
  2. `split_query`：把 `clean_query` 拆成若干互不相同的子问题（`FunctionAgent` + `probe` 工具，
     存疑挂法先探库再决定拆不拆）。
  3. `route`：给【每个】子问题判一个出口——dispatch_qa / study_plan / converse / clarify。
- **`QaCapability`**（`qa_capability.py`）：消费门口路由计划（`list[RoutedSubQuery]`）。每个 QA 子问题
  并行判定（`_decide_subq`：probe → `Admitter` 判 ok/missing_info/out_of_scope 并算 per-subq scope →
  `QueryClassifier` 判 category），再按路由顺序执行 ok 子问题（`_execute_subq` 按 category 检索 + 流式合成）；
  converse 子问题直接装饰、不检索。单子问题退化为旧单路径（无分节标题）。
- **`DocQueryService`**（`doc_query_service.py`）：装配层，按名解析可插拔检索组件，每请求新建 workflow。

> 历史：老门口 `FrontDoorAgent`（`front_door.py`）已被 `FrontDoor` 取代并删除；其 `RoutedSubQuery`
> 契约类已迁入 `components/new_front_door.py`。

## 两层记忆纪律（关键，别混成一锅）

- **会话记忆**：真·用户 turn（存用户原话，非改写版）+ 最终答案。门口只【读】它消指代；仅在 `finalize` 写。
- **本轮工作态**：`clean_query`、子问题、category、中间产物——只走 `ctx.store`，【绝不】写进会话记忆，
  否则下一轮消指代会读到污染历史。

## 流式

检索/合成进度通过 `ctx.write_event_to_stream` 推【流式专用事件】（`RetrievalStartEvent` /
`RetrievalDoneEvent` / `AnswerDeltaEvent`，定义在 `qa_capability`，`doc_workflow` re-export 供 api import），
api 层映射成前端 SSE（RetrievalStart→tool_call、RetrievalDone→tool_result、AnswerDelta→delta）。
这些事件不参与 workflow step 图。

## 协作组件

| 文件 | 角色 |
|---|---|
| `admitter.py` (`Admitter`) | 可答性闸：probe 证据判 ok/missing_info/out_of_scope，同时算 per-subq scope |
| `query_classifier.py` (`QueryClassifier`) | ok 子问题判 category：explain/compare/simple/complex |
| `answer_outliner.py` (`AnswerOutliner`) | explain 路列讲解骨架（维度）；列不出 → `EmptySkeleton` 落 agent 兜底 |
| `query_decompose.py` (`QueryDecomposer`) | 子问题再拆解 |
| `query_dimension.py` (`DimensionExtractor`) | ambiguous 归纳维度 |
| `chapter_tree.py` | scope 章节树工具（dominant_prefix / children / unique_chapters） |
| `summarizer.py` | 会话记忆增量摘要（`SUMMARY_MARKER` 头，门口 `format_history` 永远保留） |

## 新增一个分支/能力

1. 在 `query_classifier.py` 的 category 体系挂上新类别（如需）。
2. 在 `qa_capability.py` 的 `_execute_subq` 加该 category 的执行分支（参考 `retrieve`/`explain`/`split`/`assume`），
   通过 `ctx.write_event_to_stream` 推流式事件。
3. 新的【路由出口】（与 dispatch_qa 并列）才需动 `route_prompt.md` + `FrontDoor._RouteItemModel` +
   `doc_workflow.py` 的 `route` 聚合与对应 branch step。
4. 决策开关（供评测 ablation）走 `DocQueryWorkflow` 的 flag 参数。

分层约束：`api/`(Web) → `core/`(领域) → `configs/`，core 不依赖 api（守卫 `scripts/check_layering.py`）。

> 规划中的下一步架构（并发子问题 + 统一流式合成 agent，study_plan 下沉为 qa category）见
> `docs/superpowers/specs/2026-06-23-concurrent-subquery-fanin-combiner-design.md`，尚未实现。
