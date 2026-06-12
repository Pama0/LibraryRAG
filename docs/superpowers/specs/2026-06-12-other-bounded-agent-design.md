# other → 有界 agent（高难度问题自由探索）Design

**日期:** 2026-06-12
**状态:** 设计已定，待实现（other→agent 两步走的第二步）
**关联:** [other 独立分支（第一步）](../plans/...)（commit `fb0796f2`）· [项目架构](../../ARCHITECTURE.md) §2「按可预测性配控制结构」

## 1. 背景

第一步已把 `other` 从 `case _:`（与 retrievable/解析失败混走单轮检索）独立成 `OtherEvent → other_branch`，并补了"定义说检索不了、却走单轮检索"的裂缝。`other_branch` 当前暂与单轮检索同构，留好接口位。

第二步：把 `other` 真正实现为"高难度/开放问题 → 有界 agent 自由多轮调用工具探索"。这兑现 ARCHITECTURE.md §2 的核心原则——低可预测任务（路径要模型自己决定）配 **agent**，且"agent 作 workflow 的一个 step"。

## 2. 决策（已敲定）

| 维度 | 决策 |
|---|---|
| agent 载体 | 新建 `QaAgent`（LlamaIndex `FunctionAgent`），用其包 `book_search` + `list_books` 两工具。现有 `core/tools/book_tools.py` 依赖退役 `book_rag`、是坏的，**不救活**。 |
| 边界 + 超界 | `max_iterations=6` + `early_stopping_method="generate"`（超界不抛错，基于已收集工具结果做最后一次 LLM call 强制作答）。 |
| other 判定 | **积极**：preprocess prompt 放宽，把"跨主题综合/多步推理/开放设计权衡"的复杂问题主动判 `other` 交 agent，质量优先。 |
| 流式 | `ToolCall→RetrievalStartEvent`、`ToolCallResult→RetrievalDoneEvent`；中间 thought 不外露；最终答案由 chat.py 既有 `answer` 事件兜底。**前端零改动**。 |

## 3. LlamaIndex API 锚点（已核实本地源码）

- `FunctionAgent(tools=, llm=, system_prompt=, early_stopping_method="generate")`。
- `handler = agent.run(user_msg=query, max_iterations=6)`；`async for ev in handler.stream_events()`；`final = await handler`，`str(final)` 为答案。
- 流式事件类（`llama_index.core.agent.workflow.workflow_events`）：`ToolCall(tool_name, tool_kwargs, tool_id)`、`ToolCallResult(tool_name, tool_output, ...)`、`AgentStream`、`AgentOutput`。
- 超界：`num_iterations >= max_iterations` 时，`generate` → `_generate_early_stopping_response`（不抛）；`force` → `WorkflowRuntimeError`。

## 4. 实现取舍（我定的默认，可 review 调整）

1. **工具是检索器，不是合成器**：`book_search` 返回**检索到的原文片段**（top-k 拼接，截断），让 agent 多轮综合。符合 agent 范式（工具取数、agent 推理），也避免工具内重复合成。grounding 靠 system prompt 强约束"只基于检索片段"。
2. **QaAgent 每请求新建**：随 `DocQueryWorkflow`（已是每请求新建）。故可用实例变量持 per-run 的 scope + source 收集器，无并发（per-session 锁 + 每请求新实例）。
3. **source_nodes 由工具闭包收集器回传**：`book_search` 把检索到的 nodes 追加进 `self._run_sources`；`QaAgent.run` 结束后取出，随答案带回 `other_branch` → `FinalizeEvent`。
4. **agent 不接外层会话 memory**：`other_branch` 收到的 `rewritten_query` 已由门口 Router 消指代、自包含；agent 多轮的工作记忆由 `FunctionAgent` 内部维护。MVP 不注入会话历史，避免耦合。
5. **流式桥接位置**：在 `QaAgent.run` 的 `stream_events` 循环里监听 `ToolCall`/`ToolCallResult` 转译进外层 `ctx`；source 收集在工具内（因 `ToolCallResult.tool_output` 是 str、拿不到 nodes）。

## 5. 流式映射（前端零改动）

```
agent ToolCall(book_search, {query})  → ctx.write_event_to_stream(RetrievalStartEvent(query))
agent ToolCallResult                  → ctx.write_event_to_stream(RetrievalDoneEvent(count=已收集数))
agent 跑完 final                       → ctx.write_event_to_stream(AnswerDeltaEvent(final))；并作 FinalizeEvent.answer
```
chat.py `_format_event` 已把这三类映射成 tool_call / tool_result / delta，且 `final` 还会经 `answer` 事件兜底。无新事件类型。

## 6. grounding + 降级

- system prompt 强约束：只基于 `book_search` 返回的检索片段作答，检索不足须如实说明，不得用训练知识脑补（沿用 `BookAgent` 的 BOOK_SYSTEM_PROMPT 精神）。
- 工具检索空 → 返回"（未检索到相关内容）"，agent 自行决定换 query 或如实告知。
- agent 整体异常（如真抛错）→ `other_branch` try/except 降级为 `qa.retrieve` 单轮，绝不让 other 比单轮更脆。

## 7. 新增 / 改动

- **新增** `core/agent/qa_agent.py`：`QaAgent`（构造 FunctionAgent + 两工具 + run 桥接）。
- **新增** `tests/test_qa_agent.py`：mock `self.agent` 测桥接/收集/降级；工具检索测试。
- **改动** `core/workflow/doc_workflow.py`：`__init__` 建 `self.qa_agent`；`other_branch` 委托 `qa_agent.run`（try/except 降级 `qa.retrieve`）。
- **改动** `tests/test_doc_workflow.py`：`other_branch` 接线测试（stub `wf.qa_agent.run`）。
- **改动** `core/workflow/query_preprocess.py`：`_JUDGE_PROMPT` 的 other 段从"兜底剩余"改为"主动识别高难度（积极）"。
- 依赖方向：`core/agent` → `core/workflow.qa_capability`（复用流式事件类 + 检索 helper）→ 仍 core 内部，分层守卫不破（api→core→configs 不变）。

## 8. 不做（YAGNI）

- 不注入会话 memory 给 agent（MVP）。
- 不流式 agent 中间 thought（前端零改动优先）。
- 不做 token 预算（先只 max_iterations + timeout）。
- 不做多 agent / 工具检索器之外的工具（先 book_search + list_books）。

## 9. 已知风险

- 成本/延迟：多轮 LLM+工具，比单轮贵数倍、慢——max_iterations=6 + generate 收尾控住上界。
- 判定积极 → agent 触发偏多 → 成本偏高，须评测校准 prompt（本期先放宽，留评测位）。
- 质量/grounding 不确定 + 开放推理评测难：靠 system prompt 约束 + 后续评测。
- QaAgent 单测难（FunctionAgent 是真实组件）：MVP 用 mock agent 替身测桥接逻辑，不测 FunctionAgent 本身。
