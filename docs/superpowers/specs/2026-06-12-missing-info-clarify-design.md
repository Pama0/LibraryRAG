# missing_info → clarify：自然反问 + 预算耗尽降级声明假设 Design

**日期:** 2026-06-12
**状态:** 设计已定，待实现
**关联:** [ambiguous→assume](2026-06-12-ambiguous-assume-dimensions-design.md) · [项目架构](../../ARCHITECTURE.md)

## 1. 问题

`QueryPreprocessor` 把"缺了检索必需的关键限定、根本无法检索"（多为指代不明，且历史也补不全）的问题判为 `missing_info`（如「这个索引的应用场景」——"这个索引"指代不明）。

当前 `missing_info` 已有真实现（非占位），走 `clarify_branch` 反问、本轮终止，等用户下一轮补充（下一轮 `intent_router` 看到「原问题+反问+补充」一起消解）。**单轮终止 + 两层记忆**的骨架正确，但实现简陋：

```python
question = f"为了更准确地回答，请补充：{ev.clarify_reason}"
```

弱点：① 把 preprocess 给的 `reason`（如"'这个索引'指代不明"）直接拼模板，是在**陈述问题**而非**问用户**，措辞生硬；② `allow_clarify=False`（预算耗尽）降级时直接拿指代不明的 query 去检索，质量差且不透明。

## 2. 方案（L1 + 降级，YAGNI：不做 L2 召回候选 / L3 前端选项）

**L1 自然反问句**：让 `QueryPreprocessor` 在判 `missing_info` 时**顺便产出一个面向用户的完整反问句** `clarify_question`（点明不明处、引导补充、能列候选就列），并入它的输出 schema。零额外 LLM 调用（同一次 judge 调用多产一个字段）。`clarify_branch` 优先用 `clarify_question`，缺失则退回原模板拼 `reason`（绝不阻塞）。

**预算耗尽降级 = 声明假设、尽力答**：`missing_info` + `allow_clarify=False` 时不反问，改为**带假设声明的单轮检索**——挑最可能的解读，答案开头声明"原问题信息不足（{reason}），以下按最可能的解读作答"，再正常检索合成。比拿模糊 query 硬检索体面，语义上与 `assume`（赌一个解读 + 声明）一脉相承。

## 3. 实现要点

- `QueryPreprocessor`：prompt 的 `missing_info` 返回 JSON 加 `clarify_question`；`QueryJudgment` / `PreprocessResult` 各加 `clarify_question: str = ""`；`run` 透传。其它类别不返回该字段时默认空。
- `QaCapability.retrieve`：加可选 `preamble: str = ""` 参数——非空时进入答案阶段先推一个 `AnswerDeltaEvent(preamble)` 并拼在答案最前（与 `_retrieve_and_reduce` 的 preamble 语义一致）；空命中只给范围提示、不带声明。
- `DocQueryWorkflow`：`ClarifyEvent` 加 `clarify_question`；`RetrieveAgentEvent` 加 `assumption_note`；`preprocess` step 的 `missing_info` 两分支分别带上这两个字段；`clarify_branch` 用 `clarify_question`；`retrieve_branch` 把 `assumption_note` 作为 preamble 传给 `qa.retrieve`。

## 4. 流式（前端零改动）

降级路径：`RetrievalStart` → `RetrievalDone` → 声明 `AnswerDeltaEvent(assumption_note)` → 合成 token（均为 `AnswerDeltaEvent`）。clarify 路径不检索，只产一个终止性答案。SSE 映射按 `__class__.__name__`，无新事件类型，前端不变。

## 5. 测试连锁（重要）

`retrieve_branch` 改为传第 4 个参数（`assumption_note`）后，`tests/test_doc_workflow.py` 中现有 5 处 `fake_retrieve(ctx, query, book_titles)` 三参 stub 会因多收一个位置参数而 `TypeError`。实现这步时必须同步把它们的签名改为 `(ctx, query, book_titles, preamble="")`。

## 6. 降级与错误处理

- LLM 未给 `clarify_question`（非 missing_info / 解析失败）→ 默认空 → `clarify_branch` 退回模板。
- `allow_clarify=False` 降级检索若空命中 → 范围提示（不带声明）。
- 全程不阻塞，最差等同当前行为。

## 7. 不做（YAGNI）

- L2「先召回一轮→三态降级（直接答 / 带候选反问 / 开放反问）」——体验更好但有召回噪声风险，等评测看 missing_info 触发频率再议。
- L3 前端结构化选项按钮（需前端联动）。
