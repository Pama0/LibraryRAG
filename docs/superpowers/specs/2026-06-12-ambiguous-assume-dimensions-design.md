# ambiguous → assume：归纳评判维度 + 声明角度 Design

**日期:** 2026-06-12
**状态:** 设计已定，待实现
**关联:** [split_branch 拆解-检索-汇总](2026-06-11-split-branch-decompose-design.md) · [项目架构](../../ARCHITECTURE.md)

## 1. 问题

`QueryPreprocessor` 把"主题已具体、能集中命中，但用户想要的**维度/立场**未给"的问题判为
`ambiguous`（如「Vue和React哪个好」缺评价维度、「Redis做缓存好吗」缺评判角度）。

当前 `QaCapability.assume` 是占位——直接等同 `retrieve`（整句单轮检索合成）。问题：
LLM 会**自己悄悄挑一个角度**回答，① 不透明（用户不知按哪个角度答了）；② 可能挑错或
糊成泛泛而谈。

与邻类的边界（决定策略）：
- `missing_info` → 反问（clarify）：缺检索必需限定，**根本检索不了**。
- `pending_split` → 拆解汇总：要**罗列并列子项/章节**才完整。
- `ambiguous` → **assume**：能命中，只是**答的角度不定**。`assume` 的本意 = 不打断用户，
  **自行假设最合理角度、显式声明、再回答**，把纠偏权交回用户。

## 2. 方案

`assume` 与 `split` **同一条流水线**，差别只在第 2 步骨架来源与第 3 步声明：

```
1) 定位      整句宽召回，命中主题内容（同 split）
2) 归纳维度  LLM 从「问题 + 召回正文」产出 2~N 个评判维度
            每维度含 (label 维度名, query 检索子查询)
3) 声明      答案开头注入角度声明 preamble（透明 + 可纠偏）
4) 逐维度    每子查询各自检索 + 分节合成（## {label}）
5) 降级      归纳不出维度 → 退回单轮合成（用已定位结果，绝不阻塞）
```

**维度来源**（已定）：纯 LLM 从「问题 + 召回正文」归纳，遵 `QueryDecomposer` 的
「只依据给定素材、严禁编造」铁律。不锚定章节结构（评判维度常横切多个章节，目录层级
对不上）。

## 3. 与 split 的复用（DRY）

split 与 assume 的「逐项检索 → 发一次 RetrievalDone →（可选声明）→ 逐节流式合成拼接」
完全同构。抽成 `QaCapability._retrieve_and_reduce(ctx, sections, book_titles, preamble="")`：
- `sections: list[(标题, 检索/合成子查询)]`。split 传 `[(sq, sq) …]`；assume 传 `[(label, query) …]`。
- `preamble`：可选，进入答案阶段后先推一个 `AnswerDeltaEvent`，并拼在答案最前。split 不传。

split 主路径（有 sub_queries）改为调该 helper，**行为不变**，由现有 split 测试守护。
split / assume 各自的「定位、降级（空维度/空命中）」留在各自方法内（降级复用已定位的
`located`，不重复检索）。

## 4. 流式（前端零改动）

复用既有 SSE 词汇：`RetrievalStartEvent`（定位时一次）→ 各路检索后**一次**
`RetrievalDoneEvent` → preamble 声明 `AnswerDeltaEvent` → 每维度标题
`AnswerDeltaEvent(delta="\n## {label}\n")` + 该节合成 token。与 split 的事件序列同形。

## 5. 新增 / 改动

- **新增** `core/workflow/query_dimension.py`：`DimensionExtractor`（注入 LLM）+ `Dimension`
  / `DimensionSet` schema。`run(clean_query, passages, max_items) -> list[Dimension]`，
  失败/空 → `[]`。与 `QueryDecomposer` 同模式（独立可测）。
- **改动** `core/workflow/qa_capability.py`：`__init__` 加 `self.dimensioner`；新增
  `_retrieve_and_reduce` helper；`split` 主路径改用 helper；重写 `assume`。
- `doc_workflow.py` 的 `assume_branch` 已委托 `self.qa.assume`，**不改**。

## 6. 降级与错误处理

- `DimensionExtractor` 解析失败 / 空 → `[]` → `assume` 退回单轮合成。
- 定位空命中（`located` 为空且无维度）→ 范围提示（同 split / retrieve）。
- 全程不阻塞，最差等同当前 v1（整句检索合成）。

## 7. 不做（YAGNI）

- 不做 L3「轻量反问选项」（违背 assume 不打断初衷，预算策略后续再议）。
- 不锚定章节结构产维度。
- 维度数量上限复用 `max_sub_queries`，不新增配置。
