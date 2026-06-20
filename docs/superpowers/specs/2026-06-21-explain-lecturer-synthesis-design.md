# 设计：讲师级 explain —— 教学 schema + TOC 高度 + 整合写作（slice 2）

> 把 explain 从"像罗列、讲碎片"升到"像讲师讲透一个概念"。结构靠**教学 schema 维度**
> 定（自上而下、不从召回派生）+ 书的 **TOC 撑"组成"高度**；合成从 slice 1 的逐节孤立拼接
> 换成**一次整合教学写作**。
> 方向与取舍见 [docs/intent-axis-explain-note.md](../../intent-axis-explain-note.md) §8。

## Context

slice 1（已合并 master）给了 explain 一条独立工作流，但产出有两个病：

1. **罗列、不连贯**：逐节孤立合成（每节只看自己 chunk、各写各的，无主线/承接）。
2. **讲碎片、高度不对**：骨架**锚定宽召回**，而向量召回按相似度返回——内核书里
   `FIL_PAGE_UNDO_LOG` 类细节又多又密被捞上来，outliner "只依据召回" 就列了一堆碎的。

两个病要两味药。参考了 gpt-researcher（`D:\AgentLearn\gpt-researcher\gpt_researcher\prompts.py`）：
其 `generate_search_queries_prompt` 子查询是平的（不解决高度），`generate_report_prompt` 对汇总
context **一次整合写**（印证"覆盖/写作解耦"）——但它"能堆多少堆多少"的本能正是讲师的反面。
**借它整合写的架构，反着用：讲师做减法、选高度，不堆细节。**

## Goals / Non-goals

**Goals**
- `AnswerOutliner` 改造：召回锚定平列 → **教学 schema 维度化**（固定词表：是什么/作用/组成/原理/适用·边界/关系），模型选子集 + 实例化 + 按 query 定深浅；"组成"维吃 **TOC 提示**。
- `qa.explain` 合成段改造：逐节孤立拼接 → **一次整合教学写作**（`_teach_synthesize`），教案当脚手架，讲师 prompt 立 grounding 铁律 + 做减法，输出轻分节 + 节内连贯 + 开场/收束。
- eval：加 explain 类 golden 样例 + **faithfulness 评测侧度量**（守住"丢结构防幻觉"的真实风险）。
- 保留 slice 1 覆盖半截与降级阶梯，只换"骨架来源"和"合成方式"。

**Non-goals（v1 不做）**
- **运行时 faithfulness 拦截/再生成**：只评测侧度量，运行时不拦（与红线一致：不上 agent 回路/控成本）。
- **TOC 结构性推出组成**：TOC 只当提示；`dominant_prefix/children` 主导书推断留后续。
- **自定义"高度/教学法"指标**：碎片/讲透/高度靠真实冷烟 + 人读判；LLM-judge 教学指标留后续。
- **explain 多跳专门处理**：仍走 `EmptySkeleton → agent`，不变。
- **compare / lookup / design 意图**：仍非 explain → 难度分类。
- **Graph-RAG**：明确不上（单书 TOC 已是人工层级图）。重启条件见 note §8.5。

## 架构：组件（改造 + 新增，尽量复用）

| 组件 | 变化 | 职责 |
|---|---|---|
| `AnswerOutliner`（改造） | 召回锚定 `list[str]` → 教学维度化 | 入 clean_query + 宽召回 passages + TOC 提示，出**教案** `list[Dimension]`（label∈词表、query=检索子查询） |
| `qa.explain`（改合成段） | 逐节 concat → 整合写 | 宽召回 → 出教案 → 每维度检索扇出 → merge pool → `_teach_synthesize` |
| `_teach_synthesize`（新） | 取代 explain 里 `_retrieve_and_concat` 用法 | 教案 + 截断/重排 pool → 讲师 prompt → 直接 stream LLM；轻分节 + 节内连贯 + 开场/收束 |
| eval（新增） | golden + 指标 | explain golden 样例 + harness 识别 explain + faithfulness/answer_relevancy 上报 |

**复用（不动）**：教案条目用现成 `core.workflow.query_dimension.Dimension(label, query)`（assume 已在用，结构同源）；
宽 hybrid 召回 / 每维度 `_retrieve_all` / `_merge_pool` / `EmptySkeleton → agent → 单轮` 阶梯 /
`ExplainEvent`·`explain_branch` / `QueryGate` / finalize 的 `intent` metadata —— 全留。

## 教学 schema + 高度机制

**固定维度词表**（自上而下排序，模型只能选用、不自创）：

| 维度 | 讲什么 | 备注 |
|---|---|---|
| 是什么 | 定义/定位 | |
| 作用 | 解决什么问题、为什么需要 | 动机先行 |
| 组成 | 由哪些部件构成 | **吃 TOC 提示**——顶层章节即部件高度 |
| 原理 | 部件怎么协作/工作机制 | |
| 适用·边界 | 什么场景用、何时当心 | |
| 关系 | 与相邻概念的联系/对比 | 横向连接 |

**高度由 query 定**（写进 outliner prompt 的判据）：
- 宽/入门（"讲懂mysql"）→ 靠前维度（是什么/作用/组成）、每维浅、停高处不下钻部件内部。
- 具体/深（"MVCC的实现原理"）→ 聚焦被问维度（原理）下钻，前置维度一句带过。

**根治"碎片"两处守高度**：
1. **outliner**：维度只能来自固定词表，**绝不从召回派生** → 召回里的 `FIL_PAGE_UNDO_LOG` 不会被提成顶层小节。
2. **teach 合成**：按教案高度写，**低层 chunk 只在支撑某维度论点时才用，否则做减法丢掉**（讲师 prompt 立这条）。

## 数据流：qa.explain（改造）

```
qa.explain(ctx, query, book_titles):
  1. 宽 hybrid 覆盖召回 → passages                              （保留）
  2. toc_hint = _book_chapters(book_titles) 的章节标题；多书/未选/无净 TOC → 空
       （空提示 → outliner 退 schema-only 定组成）
  3. outline = await outliner.run(query, passages, toc_hint)   # list[Dimension]
       空 → raise EmptySkeleton → explain_branch 落 agent 再落单轮  （保留阶梯）
  4. ctx.write_event_to_stream(RetrievalStartEvent(query))
     retrieved = _retrieve_all([d.query for d in outline]); pool = _merge_pool(retrieved)
     pool 截断/重排到上下文预算（有 reranker 用之、否则按 score 取 top-N=rerank_candidate_k）
     ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
  5. answer = await _teach_synthesize(ctx, query, outline, pool)   # 一次整合写
  6. return answer, pool
```

`_teach_synthesize` 内：组装讲师 prompt = [讲师角色 + 反"堆全"指令] + [教案：用到的维度顺序] +
[grounding 铁律：事实只依据下面片段、不引入外部知识、与教案高度无关的细节做减法] + [pool 片段]；
`llm` 直接 stream，逐 token 发 `AnswerDeltaEvent`，拼回完整答案。输出：开场全景 → 轻分节
（`## 维度`，仅用到的）节内连贯 → 收束。

### grounding 模型（这刀根基）

| | slice 1 | slice 2 |
|---|---|---|
| 结构从哪来 | 召回锚定（→碎片） | **schema 维度（安全元知识）+ TOC 提示** |
| 事实 ground | 逐节结构隔离 | **讲师 prompt 立"只依据片段"+ 做减法**；faithfulness **评测侧**盯 |
| 合成 | 逐节孤立拼接（→罗列） | **一次整合教学写作** |

结构来自教学先验（"讲X要讲组成"是安全教学常识、非编事实）+ 书的 TOC；事实只来自 chunk。

## eval（评测侧）

- **golden 加 explain 样例**：标 `intent: "explain"`，含宽题 + 具体题。faithfulness/answer_relevancy
  不需参考答案（比的是"答案 vs 检索 context"），样例可不写长 reference。
- **指标**：
  - **faithfulness**（关键闸）——slice 2 丢了结构防幻觉，靠它盯整合写作有没有飘。
  - answer_relevancy——防跑题。
- **harness 改动**：explain 条目经 workflow 天然走 explain；据 metadata `intent` 识别 → 算
  faithfulness/relevancy、纳入均值，但**不计入难度分类准确率**（explain 无 category）；
  null 规则不误伤（explain 不在 REFUSE_CATEGORIES）。
- **诚实边界**：ragas 量 grounding + 切题，**量不了高度/碎片/讲透**——那靠真实冷烟 + 人读；
  自定义教学 LLM-judge 留后续。

## 测试（mock LLM，验解析/接线/降级，不验真 LLM 质量）

- `AnswerOutliner`：维度化输出解析（`list[Dimension]`，label∈词表、query 非空）；TOC 提示进 prompt；空/解析失败 → `[]`。
- `qa.explain`：stub outliner/`_explain_recall`/`_book_chapters`/`_retrieve_all`/`_teach_synthesize`，验
  每维度检索 + 一次 teach 调用、merge pool 回流；空教案 → raise `EmptySkeleton`。
- `_teach_synthesize`：stub `llm` 流，验教案(维度)+grounding 铁律+pool 片段都进 prompt、一次流、轻分节 `##` 发出。
- eval harness：explain 条目（intent=explain）被识别 → 算 faithfulness/relevancy、不计入分类准确率、不被 null。

## 决策锁定（评审依据）

1. faithfulness = **评测侧度量**，运行时不拦不再生成。
2. 教学 schema = **固定维度词表**，模型选子集 + 实例化 + 按 query 定深浅。
3. TOC = **强提示**喂 outliner 的"组成"维度，无净 TOC 退 schema-only。
4. 输出 = **轻分节 + 节内连贯 + 开场/收束**，一次整合写。

## 已知缺口（留后续）

- TOC 结构性推组成（主导书推断 + `chapter_tree.children`）——提示不够时再上。
- "高度/碎片/讲透"的自定义教学 LLM-judge 指标。
- explain 内多跳（现走 agent 兜底）。
- `_teach_synthesize` 的讲师 prompt 文案精修（v1 给可用版，冷烟后迭代）。

## 命名

- 教案条目复用 `Dimension(label, query)`（若嫌"评判维度"语义偏，可另立 `OutlineItem`——评审时定）。
- 新合成方法 `_teach_synthesize`；改造组件仍叫 `AnswerOutliner`（职责仍是"列答案骨架"，只是骨架改教学维度）。
