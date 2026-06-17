# pending_split 分流：罗列 vs 综合（+ 并行检索）设计

日期：2026-06-17
分支：feat/out-of-scope-classification（或新开 feat 分支）
状态：设计已确认，待写实现计划

## 背景与问题

`pending_split` 类问题当前走 `QaCapability.split()`：定位宽召回 → `QueryDecomposer`
拆成 ≤N 个子查询 → 逐子查询检索 + 逐节流式合成 → `"".join()` 拼接
（`_retrieve_and_reduce`，`qa_capability.py:230`）。

该 helper 名为 reduce，**实际没有 reduce，只是裸拼**：每节合成只看到「自己那个子查询
检索到的 nodes」，彼此不可见，最后字符串拼接。这导致两类 `pending_split` 问题里只有
一类被正确处理：

- **罗列型**（子项彼此独立，各自成段即完整）——裸拼正好对。
  例：「在 MySQL 中，索引、基于成本的优化、EXPLAIN 分析**分别**起什么作用？」
- **综合型**（答案必须跨子项推理：比较 / 讲关系 / 谈协作取舍 / 合成单一概念）——裸拼
  必然答错：跨项信息既不在任一子查询的检索结果里，也没有整合步骤去生成。
  例：「在 MySQL 中，索引、基于成本的优化、EXPLAIN 分析**是如何协作配合**的？请综合
  分析它们之间的关系与设计取舍。」

decomposer 的 prompt 里虽已写「对比类子查询应是各对比维度」，但即便拆成对比维度，
最后还是裸拼，没有「拿到所有维度素材后整合比较」的一步。**缺的是结构里的 reduce。**

附带问题：综合型问题在重变体（全开+hybrid+rerank）下会因 N 次串行合成 + 重检索撞
墙钟超时（eval SUT 默认 120s），表现为 `outcome=error`。

## 目标

1. `pending_split` 内部按「罗列 vs 综合」分流，综合型走真正的整合合成，修正答案正确性。
2. 共享检索 helper 的扇出检索改并行（`asyncio.gather`），降墙钟、缓解超时。
3. 不改门口分类与 workflow step 图：两类问题仍统一归 `pending_split`，分流在 capability 内部。

非目标（本次不做）：
- 不改 `assume`（ambiguous）的语义；它只白拿共享 helper 的并行检索，行为仍是逐角度拼接。
- list 模式的逐节合成并行（涉及流式顺序，留作独立项）。
- eval 端到端验证：开发阶段只做单元测试，eval 整段延后。
- eval SUT 超时阈值（120s）不在本设计调整。

## 整体架构与数据流

判型放在 **decomposer**（它手里「问题 + 召回素材」最全，判得最准），从纯拆解器升为
「拆解 + 判型」，返回 `(sub_queries, mode)`，`mode ∈ {"list", "synthesize"}`。

```
preprocess 判 category=pending_split
  → SplitEvent(rewritten_query, split_reason)        # 不变
  → split_branch → qa.split()                        # 不变（薄委托）
       1) 定位宽召回（同现状）
       2) decomposer.run() → (sub_queries, mode)      # 升级点
       3) mode 分流：
          ├─ "list"       → _retrieve_and_concat       # = 现 _retrieve_and_reduce，改名
          └─ "synthesize" → _retrieve_and_synthesize   # 新增
```

`split()` 对外仍只返回 `(answer, source_nodes)`，对 `split_branch` / `finalize` /
workflow step 图完全透明。

### 判型语义（写进 decomposer prompt）

- `list`（罗列）：各子项答案**彼此独立、各自成段即完整**。典型「分别 / 各自 / 列举各自功能」。
- `synthesize`（综合）：答案**必须跨子项推理**——比较、讲关系、谈协作取舍、合成单一概念。

判断准绳一句话：**「每个子项的答案能不能单独成立？能 → list；必须放到一起才说得清 →
synthesize。」** 对比类多数落 synthesize。

**保守偏置**：拿不准时选 `list`。误判方向的代价不对称——把综合题当罗列，答案碎（可感知、
可纠）；把罗列题当综合，会把一堆不相关片段塞进一次合成、答案发散（更糟）。

## 两种执行模式

### list 模式：`_retrieve_and_concat`（= 现 `_retrieve_and_reduce` 改名）

行为不变，仅两处：
- 改名去掉误导的「reduce」（它从来没 reduce）。
- 扇出检索循环改并行（见下「并行检索」）；逐节流式合成**仍串行**，保分节流式顺序。

输出：多段 `## 标题` + 逐节逐 token 流。

### synthesize 模式：`_retrieve_and_synthesize`（新增）

1. **扇出检索**：对每个子查询检索（并行），目的是把跨子项素材召回全（子查询此处只为拓宽召回面）。
2. **去重合并**：按 node id 去重，合并成单一 node 池。
3. **收口截断**：
   - 有 reranker → 拿**原始问题**（非子查询）对合并池重排，截到预算上限；
   - 无 reranker → 按检索分数排序后截断到同一预算；
   - 预算上限取一个常量（量级同 `rerank_candidate_k` → `similarity_top_k`，实现时定常量），防止
     N×top_k 撑爆 / 稀释上下文。
4. **一次整合合成**：`_synthesize_stream(ctx, 原始问题, 合并池)` 做**单次**流式合成。LLM 同时
   看到所有子项原始片段，才能真正比较 / 讲关系 / 谈取舍。

输出：**单段连贯答案**，一条流——无 preamble、无 `## 小节` 切割。

两种模式都只发**一次** `RetrievalDoneEvent`（扇出检索全做完后），前端 SSE 契约不变。

## 并行检索

两种模式的扇出检索从串行 `for ... await` 改为 `asyncio.gather`：

- list（`_retrieve_and_concat`）：并行检索全部子查询拿到各自 nodes；合成循环不变（串行，保流式顺序）。
- synthesize（`_retrieve_and_synthesize`）：并行检索 → 合并去重 → 单次合成。
- `assume` 调同一改名后的 helper → **白拿并行检索**（不获得 synthesize 模式）。
- 用 `asyncio.Semaphore(并发上限)` 兜底：底层每次检索含 embedding / BM25 / bge rerank 调用，
  限并发防打爆 / 429。上限取一个保守常量（实现时定）。

墙钟：检索从 Σ 各子查询 → max；synthesize 合成从 N 次串行 → 1 次。综合型问题基本不再撞
120s；list 模式合成仍 N 次串行（本设计不碰）。

## 边界与降级

所有异常倒向现状最安全行为，绝不阻塞：

| 情形 | 处理 |
|---|---|
| LLM 解析失败 / 返回空 `sub_queries` | 返回 `([], "list")`；`split()` 走现有降级：单轮检索+合成 |
| 返回 `sub_queries` 但 `mode` 缺失/非法 | 默认 `"list"`（最保守，绝不误把罗列题硬整合） |
| `mode="synthesize"` 但只拆出 1 个子查询 | 退化为单轮检索+合成（1 子项无所谓整合） |
| `mode="synthesize"` 扇出后合并池为空 | 现有空命中文案「在《X》中没有检索到…」 |
| 定位宽召回 `located` 为空 | 维持现状：无素材可判型 → 空命中文案 |

schema 变更：`Decomposition` 加 `mode: Literal["list","synthesize"] = "list"`
（Pydantic 默认值兜底，LLM 漏给也安全）。

## 测试（TDD，开发阶段唯一验证手段）

沿用「mock LLM 控输出 / 桩掉协作者」模式。

**`QueryDecomposer`**（`query_decompose.py`）
- 返回 `{"sub_queries":[...],"mode":"list"}` → `(subs, "list")`
- 返回 `mode:"synthesize"` → `(subs, "synthesize")`
- `mode` 缺失 → 默认 `"list"`
- `mode` 非法值（如 `"foo"`）→ 回落 `"list"`
- LLM 报错 / 空 → `([], "list")`

**`QaCapability.split`**（桩 `_retrieve_nodes` / `_stream_tokens` / `decomposer` / `reranker`）
- decomposer 返回 `list` → 走 `_retrieve_and_concat`：逐节检索+合成+拼接，带 `## 标题`
- decomposer 返回 `synthesize` → 走 `_retrieve_and_synthesize`：**只调用一次**
  `_synthesize_stream`；注入重复 id 的桩 → 验证去重；合成 query 是**原始问题**而非子查询
- synthesize + 单子查询 → 退化单轮
- synthesize + 合并池空 → 空命中文案
- decomposer 空 → 现有单轮降级（回归）
- 并行检索：桩检索记录调用，验证扇出为并发发起（如以 gather 收集、调用计数正确）

**eval 端到端**：延后，不在本次开发范围。日后跑 `全开+hybrid+rerank`，确认综合型从
error/碎片 → 连贯整合答案、罗列型行为不变、faithfulness/answer_relevancy/
factual_correctness 在综合型上回升。
