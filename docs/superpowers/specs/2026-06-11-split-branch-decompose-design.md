# split_branch 拆解-检索-汇总设计（结构主 + 内容辅）

**日期:** 2026-06-11
**状态:** 已设计，待写实现计划
**关联:** [项目架构](../../ARCHITECTURE.md) · [intent-router + preprocess 拆分](2026-06-11-intent-router-and-preprocess-split-design.md)（本设计填实其中 `doc_workflow.split_branch` 的 TODO）

## 背景与要解决的矛盾

`core/workflow/doc_workflow.py` 的 `split_branch`（`pending_split` 分支）当前是占位：直接按整句检索，没有真正拆解。要把它做成"拆子问题 → 多路检索 → 汇总"。

**核心矛盾**：用 LLM 把 query 拆成子问题，默认了 LLM **已经懂这个领域**。
- "讲讲MySQL索引" → LLM 懂 → 能拆「聚簇/二级/覆盖索引…」。✅
- "讲讲openclaw" / "openclaw 的工具系统" → LLM 训练时没见过 → 拒答或瞎拆通用子问题。❌

**结论**：对未知概念，拆解必须**靠检索接地**——先检索了解概念是什么、文档里真实有哪些子项，再基于召回内容拆，而不是凭 LLM 记忆。LLM 在此是"对召回材料做归纳的推理器"，不是知识源。

## 适用范围（三类问法，同一条流水线）

| 问法 | "概念"粒度 | 骨架（子问题）来源 |
|---|---|---|
| openclaw 这本书讲了什么 | 整本书 | 顶层章节 |
| openclaw 的工具系统（章节内中层概念，对应~10 实体） | 某小节 | 该小节子树 + 正文枚举 |
| openclaw 和 React 的区别 | 多实体对比 | 命中对比章节子树；子项=对比维度 |

三者差别只在"概念多大、骨架从哪长"，流程同为 **定位 → 建骨架 → 逐项检索 → 汇总**。对比类不单开分支：LLM 精炼那步按 query 意图决定子项形状（实体/小节 vs 对比维度）。

## 数据前提（已核实）

集合 `book_knowledge` 中 `openclaw_guide-v1.2.2`：519 chunk，`chapter` **100% 填充**，且为**层级编号**标题（`1.1` / `1.1.1` / `1.2` / `1.2.1`…）。存在少量噪声标题（如 `(messages/prompt + tool`，字号探测误判）。MySQL 书同为满填。
- 层级编号 → 可按前缀确定性地建树、取兄弟/子节点。
- quick 入库路径（`data_loader` 的 fast 流程）`chapter=""` → 无结构，需内容兜底。

## 决策：结构主 + 内容辅（模式3）

- **信结构**（章节标题做骨架）：完整（不依赖召回，章节表有几个子节就是几个）、确定、便宜、可溯源；但脆——依赖解析质量、抓不到"正文里列举而非小标题"的实体、quick 入库无结构。
- **信内容**（LLM 从召回 chunk 枚举）：不挑解析、能抓正文级实体、粒度自适应；但**被召回率封顶会漏实体**（拆解的头号大忌）、不确定。
- **取主次 = 结构主、内容辅**：结构当"完整性脊柱"（治内容法的漏），把章节子树标题 **+** 召回正文**一起**喂 LLM，由 LLM 去噪、补正文级实体、控粒度。结构缺失/太脏 → 自动倒向内容主导。

## 流水线四步

1. **定位**：`_retrieve_nodes(clean_query, book_titles)` 一轮宽召回 → 收集命中 chunk 的 `chapter` 集合。
   - 命中**聚于某前缀**（如多数落在 `3.2.*`）→ 中层概念，锁定该前缀子树。
   - 命中**散在多个顶层章节** → 整本书概览，骨架取顶层章节。
   - 命中分布本身既定位概念位置、又决定"在树的哪一层切骨架"。
2. **建骨架**（结构主内容辅）：把（① 定位到的章节子树标题列表 ② 召回 chunk 正文）喂 `QueryDecomposer` LLM → 产出 ≤N 个子查询（N=上限，默认 6；超出则归并/取重点）。无章节元数据 → 仅凭召回正文枚举。
3. **逐项检索**：每个子查询各做一轮 focused 检索（scope 硬约束限本书，沿用 `_make_filters`）。
4. **汇总（map-reduce）**：每子项各自合成一段（专属上下文 → 覆盖有保证），按骨架拼成结构化答案；reduce 尽量轻（拼接，必要时一句总起）。
   - **不选"并池一次合成"**：10×top-k≈50 chunk 一次喂 → context 压力 + lost-in-the-middle，会在合成端重新漏覆盖，自毁拆解价值。

## 组件划分（各自单一职责、可独立测试）

### `core/workflow/chapter_tree.py`（纯逻辑，不碰 LLM/网络）
- 输入：某书的 chapter 元数据列表（由调用方从 `index_manager.chroma_collection` 取）。
- 能力：
  - 把编号标题解析成树（按 `1.2.1` 形式的前缀）。
  - 取某前缀下的兄弟/子节点标题。
  - 从一组命中 chapter 推"主导前缀"（多数命中共享的最深公共前缀）；命中发散则返回空 → 信号为"取顶层"。
- 纯函数，给定元数据即可测，无需 chroma/LLM。

### `core/workflow/query_decompose.py`：`QueryDecomposer`（注入 LLM）
- 输入：`clean_query` + 子树标题列表 + 召回正文片段。
- 输出：子查询列表（≤N），去噪、Pydantic 校验；解析失败/空 → 返回空列表（由 split_branch 降级）。
- 与 `IntentRouter` / `QueryPreprocessor` 同构：`run(...)` 单入口，json_object + Pydantic，mock LLM 可测。
- prompt 要点：基于"给定的标题 + 正文"产出并列子项，禁止编造文档里没有的实体；按 query 意图（罗列/对比）决定子项形状；控制在 N 个内。

### `doc_workflow.split_branch`（编排 + 流式）
- 编排四步；复用 `_retrieve_nodes` / `_make_filters` / `_synthesize_stream`。
- 子项上限 N 作为 workflow 配置（默认 6）。
- 降级：`QueryDecomposer` 返回空 → 退化为单轮检索+合成（等同 `retrieve_branch`），绝不阻塞。

## 流式（前端零改动）

复用既有 SSE 词汇，但**不per-section发 tool_call**（每个新 tool_call 会把前一节答案挤进"思考步"并清空 content，割裂答案）。

- `RetrievalStartEvent(query=原问题)` → 前端 `tool_call`。
- 拆解 + 各路检索完成后发**一次** `RetrievalDoneEvent` → 前端 `tool_result`（翻入答案阶段）。
- 之后每个子项：先推一个标题 `AnswerDeltaEvent(delta="\n## {子项}\n")`，再推该节合成 token。
- 答案在单个 content 块内按节累积，保留打字效果与结构。

## 降级与边界

- LLM 拆解失败 / 空结果 → 单轮检索+合成兜底。
- 无章节元数据（quick 入库）→ 内容主导枚举。
- 对比的另一方既不在库、书中也未提及 → 诚实声明"基于文档给出在库一方的内容，另一方不在知识库"，不编造（守 grounding）。
- 子项上限 N 封顶，控成本/延迟（每子项一次合成调用）。

## 错误处理（沿用既有铁律）

- 结构化输出走 json_object + Pydantic 校验（DeepSeek 稳定端点，不依赖 strict schema）。
- 任一 LLM 步失败均降级，不阻塞用户。
- 中间产物（子查询、各节草稿）只走 workflow `Context` / 局部变量，**绝不写会话记忆**。

## 测试（TDD）

- `chapter_tree`：树构建 / 前缀兄弟·子节点 / 主导前缀推断 / 命中发散返回空（纯函数，构造元数据）。
- `QueryDecomposer`：mock LLM —— 子查询解析、上限裁剪、去噪、解析失败降级为空、prompt 含给定标题与正文。
- `split_branch`：stub `_retrieve_nodes` / `chapter_tree` / `QueryDecomposer` / `_synthesize_stream` —— 验证 定位→骨架→逐项→拼接 接线；空骨架降级；流式分节事件（一次 RetrievalDone + 每节标题 delta）。

## 非目标（本次）

- 不做 study_plan / 其他 capability。
- 不改 ambiguous（`assume_branch`）分支逻辑。
- 不引入跨书拆解（scope 仍限用户选定范围）。
- 不做子项并行检索的并发优化（先串行，真测出延迟瓶颈再说）。
