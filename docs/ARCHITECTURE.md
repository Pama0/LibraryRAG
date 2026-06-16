# 项目架构

**日期:** 2026-06-11
**状态:** 构想已定，分阶段落地
**关联:** [step1 拆分设计](superpowers/specs/2026-06-11-intent-router-and-preprocess-split-design.md)

> 本文记录产品愿景与系统级架构决策。具体某一层的实现细节见对应 spec。

## 1. 产品愿景

文档 / 书籍的 AI 助手。能力从"受限检索"到"开放推理"是一条光谱：

1. **RAG 问答** —— 基于已入库书籍/文档回答具体问题（当前已实现）。
2. **学习计划** —— 根据一本书/文档的结构，生成结构化学习计划。
3. **人生计划 / 决策支持** —— 融合书的内容与**思想**，结合**用户长期记忆**（目标、处境、价值观），帮用户做超出 RAG 范畴的长期规划或决策。
4. **进度 / 复盘** —— 读取已生成的计划产物与用户记忆，更新与回顾进度。

**差异化核心**：人生/决策建议必须**显式锚定到书的观点**（可溯源），否则会退化成脱离原书的通用 life coach，丢掉产品价值。

## 2. 核心原则：按可预测性给每个能力配控制结构

不是"整个系统 workflow 还是 agent"的二选一。判据：

| 任务可预测性 | 控制结构 | 理由 |
|---|---|---|
| 高（步骤已知、产出形态固定） | **Workflow** | 要确定性、可观测、可测试、grounding 可控 |
| 低（路径无法预定、要模型自己决定下一步） | **Agent** | 开放推理、多轮、试探、反问 |

**workflow 与 agent 是可组合的积木，不是竞品**：agent 可作 workflow 的一个 step；workflow 可作 agent 的一个 tool。系统顶层是个编排器（router），在"能力（capability）"这一层才落到具体形状。

## 3. 系统分层

```
┌──────────────────────────────────────────────────┐
│  Layer 0  Memory & Retrieval 服务（横切，注入所有层）  │
│   - 文档索引 (RAG，已有：Chroma + index_manager)      │
│   - 用户记忆（规划中）：profile/目标/价值观             │
│     + plan 产物 + 进度。结构化 DB + 可检索层           │
└──────────────────────────────────────────────────┘
              ↑ 被下面所有层按需读/写
┌──────────────────────────────────────────────────┐
│  Layer 1  Intent Router                            │
│   通用净化（规范化 + 指代消解）→ clean_query           │
│   意图分类（LLM）→ qa / study_plan / life_plan / ...  │
│   分发（确定性 dispatch）                             │
└──────────────────────────────────────────────────┘
              ↓ 路由到对应 capability
┌───────────────┬────────────────┬─────────────────┐
│ QA capability │ StudyPlan cap.  │ LifePlan cap.   │
│  = Workflow   │  = Workflow     │  = Agent        │
│ 降噪+难度分类  │ 拆解→排序→渲染   │ 开放推理 + 反问   │
│ →检索→合成    │ 产 plan 产物    │ 工具:RAG+用户记忆 │
└───────────────┴────────────────┴─────────────────┘
              ↓ 各能力收尾时回写
        Memory（会话 / profile / plan 产物 / 进度）
```

**为什么不是"一个大 agent 全包"**：会在受限任务（QA、出计划）上丢失确定性与可观测；高风险输出（人生建议）更难保证 grounding。
**为什么不是"一个大 workflow 全包"**：无法把开放式的人生决策预定义成固定步骤。

## 4. Memory：必须当独立服务做（地基）

Memory 是整个构想的地基，**最该提前定接口**。要点：

1. **分类型，别一锅**：
   - **会话记忆**（短期，per-session）—— 已有（`ChatMemoryBuffer`）。
   - **用户长期记忆**（跨会话：目标、处境、价值观、约束）—— life_plan 的命脉。
   - **Plan 产物是有状态的一等对象**（学习/人生计划有创建/修订/追踪生命周期），落 DB，能力通过工具操作它，**不塞进 chat buffer**。
   - 可选：用户笔记/反思，本身可 RAG 化。
2. **Memory 是被注入的服务，不归任何 workflow/agent 所有**。每个 capability 按需申请它要的**记忆切片**：QA 要会话上下文；life_plan 要 profile+目标+历史。
3. **两层记忆铁律**（已在 QA 落地，向全系统推广）：
   - **会话记忆**：只存「用户原话 + 最终答案」，供 Router 的指代消解读取。
   - **本轮工作态**：改写 query、分类、子问题、中间产物——只走 workflow `Context`，**绝不写进会话记忆**，否则下轮指代消解读到污染历史。
4. **隐私/可控**：存人生数据 → 一开始就设计「可更正、可删除、可分 scope」。

## 5. 当前状态 → 目标

| 资产 | 现状 | 在目标架构中的位置 |
|---|---|---|
| `core/workflow/query_preprocess.py` | step1：规范化+指代+降噪+难度分类（一次 LLM call） | **拆分**：通用净化上提 Router，降噪+难度留 QA。见 [拆分 spec](superpowers/specs/2026-06-11-intent-router-and-preprocess-split-design.md) |
| `core/workflow/doc_workflow.py` | 顶层编排骨架（preprocess→路由→分支 agent→finalize） | **QA capability** 的种子 + Router 雏形 |
| Chroma + `index_manager` | 文档索引 | Layer 0 检索服务 |
| `core/retrieval/rerank.py` | 可插拔 Reranker（bge 交叉编码器，注入式）：装配时按名字注入，不传=基线（直召 top_k），传入=过召回后重排截断；eval `VARIANTS` 以名字选择，ablation 量化增益 | Layer 0 检索后处理 |
| 用户记忆系统 | 未建 | Layer 0，规划中 |

## 6. 落地路径（增量，不一次造完）

1. **Router 雏形**：把 step1 升维——通用净化 + 意图分类（先只有 qa 一类也行，留好扩展位）。
2. **QA、study_plan 做成 workflow**，产出结构化 plan 产物落 DB。
3. **引入 life_plan agent**：第一个真正需要 agent 自主性 + 用户记忆的能力。
4. **Memory 服务接口先行**，随能力增量填实（v1 可以只是 chat buffer + 一张 user_profile 表，但接口要立住）。

## 7. 既有约束（沿用）

- **分层方向单向**：`api/`(Web) → `core/`(领域) → `configs/`。守卫：`python scripts/check_layering.py`。新增能力仍遵守。
- **工具在组装层创建、注入 Agent**（见 CLAUDE.md）。
- **LLM**：DeepSeek（`OpenAILike`，已关 thinking）。结构化输出走 json_object + Pydantic 校验，不依赖 OpenAI strict schema（DeepSeek 稳定端点不支持）。

## 8. 已知风险

- **life_plan 的 grounding**：书的"思想"比 RAG 内容模糊，须让书的依据在建议里显式可溯源。
- **plan 是有状态对象**：要 DB 化、有生命周期——这决定 memory 层 schema，事后难补。
- **隐私**：用户人生数据的可更正/删除/分 scope。
- **成本/延迟**：多层 LLM 调用（Router + 能力内）会叠加，需在真测出瓶颈后再做合并优化，避免过早耦合。
