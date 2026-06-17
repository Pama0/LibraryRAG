# LexRAG — 技术书籍知识库助手（RAG 问答 + 评测驱动迭代）

> 上传技术书籍 PDF，做可溯源的 RAG 问答；并配套一套量化评测体系，用数据驱动系统迭代。

## 这是什么

LexRAG 是一个技术书籍 / 文档的 AI 知识库助手：把书籍 PDF 解析入库，基于检索增强生成（RAG）回答书里的具体问题。当前已实现文档问答（QA），并按"从受限检索到开放推理"的能力光谱规划了学习计划、人生/决策支持等后续能力。

它和一般 RAG demo 的区别在于：**不止"能问答"，而是带一套量化评测体系，能从评测数据反推出系统缺陷、定位根因、修复并再验证**——下面的「评测驱动迭代」就是一个完整的真实例子。

## 架构概览

顶层是一个编排器：**IntentRouter**（净化 query：规范化 + 指代消解 → 意图分类）→ 分发到对应**能力（capability）**。QA 能力内部再做：探测检索（probe）→ 难度分类 → 按类别走不同分支（单轮检索 / 拆解汇总 / 归纳维度 / 反问澄清 / 高难度 agent）。

核心原则：**按任务可预测性给每个能力配 workflow 或 agent，二者是可组合的积木，不是二选一**——高可预测（步骤已知）用 workflow 求确定性与可观测；低可预测（路径需模型自决）用 agent。

```
Layer 0  检索 & 记忆服务（横切，注入各层）：Chroma 向量库 + 可插拔 Retriever/Reranker
Layer 1  IntentRouter：净化 + 意图分类 + 确定性分发
Layer 2  能力层：QA(workflow) / StudyPlan(workflow，规划中) / LifePlan(agent，规划中)
```

> 完整产品愿景、分层依据与落地路径见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## ⭐ 评测驱动迭代

用 **ragas 的 5 个指标**（faithfulness / answer_relevancy / context_precision / context_recall / factual_correctness）+ **自定义分类准确率**，对"决策路由 RAG"做 **baseline vs 变体（ablation）** 对比，量化每个决策（probe / split / rerank / hybrid 等）的增益。被测 LLM 与评测 judge LLM 解耦，互不污染。

### Ablation 对比表

| 配置 | 分类准确率 | context_precision | context_recall | factual_correctness | faithfulness | answer_relevancy |
|---|---|---|---|---|---|---|
| baseline(全单轮) | 0.70 | 0.36 | 0.62 | 0.47 | 0.68 | 0.60 |
| +probe | 0.61 (-0.09) | 0.36 (+0.00) | 0.58 (-0.04) | 0.44 (-0.02) | 0.71 (+0.03) | 0.59 (-0.01) |
| +probe+split | 0.65 (-0.04) | 0.35 (-0.01) | 0.64 (+0.02) | 0.41 (-0.06) | 0.74 (+0.06) | 0.55 (-0.05) |
| 全开 | 0.65 (-0.04) | 0.31 (-0.05) | 0.79 (+0.17) | 0.43 (-0.04) | 0.80 (+0.11) | 0.58 (-0.02) |
| 全开+rerank | 0.65 (-0.04) | 0.55 (+0.19) | 0.82 (+0.20) | 0.45 (-0.02) | 0.79 (+0.10) | 0.59 (-0.01) |
| 全开+hybrid | 0.65 (-0.04) | 0.31 (-0.05) | 0.84 (+0.22) | 0.40 (-0.06) | 0.79 (+0.10) | 0.55 (-0.05) |
| 全开+hybrid+rerank | 0.57 (-0.13) | 0.52 (+0.16) | 0.78 (+0.15) | 0.51 (+0.05) | 0.89 (+0.20) | 0.53 (-0.07) |

> 注：此表为 `out_of_scope` 分类引入**之前**的决策对比快照（分类准确率列基于当时的分类体系）。

**读表结论**：检索侧增益最明确——rerank 把 context_precision 从 0.31 拉到 0.55，hybrid 把 context_recall 顶到 0.84，全开+hybrid+rerank 的 faithfulness 达 0.89、factual_correctness 0.51（均为全场最高）。而 probe/split 等决策链对**分类准确率**为负贡献，由此定位到下一个排查对象——见下面的 case study。

### Case study：评测如何发现并修掉一个真实缺陷（out_of_scope）

1. **现象**：ablation 表显示分类准确率被决策链拖低；逐条看明细，发现「PostgreSQL 的 MVCC 是怎么实现的？」这类问题被误判。
2. **洞察**：直觉是"靠探测召回为空判断库外问题"，但这行不通——向量检索（ANN）只取最近邻、几乎从不返空，"召回为空"是个永不触发的死信号。库外问题的真实表现是"召回了 5 段最近邻、但内容全不相关"。
3. **根因**：分类体系把"信息不足（该反问澄清）"和"库外（库里根本没有）"揉成了一类（`missing_info`），判据失效，库外问题要么被误判可检索、要么被无意义地反问。
4. **解决**：新增独立的 `out_of_scope` 分类，判据锚定**召回片段与问题主体实体的相关性**（而非数量/空否）；命中时如实告知"知识库里暂无相关内容"，不反问、不硬答；`missing_info` 收窄回本义（信息不足才反问）；`out_of_scope` 最高优先级，解决"既信息不足又库外"的边界。
5. **验证**：真实 LLM live smoke **7/7**（PostgreSQL / MongoDB / Oracle / Cassandra → out_of_scope，且控制项 retrievable / missing_info 不回归），单元测试 159 passed。

> 设计与裁决细节见 [out_of_scope 设计文档](docs/superpowers/specs/2026-06-17-out-of-scope-classification-design.md)。

### 诚实标注

- 金标准集 golden 仅 **23 条**，属小样本、定性为主，**非统计显著**。
- 评测 judge 是 **LLM 自评**（DeepSeek），存在 LLM-as-judge 的已知局限。
- ablation 为**单次运行**，LLM 输出有方差；多次平均是后续方向。

> 评测体系全景（数据集生成方法学、指标字段映射、脚本地图、已知问题）见 [docs/EVAL_OVERVIEW.md](docs/EVAL_OVERVIEW.md)。

## 快速开始

环境：Python 3.12+，虚拟环境 `.venv`。**所有命令从项目根目录运行**（模块导入要求）。

```bash
# 激活虚拟环境（PowerShell）
.venv\Scripts\activate
pip install -r requirements.txt
# .env 配置 DEEPSEEK_API_KEY（主系统与评测 judge 共用）

python main.py                                   # CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接）

# 评测（需 DEEPSEEK_API_KEY）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 5   # 冒烟，确认链路通
python -m eval.harness.compare --testset eval/dataset/golden.jsonl             # 全量 ablation（默认落盘 eval/results/）
```

## 技术栈

- **LlamaIndex** — workflow 编排与 RAG 基础设施
- **Chroma** — 向量数据库
- **DeepSeek** — 主 LLM（`OpenAILike` 接入，已关 thinking）
- **ragas** — 评测指标
- **rank-bm25 + jieba** — hybrid 检索的稀疏侧（中文分词、RRF 融合）
- **bge-reranker** — 交叉编码器重排（可插拔）
- **FastAPI** — Web 服务

## 路线图

能力从"受限检索"到"开放推理"是一条光谱：

1. **RAG 问答** —— 基于已入库书籍回答具体问题（✅ 已实现）
2. **学习计划** —— 按一本书的结构生成结构化学习计划（workflow，规划中）
3. **人生 / 决策支持** —— 融合书的观点 + 用户长期记忆做长期规划，**建议显式锚定到书的依据、可溯源**（agent，规划中）
4. **进度 / 复盘** —— 读取已生成的计划产物与记忆，更新回顾进度（规划中）

## 项目结构

```
api/        Web 服务（FastAPI）
core/       领域逻辑：agent / workflow / retrieval / rag
configs/    LLM / embedding 等配置
eval/       评测体系：harness（runner/指标/SUT）+ datagen + dataset + results
legacy/     冻结的早期法律条文 RAG（不保证可运行）
```

依赖方向单向：`api/` → `core/` → `configs/`，由 `python scripts/check_layering.py` 守卫。
