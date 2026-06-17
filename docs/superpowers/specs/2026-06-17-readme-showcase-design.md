# README 展示页设计：LexRAG（评测驱动迭代为亮点）

**日期**：2026-06-17
**仓库**：github.com/Pama0/LexRAG（根目录当前无 README.md）

## 背景与目标

项目根缺 README，GitHub 首页空白。目标是写一份**中文为主、完整的项目 README**，把"评测驱动改进闭环"（跑 ablation → 从数据发现缺陷 → 定位根因 → 修 → 再验证）作为**加星亮点**突出，面向国内招聘/同行读者展示工程深度。

素材已就绪：`docs/ARCHITECTURE.md`（产品愿景+分层）、`docs/EVAL_OVERVIEW.md`（评测体系全景）、本会话 ablation 结果 `eval/results/compare_20260617_165829.md`、out_of_scope 的 spec/plan。README 提供入口与亮点叙事，深挖链到这些文档。

## 语言与范围（已定）

- 语言：中文为主。
- 范围：完整 README（定位/架构/快速开始/路线图/结构），评测章节加星。
- 数据新鲜度：**方案 A** —— 直接复用现有 ablation 表（out_of_scope 引入前跑的），表下加注脚说明，out_of_scope 作为"随后一次迭代"的 case study，用 live smoke 7/7 收尾。

## README 结构（`README.md`）

```
# LexRAG — 技术书籍知识库助手（RAG 问答 + 评测驱动迭代）
> 一句话定位：上传技术书籍 PDF，做可溯源的 RAG 问答；配套量化评测体系驱动迭代。

## 这是什么
## 架构概览
## ⭐ 评测驱动迭代
## 快速开始
## 技术栈
## 路线图
## 项目结构
```

### 1. 这是什么
3–5 句：技术书籍知识库助手（PDF 入库 + RAG 问答）；当前已实现 QA，规划 study_plan / life_plan。点明差异化：**不止"能问答"，而是有一套量化评测体系，能从评测数据反推系统缺陷并修复**。

### 2. 架构概览
- 一段话：顶层 IntentRouter（净化+意图分类）→ QA capability（probe→难度分类→分支检索合成）。
- 核心原则一句话：**按任务可预测性给每个能力配 workflow 或 agent，二者可组合**（取自 ARCHITECTURE §2）。
- 精简分层文本图（Layer0 检索/记忆 → Router → capability），不照搬 ARCHITECTURE 全图。
- 链 `docs/ARCHITECTURE.md` 看完整愿景与落地路径。

### 3. ⭐ 评测驱动迭代（亮点章节）
- **评测体系一句话**：用 ragas 5 指标（faithfulness / answer_relevancy / context_precision / context_recall / factual_correctness）+ 自定义分类准确率，对"决策路由 RAG"做 baseline vs 变体（ablation）对比；被测 LLM 与 judge LLM 解耦。
- **ablation delta 表**：贴 `eval/results/compare_20260617_165829.md` 的 7 行表（baseline → +probe → +probe+split → 全开 → 全开+rerank → 全开+hybrid → 全开+hybrid+rerank）。表下注脚：「此表为 out_of_scope 引入前的决策对比快照；分类准确率列基于当时的分类体系。」一句话读表结论：**检索侧（rerank/hybrid）增益明确（context_precision 0.31→0.55、faithfulness 升至 0.89）；probe/split 等决策链对分类准确率为负贡献，已定位为下一步排查对象**。
- **Case study：out_of_scope（评测如何发现并修掉一个真实缺陷）**，5 步叙事：
  1. 现象：ablation 表显示分类准确率被决策链拖低；逐条看明细发现 PostgreSQL 这类问题被误判。
  2. 洞察：直觉"靠召回为空判库外"行不通——向量 ANN 只取最近邻、几乎从不返空，"召回为空"是死信号。
  3. 根因：分类体系把"信息不足（需反问）"和"库外（库里没有）"揉成一类，判据失效。
  4. 解决：新增独立 `out_of_scope` 分类，判据锚定**召回片段与问题主体实体的相关性**；命中如实告知、不反问不硬答；missing_info 收窄回本义；out_of_scope 最高优先级解决"既不足又库外"的边界。
  5. 验证：真实 LLM live smoke **7/7**（PostgreSQL/MongoDB/Oracle/Cassandra → out_of_scope，控制项不回归），单测 159 passed。链到 `docs/superpowers/specs/2026-06-17-out-of-scope-classification-design.md`。
- **诚实标注**（小字/引用块）：golden 仅 23 条（小样本、定性为主，非统计显著）；judge 为 LLM 自评（DeepSeek，LLM-as-judge 已知局限）；ablation 为单次运行，LLM 有方差，多次平均是后续方向。
- 链 `docs/EVAL_OVERVIEW.md` 看评测体系全景（数据集生成方法学、指标字段映射、脚本地图）。

### 4. 快速开始
取自 CLAUDE.md + EVAL_OVERVIEW：
```bash
# 环境：Python 3.12+，虚拟环境 .venv
.venv\Scripts\activate                 # PowerShell
pip install -r requirements.txt
# .env 配置 DEEPSEEK_API_KEY（主系统 + 评测 judge 共用）

python main.py                                   # CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务

# 评测（需 DEEPSEEK_API_KEY）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 5   # 冒烟
python -m eval.harness.compare --testset eval/dataset/golden.jsonl             # 全量 ablation
```
注明：从项目根目录运行（模块导入要求）。

### 5. 技术栈
LlamaIndex（workflow 编排）、Chroma（向量库）、DeepSeek（`OpenAILike`，关 thinking）、ragas（评测）、rank-bm25 + jieba（hybrid 检索）、bge-reranker（重排）、FastAPI（Web）。

### 6. 路线图
取自 ARCHITECTURE §1：QA 问答（✅ 已实现）→ 学习计划（workflow）→ 人生/决策支持（agent + 用户长期记忆，可溯源到书的观点）→ 进度复盘。

### 7. 项目结构
简表：`api/`(Web) → `core/`(领域：agent/workflow/retrieval/rag) → `configs/`；`eval/`(评测体系)；`legacy/`(冻结的早期法条 RAG)。注明分层方向单向，守卫 `scripts/check_layering.py`。

## 附带一致性修复（纳入范围）

README 链向 `docs/EVAL_OVERVIEW.md`，但该文档现仍与已合入 master 的 out_of_scope 改动矛盾，需同步：
- §2 category 表：`missing_info` 行含义改为"信息不足/指代不明"，新增 `out_of_scope` 行（库外，如实告知）。
- §4 golden 组成表：`missing_info` 8（含库外 4）→ `missing_info` 4 + `out_of_scope` 4。
- §8 已知问题 #1（库外误判 retrievable）：标为**已修**，一句话指向 out_of_scope 方案。
- §9 待办：勾掉"修库外 bug"项。
- 仅改这些事实陈述，不重写文档其余部分。

## 不做（YAGNI）

- 不写英文版/双语；不堆 CI badge / shields；不重跑 ablation（用现有表 + 注脚）；不改 ARCHITECTURE 愿景内容；不动评测脚本。

## 验收

- 根目录有 `README.md`，中文，含上述 7 章节，GitHub 首页可读。
- ablation 表与 case study 叙事一致（注脚说明时间线）。
- 诚实标注三点齐全。
- 所有内链路径有效（ARCHITECTURE / EVAL_OVERVIEW / out-of-scope spec）。
- EVAL_OVERVIEW 与 README 不再自相矛盾（库外 bug 状态一致）。
- 快速开始命令与 CLAUDE.md 一致、可照做。
