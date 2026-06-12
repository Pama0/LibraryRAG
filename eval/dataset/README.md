# 评测测试集说明

## 文件

- `golden.seed.jsonl` — 金标准**种子**（每类 1~2 条示范格式，`reference`/`reference_contexts` 是 `<占位>`）。
- `golden.jsonl` — **完整金标准集**（30~50 条，需按本说明人工补全/校验后生成，用于决策对比评测）。
- `testset.draft.jsonl` / `testset.jsonl` — ragas TestsetGenerator 自动生成（Phase 2 扩样用）。

## 字段

| 字段 | 含义 |
|------|------|
| `user_input` | 用户问题（原话，可含口语/措辞变体） |
| `category` | **金标准应走的类**（分类准确率以此为准） |
| `scope` | book_titles 列表；`null` = 全库 |
| `reference` | 标准答案（factual_correctness / context_recall 用） |
| `reference_contexts` | 相关原文片段（来自你书库的真实内容） |

## category 标注准则（边界约定）

| 类 | 判据 |
|----|------|
| `retrievable` | 单一概念、单轮检索可集中命中。含「X是什么 / 讲明白X」**当 X 是单一概念**（即便 X 是冷门专名）。 |
| `pending_split` | X 是大主题、答案需罗列并列子项 / 横跨多章节（如「讲讲MySQL」「A和B的区别」）。 |
| `ambiguous` | 主题具体、能命中，但缺评判维度/角度（如「Redis做缓存好吗」）。 |
| `missing_info` | 缺检索必需限定，且**库里确实没有**（指代不明、查无此物）。 |
| `other` | 召回到内容、但需跨主题综合 / 多步推理 / 开放权衡。 |

## 必须覆盖

- **每类 ≥3 条**，难度分层（简单/中/难）。
- **「库里有但易误判」case**（如 `给我讲明白openclaw` 应判 retrievable，曾被误判 other）——这是量化 probe-then-classify 提升的关键样本。
- 措辞变体（口语、错别字、长短句）覆盖 router 净化能力。

## 标注流程

1. 从 `testset.draft.jsonl` 挑选 + 人工加 `category` 标注；或直接手写。
2. `reference`/`reference_contexts` 对照你书库里书的真实内容填写（决定 ragas 指标准确性）。
3. 边界 case 按上表准则裁定，存疑的记进本文件备注。
4. 汇总成 `golden.jsonl`，跑 `python -m eval.compare --testset eval/dataset/golden.jsonl ...`。
