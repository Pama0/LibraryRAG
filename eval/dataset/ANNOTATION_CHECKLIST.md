# 金标准集标注 Checklist（逐条可操作）

目标：产出 `golden.jsonl`，30~50 条，每类 ≥3、难度分层、含易误判 case。
配合 `README.md`（字段定义 + category 准则）使用。

---

## 阶段 0 · 准备（做一次）

- [ ] 看清书库里有哪些书：启动服务问"有哪些书"，或
      `python -c "from core.rag.data_loader import RAGIndexManager; import os; m=RAGIndexManager(persist_dir=os.path.join(os.getcwd(),'chroma_db')); d=m.chroma_collection.get(include=['metadatas']); print({(x or {}).get('book_title') for x in d['metadatas']})"`
- [ ] 新建空文件 `eval/dataset/golden.jsonl`，准备逐行写入。
- [ ] 记住配额：retrievable / pending_split / ambiguous / missing_info / other 各 ≥3 条。

## 阶段 1 · 每条 query 的标注（6 字段，逐条做）

对每一条，依次填：

- [ ] **`user_input`**：写**用户真会问的原话**。故意制造措辞变体（口语「给我讲明白X」、错别字、长/短句、Web 搜索式短语），别都写成标准书面问句——这样才考得出 router 净化能力。
- [ ] **`category`**：用下面【阶段 2 判定树】定，**只填一个**。
- [ ] **`scope`**：全库填 `null`；只针对某本书填 `["书名"]`（Phase 1 多数填 `null`）。
- [ ] **`reference`**：标准答案。**从书里摘录/总结**，别用你自己的知识编。missing_info 类**留空字符串 `""`**。
- [ ] **`reference_contexts`**：支撑答案的**原文片段**，从书里复制（可多段）。missing_info 类**留空 `[]`**。
- [ ] 写成一行 JSON，追加到 `golden.jsonl`。

## 阶段 2 · category 判定树（按顺序问，命中即定）

```
Q1 这问题库里有相关内容吗？（拿不准就真去检索探一下）
   否 → missing_info     （reference="" , contexts=[]）
   是 ↓
Q2 问题缺评判维度/角度吗？（"X好吗""X和Y哪个好""怎么样"）
   是 → ambiguous
   否 ↓
Q3 是【单一概念】、单轮能集中命中吗？（"X是什么""X怎么实现"，X 是一个概念）
   是 → retrievable
   否 ↓
Q4 X 是【大主题/多实体】、要罗列子项或跨多章节吗？（"讲讲X""A和B的区别"）
   是 → pending_split
   否 ↓
Q5 要【跨主题综合/多步推理/开放权衡】吗？（"综合评价X""结合多个概念设计一套方案"）
   是 → other
   否 → retrievable（兜底）
```

**⚠️ 最容易标错的一条（务必照做）**：
> 你**认不认识 query 里的词，不影响 category**。`给我讲明白openclaw`——openclaw 是库里一个**单一概念**，所以是 **retrievable**，不是 other、不是 missing_info。**绝不能因为"这词我没听过 / 它好像很复杂"就标 other 或 missing_info。** 这正是要量化修复的误判，金标准必须标对。

## 阶段 3 · 覆盖度检查（标到一定量后核对）

- [ ] `retrievable` ≥3，其中**至少 1 条易误判**（如「给我讲明白<某专名>」「<冷门概念>是什么」）。
- [ ] `pending_split` ≥3（如「讲讲MySQL」「聚簇索引和二级索引的区别」「怎么优化MySQL」）。
- [ ] `ambiguous` ≥3（如「Redis做缓存好吗」「MySQL大表查询慢怎么优化」）。
- [ ] `missing_info` ≥3（如「这个索引的应用场景」「上面说的那个怎么配」——指代不明、库里查无）。
- [ ] `other` ≥3（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
- [ ] **难度分层**：每类都有简单/中/难的样本，不要全是简单题。
- [ ] **措辞变体**：口语、错别字、长短句各有若干。

## 阶段 4 · 质量校验（提交前过一遍）

- [ ] 每条 `reference_contexts` 真的是**从书里复制**的，不是你编/总结的（ragas 召回指标靠它）。
- [ ] `missing_info` 条目的 `reference=""` 且 `contexts=[]`。
- [ ] 边界 case（retrievable vs pending_split 拿不准的）按【阶段 2】裁定，存疑的在本文件底部记一行备注。
- [ ] 每行是**合法单行 JSON**（可 `python -c "import json;[json.loads(l) for l in open('eval/dataset/golden.jsonl',encoding='utf-8') if l.strip()]"` 校验不报错）。
- [ ] 总数 30~50，五类配额都达标。

## 阶段 5 · 跑出对比表

- [ ] 先小样冒烟（`--limit 5`）确认链路通：
      `python -m eval.compare --testset eval/dataset/golden.jsonl --limit 5 --variants "baseline(全单轮)" "+probe"`
- [ ] 跑全量首张对比表（证明 probe-then-classify）：
      `python -m eval.compare --testset eval/dataset/golden.jsonl --variants "baseline(全单轮)" "+probe" --baseline "baseline(全单轮)"`
- [ ] 看**分类准确率**列：`+probe` 应高于 baseline；看 `other` 误判（结合 category_distribution）应下降——这就是 openclaw 修复的量化。
- [ ] 逐个加变体看其它决策：`"+probe+split"`、`"全开"`。
- [ ] 把对比表（Markdown）存进 `docs/`，作项目/简历证据。

---

## 备注（标注存疑记这里）

- （示例）「怎么优化MySQL」标 pending_split 还是 ambiguous？——按准则：缺维度→ambiguous，要罗列整片→pending_split。本集按 ____ 处理。
