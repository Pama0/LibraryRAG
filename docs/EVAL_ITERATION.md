# 评测驱动迭代

> LibraryRAG 不止"能问答"，还带一套量化评测体系：能从评测数据反推系统缺陷、定位根因、修复并再验证。本文记录这条迭代链路的真实过程；评测体系本身的设计（被测什么、指标、脚本、怎么跑）见 [docs/EVAL_OVERVIEW.md](EVAL_OVERVIEW.md)。

用 **ragas 的 5 个指标**（faithfulness / answer_relevancy / context_precision / context_recall / factual_correctness）+ **自定义分类准确率**，对"决策路由 RAG"做 **baseline vs 变体（ablation）** 对比，量化每个决策（probe / split / rerank / hybrid 等）的增益。被测 LLM 与评测 judge LLM 解耦，互不污染。

## 里程碑记录
项目会根据评测数据，发现问题并修改问题，以及反哺补全完善评测数据集，这里记录下一些里程碑式的测评结果，并记录根据此次测评，对项目做出了哪些改动

## Ablation 对比表

| 配置 | 分类准确率 | context_precision | context_recall | factual_correctness | faithfulness | answer_relevancy |
|---|---|---|---|---|---|---|
| baseline(全单轮) | 0.70 | 0.36 | 0.62 | 0.47 | 0.68 | 0.60 |
| +probe | 0.61 (-0.09) | 0.36 (+0.00) | 0.58 (-0.04) | 0.44 (-0.02) | 0.71 (+0.03) | 0.59 (-0.01) |
| +probe+split | 0.65 (-0.04) | 0.35 (-0.01) | 0.64 (+0.02) | 0.41 (-0.06) | 0.74 (+0.06) | 0.55 (-0.05) |
| 全开 | 0.65 (-0.04) | 0.31 (-0.05) | 0.79 (+0.17) | 0.43 (-0.04) | 0.80 (+0.11) | 0.58 (-0.02) |
| 全开+rerank | 0.65 (-0.04) | 0.55 (+0.19) | 0.82 (+0.20) | 0.45 (-0.02) | 0.79 (+0.10) | 0.59 (-0.01) |
| 全开+hybrid | 0.65 (-0.04) | 0.31 (-0.05) | 0.84 (+0.22) | 0.40 (-0.06) | 0.79 (+0.10) | 0.55 (-0.05) |
| 全开+hybrid+rerank | 0.57 (-0.13) | 0.52 (+0.16) | 0.78 (+0.15) | 0.51 (+0.05) | 0.89 (+0.20) | 0.53 (-0.07) |

**读表结论**：检索侧增益最明确——rerank 把 context_precision 从 0.31 拉到 0.55，hybrid 把 context_recall 顶到 0.84，全开+hybrid+rerank 的 faithfulness 达 0.89、factual_correctness 0.51（均为全场最高）。而 probe/split 等决策链对**分类准确率**为负贡献，由此定位到下一个排查对象——见下面的 case study。

1. 观察：分类准确率偏低，意味着路由分类频繁分配错误路由。观察测评实际结果表，发现分类体系把"信息不足（该反问澄清）"和"库外（库里根本没有）"揉成了一类（`missing_info`）
   行为：路由分类新增`out_of_scope` 分类，以应对超出知识库的问答
2. 观察：全开+hybrid+rerank的分类准确率比其他低，经过排查，是因为有的路径加上混合检索和重排导致了超时
3. 行为：将子问题查询步骤改为并行，并优化子问题查询的路径，分为子问题结果罗列和子问题结果合成路径

### 里程碑：agent 自主规划路线的答案相关性归因（2026-06-19）

把第二被测系统 `agent(自主规划)`（绕过决策路由、让 AutoAgent 自主多轮检索）与 workflow 同台对比，首张表上 agent 的 answer_relevancy=0.60、faithfulness=0.56，**双双低于 baseline（0.76 / 0.60）**，且时延 ~47s/条、token ~12615/条（约 workflow 的 10×/100×）。直觉是"agent 答得更差"，但逐条归因发现不是。

**观察**：把 agent 的 answer_relevancy 按金标准类别拆开——

| 子集 | answer_relevancy | 条数 |
|---|---|---|
| 全部已答（=报表 0.60） | 0.603 | 20 |
| 真该答的（retrievable/ambiguous/split/other） | **0.742** | 14 |
| 该拒答却硬答的（missing_info/out_of_scope） | **0.279** | 6 |

agent 在**真该回答**的题上 AR=0.74，与 baseline 0.76 基本持平；0.60 的低分几乎全部来自 6 条"库外/信息不足"题。

**根因**：agent 没有路由层，对这两类**不会拒答**——

- 库外题（PostgreSQL MVCC）→ 检索不到却**编造**了一段答案（faithfulness=0.0）；
- 其余库外题（Mongo/Oracle/Cassandra）→ 软拒"当前知识库只有这些书"，但 ragas 判其与问题不相关 → AR=0.0，且被标 `answered` 照常计入；
- 信息不足题 → 瞎猜（AR 0.38/0.48）。

对照 workflow：这 8 类题被短路成 `empty`，ragas **不计分**。所以"agent 不如 baseline"一半是**口径差异**（两者 AR 不在同一批题上平均），一半是**缺护栏的真实缺陷**（库外幻觉）。

**行为**（两处改动）：

1. **评测口径对齐**：`score_row` 对金标准为 `missing_info`/`out_of_scope` 的条目，把 5 个 ragas 指标统一归 null（对所有被测系统一致），拒答是否正确改由分类准确率衡量。这两类的答案质量本就不该用 ragas 打分。
2. **补 agent 护栏**：在 `AUTO_AGENT_SYSTEM_PROMPT` 加"检索之后三选一收场"——召回与问题主体技术实体不相关→如实告知库外（不编造）；召回到相关主题但指代不明→反问澄清；否则正常作答。判断只在检索之后、基于召回结果，不在检索前凭"不认识词"反问。

**修正后对比表**（口径对齐后，agent 答案质量回到真实画像）：

| 配置 | 分类准确率 | context_precision | context_recall | factual_correctness | faithfulness | answer_relevancy | 时延(s/条) | tokens/条 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.87 | 0.36 | 0.59 | 0.47 | 0.60 | 0.76 | 3.94 | 123.52 |
| all | 0.82 | 0.52 | 0.75 | 0.52 | 0.86 | 0.79 | 27.19 | — |
| agent(自主规划) | — | 0.53 | 0.82 | 0.48 | 0.76 | 0.74 | 46.92 | 12615.13 |

**读表结论**：口径对齐后，agent 的答案质量与 workflow 持平（AR 0.74、faithfulness 0.76），它真正的劣势是**成本**——时延高一个数量级、token 高两个数量级。这正向支撑项目论点：决策路由的价值不在"答得更好"，而在"该不该答、答不出时怎么收场"——护栏让系统在 35% 的边界题上不翻车，而裸 agent 在这些题上既贵又会幻觉。

> 遗留：归 null 后，agent 在库外题上的幻觉信号（faithfulness=0）也从指标里消失了。要保留这个信号，后续可加一个"拒答正确率"（该拒答的题里系统是否真的没硬答）——见 [docs/EVAL_OVERVIEW.md](EVAL_OVERVIEW.md) 待办。

## Case study：评测如何发现并修掉一个真实缺陷（out_of_scope）

1. **现象**：ablation 表显示分类准确率被决策链拖低；逐条看明细，发现「PostgreSQL 的 MVCC 是怎么实现的？」这类问题被误判。
2. **洞察**：直觉是"靠探测召回为空判断库外问题"，但这行不通——向量检索（ANN）只取最近邻、几乎从不返空，"召回为空"是个永不触发的死信号。库外问题的真实表现是"召回了 5 段最近邻、但内容全不相关"。
3. **根因**：分类体系把"信息不足（该反问澄清）"和"库外（库里根本没有）"揉成了一类（`missing_info`），判据失效，库外问题要么被误判可检索、要么被无意义地反问。
4. **解决**：新增独立的 `out_of_scope` 分类，判据锚定**召回片段与问题主体实体的相关性**（而非数量/空否）；命中时如实告知"知识库里暂无相关内容"，不反问、不硬答；`missing_info` 收窄回本义（信息不足才反问）；`out_of_scope` 最高优先级，解决"既信息不足又库外"的边界。
5. **验证**：真实 LLM live smoke **7/7**（PostgreSQL / MongoDB / Oracle / Cassandra → out_of_scope，且控制项 retrievable / missing_info 不回归），单元测试 159 passed。

> 设计与裁决细节见 [out_of_scope 设计文档](superpowers/specs/2026-06-17-out-of-scope-classification-design.md)。

## 诚实标注

- 金标准集 golden 仅 **23 条**，属小样本、定性为主，**非统计显著**。
- 评测 judge 是 **LLM 自评**（DeepSeek），存在 LLM-as-judge 的已知局限。
- ablation 为**单次运行**，LLM 输出有方差；多次平均是后续方向。

## 怎么复跑

```bash
# 评测（需 DEEPSEEK_API_KEY）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 5   # 冒烟，确认链路通
python -m eval.harness.compare --testset eval/dataset/golden.jsonl             # 全量 ablation（默认落盘 eval/results/）
```

> 评测体系全景（数据集生成方法学、指标字段映射、脚本地图、已知问题）见 [docs/EVAL_OVERVIEW.md](EVAL_OVERVIEW.md)。
