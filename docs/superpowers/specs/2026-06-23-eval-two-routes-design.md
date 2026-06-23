# eval 收敛为「workflow vs agent」两路线 — 设计

日期：2026-06-23

## 背景

`DocQueryWorkflow` 重构后 step 图变为 `start → clean → split → qa_branch → combine → finalize`：
`route` 步骤删除、所有子问题一律汇入 QA、`combine` 恒产文本（含拒答也走散文）。
现有 eval harness 是按旧门口路由设计的，已有三处与新工作流脱节：

1. `compare.py` 的 ablation flag 矩阵（baseline / +probe / 全开 / +hybrid / +rerank / all）
   多数变体在新编排里无对应分支，失去意义。
2. `map_doc_result` 的 outcome 语义：拒答（missing_info / out_of_scope）现经 combine
   产出散文文本 + 无 source nodes，旧的 clarify/拒答区分丢失。
3. 分类准确率指标：`golden.jsonl` 用旧标签（ambiguous / retrievable / pending_split /
   other / missing_info / out_of_scope），新工作流 `qa_meta["category"]` 产
   explain / compare / simple / complex / multi / out_of_scope，二者不对齐；且 agent 路线
   压根不产 category。

## 目标

把 eval 收敛为**两个 SUT 路线**对比，砍掉失效的 flag 矩阵、分类准确率指标与单系统 runner：

- **workflow** —— `DocQueryWorkflowSystem` 默认 flag（已确认 = `DocQueryService` 生产配置：
  probe + agent on、vector 检索、explain 走 hybrid）。
- **agent** —— `AgentSystem`（`AutoAgent`，自带 hybrid + rerank 默认）。

保留指标：5 个 ragas 答案质量（context_precision / context_recall / factual_correctness /
faithfulness / answer_relevancy）+ 成本（时延 s/条、tokens/条）。

## 非目标

- 不重做指标体系、不重标 golden 数据集。
- 不动 ragas judge 配置（`config.py` / `metrics.py` / `meter.py`）。
- 不在工作流侧新增任何 metadata 出口。

## 改动分文件

### `eval/harness/run_eval.py` —— 删除整个文件

单系统 runner 不再需要（唯一入口收敛到 `compare.py`）。删除前把 `compare.py` 依赖的纯函数
搬入 `compare.py`：

- 搬迁：`load_testset` / `_row_to_dict` / `score_row` / `aggregate`。
- 丢弃：`SINGLE_SYSTEM_LABEL` / `build_single_report`（单系统专用）。

### `eval/harness/compare.py`（成为唯一 runner）

- 纳入从 `run_eval.py` 搬来的 `load_testset` / `_row_to_dict` / `score_row` / `aggregate`
  （不再 `from eval.harness.run_eval import ...`）。
- `VARIANTS` 矩阵整体替换为两条命名路线：
  - `"workflow"` → `DocQueryWorkflowSystem`（无 flags，= 生产默认）。
  - `"agent"` → 哨兵（`build_sut` 里分流到 `AgentSystem`）。
  - 删除 baseline / +probe / 全开 / +hybrid / +rerank / all 全部组合与 `AGENT_VARIANT` 间接层。
- `build_sut`：`"workflow"` → `DocQueryWorkflowSystem(index_manager, llm)`；
  `"agent"` → `AgentSystem(index_manager, llm)`。
- 默认 `--variants` = 两条都跑；默认 `--baseline = "workflow"`（delta 列 = agent 相对 workflow）。
- 并发 / 串行 / 进度 / 落盘逻辑不动。

#### `score_row`（搬入 compare.py 后按此调整）

- 删 SUT `category` 相关字段输出。
- **保留**对 golden 拒答行（`expected_category ∈ {missing_info, out_of_scope}`，共 8/23 条）
  跳过 ragas 的闸门：拒答题以「应拒答」为 reference，ragas 打分无意义，跳过避免污染质量均值。
- 即 `out.outcome != "answered"` 或 `expected_category ∈ REFUSE_CATEGORIES` 时只返回 base，不打分。

#### `aggregate`（搬入 compare.py 后按此调整）

- 删 `classification` 块（cls_total / cls_correct / accuracy）与 `category_distribution`。
- 保留 `outcome_distribution` / `metric_means` / `cost`。

### `eval/harness/sut.py`

- `RagOutput`：删 `category` 字段。
- `map_doc_result`：不再读 `metadata.category`；outcome 映射保持 answered / empty / error
  （combine 恒产文本：有 nodes → answered，无 nodes（converse / 拒答）→ empty，异常 → error）。
- `map_agent_result`：去掉 category 参数位（原本恒空）。
- `DocQueryWorkflowSystem`：保留 `flags` 入参（默认 `{}` = 生产配置），两路线对比只用默认。
- `AgentSystem`：不变。

### `eval/harness/report.py`

- `_COLS`：删「分类准确率」列；保留 5 ragas + 2 成本列。
- `_DETAIL_COLS`：删 `category` / `match`；保留 `expected_category`（golden 标签，供人工查阅）。
- `write_detail_csv`：去掉 `match` 计算。

### 不动

`metrics.py`、`config.py`、`meter.py`、`golden.jsonl` 及其标签。

## 测试改动（与实现同步）

现有单测深度耦合 classification / category / run_eval，需随实现调整：

- `tests/test_eval_run.py` —— 删除（`run_eval` 模块已删）。其中仍有效的纯函数测试迁入
  `tests/test_eval_compare.py`（因 `score_row` / `aggregate` / `_row_to_dict` 搬入 compare.py）：
  - 保留并迁移：`aggregate` 的 means/cost 块、`score_row` 的 latency/tokens/meter reset、
    非 answered 跳过打分、**拒答行（out_of_scope / missing_info）跳过打分**。
  - 删除：所有 `classification` 准确率/分布相关用例、`build_single_report` 用例、
    `score_row carries category`（SUT category 已删）。
- `tests/test_eval_sut.py` —— 删 `out.category` 断言（`RagOutput.category` 已删）；
  保留 outcome（answered/empty/error）映射断言。
- `tests/test_eval_report.py` —— 删「分类准确率」列断言、`write_detail_csv` 的 `match` 列断言；
  `default_result_paths` 测试改用默认 `compare` 前缀（`run_eval` 入口已无）。
- `tests/test_eval_compare.py` —— 样例 report 去掉 `classification` 块；
  `build_sut` 测试覆盖两路线（`"workflow"` → `DocQueryWorkflowSystem`，`"agent"` → `AgentSystem`）。

### datagen 文案

`merge_golden.py` / `generate_testset.py` 注释里「供 run_eval 使用 / 读作」改为指向 compare
（仅注释措辞，无逻辑改动）。

## 验证

- smoke：`python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 2`
  （两路线各跑 2 条，出对比表 + detail.csv）。
- 分层守卫：`python scripts/check_layering.py`（core 不依赖 api）。
- 全量集成 smoke：去掉 `--limit` 跑全 23 条，确认无引用 `run_eval` 的残留导入报错。
