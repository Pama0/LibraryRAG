# out_of_scope 分类 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `out_of_scope` 分类，把「库外问题」从 `missing_info`（反问）中解耦，命中时如实告知「库里没有」、不检索不反问。

**Architecture:** 判据锚定在「probe 召回相关性」单一轴上：召回片段与问题主题明显不相关 → `out_of_scope`（最优先）；召回相关但缺限定 → `missing_info`。改动在分类 prompt+schema（`query_preprocess.py`）与 workflow 分支（`doc_workflow.py`），外加数据集重标。评测侧 `aggregate` 用字符串比对枚举值，无需改。

**Tech Stack:** Python 3.12, llama-index workflow, Pydantic, pytest（async）。

## Global Constraints

- 所有 I/O 用 `async/await`；函数签名加类型注解（中文注释可接受）。
- 从项目根目录运行；子模块内用相对导入、根脚本用绝对导入。
- core 不依赖 api（守卫 `python scripts/check_layering.py`）。
- 解析失败一律降级 `retrievable`，绝不阻塞（现有不变）。
- `out_of_scope` 命中固定话术：**「知识库里暂无与该问题相关的内容。」**（不动态列书名）。
- 不加相似度阈值 cutoff；不加 out_of_scope 专属 ablation flag。
- pytest 运行：`.venv\Scripts\python.exe -m pytest`（async 测试已由现有 pytest 配置支持，仿照同目录测试写法）。

---

### Task 1: 分类 schema + prompt 增加 out_of_scope（`query_preprocess.py`）

**Files:**
- Modify: `core/workflow/query_preprocess.py`（prompt `_JUDGE_PROMPT` 第 26–79 行；`QueryJudgment.category` 第 100 行）
- Test: `tests/test_query_preprocess.py`

**Interfaces:**
- Consumes: 现有 `QueryPreprocessor.run(clean_query, retrieval_context="")` → `PreprocessResult`（不变签名）。
- Produces: `PreprocessResult.category` 现在可取值 `"out_of_scope"`；`QueryJudgment.category` 的 `Literal` 接受 `"out_of_scope"`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_query_preprocess.py` 末尾追加：

```python
async def test_run_classifies_out_of_scope():
    # 库外：问题清晰但探测召回片段与主题无关（库里没有该主题）
    llm = FakeLLM([
        '{"category": "out_of_scope", "rewritten_query": "PostgreSQL的MVCC是怎么实现的", "reason": "库外，召回片段均不相关"}'
    ])
    result = await _pp(llm).run("PostgreSQL的MVCC是怎么实现的")
    assert result.category == "out_of_scope"
    assert result.reason == "库外，召回片段均不相关"


async def test_run_accepts_out_of_scope_in_schema():
    # out_of_scope 必须在 Literal 枚举内，不能被 Pydantic 当非法值降级
    llm = FakeLLM([
        '{"category": "out_of_scope", "rewritten_query": "MongoDB分片"}'
    ])
    result = await _pp(llm).run("MongoDB分片")
    assert result.category == "out_of_scope"   # 未被降级成 retrievable
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_preprocess.py::test_run_classifies_out_of_scope tests/test_query_preprocess.py::test_run_accepts_out_of_scope_in_schema -v`
Expected: FAIL —— `out_of_scope` 不在 `Literal` 内，`model_validate_json` 拒绝 → 降级为 `retrievable`，断言 `category == "out_of_scope"` 失败。

- [ ] **Step 3: schema 加枚举值**

`core/workflow/query_preprocess.py` 第 100 行：

```python
    category: Literal["retrievable", "pending_split", "missing_info", "ambiguous", "other"]
```

改为：

```python
    category: Literal[
        "retrievable", "pending_split", "missing_info", "ambiguous", "other", "out_of_scope"
    ]
```

- [ ] **Step 4: prompt —— missing_info 收窄**

第 45–46 行当前为：

```
- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索；**且探测召回为空、或片段明显与问题无关**（知识库里确实没有相关内容）。多为指代不明且历史无从补全。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
```

改为：

```
- missing_info（信息不足）：**末尾【知识库探测召回】到了与问题相关的主题**，但缺了检索必需的关键限定/指代不明，补充后才能精确命中。多为指代不明。
  如「这个索引的应用场景是什么」——库里有索引内容，但"这个索引"指代不明（全文索引？B+树索引？其他？），补充后即可检索。
```

- [ ] **Step 5: prompt —— 新增 out_of_scope 类**

在 `other` 类定义块之后（第 63 行 `返回 {"category":"other"...}` 那行之后、第 65 行优先级段之前）插入：

```
- out_of_scope（库外）：**末尾【知识库探测召回】的片段与问题主题明显不相关**（即知识库里没有该主题的内容），无论问题是否完整、是否缺限定——一律判 out_of_scope。因为库里没有的内容，反问也补不出来。
  特征：召回片段讲的全是另一个主题。如「PostgreSQL的MVCC怎么实现」「MongoDB分片」「Oracle RAC」——本库召回到的都是别的主题（如 MySQL），与问题无关。
  返回 {"category":"out_of_scope","rewritten_query": "处理后的 query","reason": "库外原因，如'PostgreSQL 不在本库主题范围，召回片段均不相关'"}
```

- [ ] **Step 6: prompt —— 更新优先级段**

第 65 行当前为：

```
【不可以】归类的优先级：先判断信息是否不足(missing_info)，再判断是否角度不定(ambiguous)，再判断是否单纯并列罗列(pending_split)；若以上都不是、但问题需要跨主题综合/多步推理/开放权衡，则判 other（积极）；其余能单轮集中命中的归 retrievable。
```

改为：

```
【不可以】归类的优先级：**最先看末尾【知识库探测召回】是否与问题主题相关——若召回片段明显不相关（库里没有该主题），直接判 out_of_scope（最优先，无论问题是否完整、是否缺限定）**；在召回相关的前提下，再判断信息是否不足(missing_info)，再判断是否角度不定(ambiguous)，再判断是否单纯并列罗列(pending_split)；若以上都不是、但问题需要跨主题综合/多步推理/开放权衡，则判 other（积极）；其余能单轮集中命中的归 retrievable。
```

- [ ] **Step 7: prompt —— 更新末尾枚举约束**

第 73 行当前为：

```
category 仅为[retrievable|pending_split|missing_info|ambiguous|other]不允许有其他词，rewritten_query 始终返回处理后的 query，reason返回对应的原因，结果只返回 JSON，不要其他任何内容。
```

改为：

```
category 仅为[retrievable|pending_split|missing_info|ambiguous|other|out_of_scope]不允许有其他词，rewritten_query 始终返回处理后的 query，reason返回对应的原因，结果只返回 JSON，不要其他任何内容。
```

- [ ] **Step 8: 运行测试，确认通过（含回归）**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_preprocess.py -v`
Expected: PASS —— 新增 2 条通过，原有 missing_info/其他用例不破。

- [ ] **Step 9: Commit**

```bash
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "feat(classify): 新增 out_of_scope 类，missing_info 收窄回信息不足"
```

---

### Task 2: workflow 加 out_of_scope 分支（`doc_workflow.py`）

**Files:**
- Modify: `core/workflow/doc_workflow.py`（事件类区第 53–116 行；`preprocess` 第 187–223 行；分支区）
- Test: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: `PreprocessResult.category == "out_of_scope"`（Task 1 产出）。
- Produces: `OutOfScopeEvent`（无字段）；`out_of_scope_branch` step → `FinalizeEvent(answer="知识库里暂无与该问题相关的内容。", source_nodes=[])`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_doc_workflow.py` 的 `test_chitchat_responds_without_retrieval_or_classify` 之后追加：

```python
async def test_out_of_scope_responds_without_retrieval_or_clarify():
    # 库外问题（PostgreSQL）→ out_of_scope → 固定话术，不检索/不反问
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "PostgreSQL的MVCC是怎么实现的"}',
        '{"category": "out_of_scope", "rewritten_query": "PostgreSQL的MVCC是怎么实现的", "reason": "库外，召回片段均不相关"}',
    ])
    wf = _wf(llm)

    called = {"retrieve": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        called["retrieve"] = True
        return "不应被调用", []

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="PostgreSQL的MVCC是怎么实现的", memory=FakeMemory())
    assert called["retrieve"] is False                 # 库外不检索
    assert "知识库里暂无" in str(result.response)        # 固定话术
    assert result.source_nodes == []
    assert result.metadata.get("category") == "out_of_scope"  # 分类回流 metadata
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py::test_out_of_scope_responds_without_retrieval_or_clarify -v`
Expected: FAIL —— `preprocess` 的 `match` 无 `out_of_scope` case，落入 `case _` fallback 走 `RetrieveAgentEvent`，`called["retrieve"]` 变 True，断言失败。

- [ ] **Step 3: 新增 OutOfScopeEvent 事件类**

`core/workflow/doc_workflow.py`，在 `ChitchatEvent`（第 62–63 行）之后插入：

```python
class OutOfScopeEvent(Event):
    """out_of_scope → 库外问题（探测召回片段与主题无关）。如实告知，不检索/不反问。"""
```

- [ ] **Step 4: preprocess 返回类型注解 + match 加 case**

第 190 行返回注解当前为：

```python
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent | OtherEvent":
```

改为：

```python
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent | OtherEvent | OutOfScopeEvent":
```

并在 `match result.category:` 块内（第 200 行 `case "pending_split":` 之前）插入：

```python
            case "out_of_scope":
                return OutOfScopeEvent()
```

- [ ] **Step 5: 新增 out_of_scope_branch**

在 `chitchat_branch`（第 233–239 行）之后插入：

```python
    @step
    async def out_of_scope_branch(self, ctx: Context, ev: OutOfScopeEvent) -> FinalizeEvent:
        # 库外：探测召回片段与问题主题无关 → 如实告知，不检索/不合成/不反问。
        return FinalizeEvent(
            answer="知识库里暂无与该问题相关的内容。", source_nodes=[]
        )
```

- [ ] **Step 6: 运行测试，确认通过（含回归）**

Run: `.venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -v`
Expected: PASS —— 新增用例通过，现有 missing_info/chitchat/other 等编排用例不破。

- [ ] **Step 7: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): 加 out_of_scope 分支，库外问题如实告知不检索"
```

---

### Task 3: 数据集重标 + README 准则更新

**Files:**
- Modify: `eval/dataset/golden.jsonl`（第 20–23 行）
- Modify: `eval/dataset/README.md`（第 28 行 category 准则表）

**Interfaces:**
- Consumes: Task 1 的 `out_of_scope` 枚举值（评测分类准确率以 golden `category` 为金标准）。
- Produces: golden 第 20–23 条金标准 = `out_of_scope`；README 准则表区分 missing_info / out_of_scope。

- [ ] **Step 1: 重标 golden 第 20 条（PostgreSQL）**

`eval/dataset/golden.jsonl` 第 20 行：

```
{"user_input": "PostgreSQL的MVCC是怎么实现的？", "category": "missing_info", "scope": null, "reference": ""}
```

改为：

```
{"user_input": "PostgreSQL的MVCC是怎么实现的？", "category": "out_of_scope", "scope": null, "reference": ""}
```

- [ ] **Step 2: 重标 golden 第 21 条（MongoDB）**

第 21 行：

```
{"user_input": "MongoDB的分片机制是怎样的？", "category": "missing_info", "scope": null, "reference": ""}
```

改为：

```
{"user_input": "MongoDB的分片机制是怎样的？", "category": "out_of_scope", "scope": null, "reference": ""}
```

- [ ] **Step 3: 重标 golden 第 22 条（Oracle）**

第 22 行：

```
{"user_input": "Oracle的RAC架构是什么？", "category": "missing_info", "scope": null, "reference": ""}
```

改为：

```
{"user_input": "Oracle的RAC架构是什么？", "category": "out_of_scope", "scope": null, "reference": ""}
```

- [ ] **Step 4: 重标 golden 第 23 条（Cassandra）**

第 23 行：

```
{"user_input": "Cassandra的一致性级别有哪些？", "category": "missing_info", "scope": null, "reference": ""}
```

改为：

```
{"user_input": "Cassandra的一致性级别有哪些？", "category": "out_of_scope", "scope": null, "reference": ""}
```

- [ ] **Step 5: 校验 golden.jsonl 仍是合法 JSONL + 计数**

Run:
```bash
.venv/Scripts/python.exe -c "import json,collections; rows=[json.loads(l) for l in open('eval/dataset/golden.jsonl',encoding='utf-8') if l.strip()]; print(len(rows),'rows'); print(collections.Counter(r['category'] for r in rows))"
```
Expected: `23 rows` 且 `Counter` 中 `out_of_scope: 4`、`missing_info: 4`（第 16–19 行保留）。

- [ ] **Step 6: 更新 README category 准则表**

`eval/dataset/README.md` 第 28 行：

```
| `missing_info` | 缺检索必需限定，且**库里确实没有**（指代不明、查无此物）。 |
```

替换为两行：

```
| `missing_info` | 召回到相关主题、但**缺关键限定/指代不明**，补充后才能精确检索（如「这个索引的应用场景」）。→ 反问澄清。 |
| `out_of_scope` | 问题清晰，但**探测召回片段与主题明显不相关**（库里没有该主题，如 PostgreSQL/MongoDB）。→ 如实告知「库里没有」，不反问、不硬答。 |
```

- [ ] **Step 7: Commit**

```bash
git add eval/dataset/golden.jsonl eval/dataset/README.md
git commit -m "data(eval): golden 4 条库外问题重标 out_of_scope + README 准则区分两类"
```

---

### Task 4: 全量回归 + smoke 验证

**Files:**
- 无改动（纯验证）

**Interfaces:**
- Consumes: Task 1–3 的全部产出。

- [ ] **Step 1: 跑全量单测**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全绿（重点 `tests/test_query_preprocess.py`、`tests/test_doc_workflow.py`、`tests/test_eval_run.py`、`tests/test_compare.py` 不破）。

- [ ] **Step 2: 分层守卫**

Run: `.venv\Scripts\python.exe scripts/check_layering.py`
Expected: 通过（core 未引入 api 依赖）。

- [ ] **Step 3: 人工 smoke（需 DEEPSEEK_API_KEY，probe 开）**

Run: `.venv\Scripts\python.exe -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 23 --variants "全开+hybrid+rerank"`
Expected: 落盘 `eval/results/compare_<时间戳>.md`；明细 CSV 中 PostgreSQL/MongoDB/Oracle/Cassandra 四条的 `category` 列应为 `out_of_scope`、`match=1`，`response` 为「知识库里暂无与该问题相关的内容。」。

> 说明：此 smoke 调用真实 DeepSeek API。若 `out_of_scope` 识别率不达预期，回到 Task 1 Step 5 调 prompt 措辞（属预期迭代，不是计划失败）。

- [ ] **Step 4: 完成分支收尾**

实现完成、测试全绿后，使用 superpowers:finishing-a-development-branch 决定合并/PR。
