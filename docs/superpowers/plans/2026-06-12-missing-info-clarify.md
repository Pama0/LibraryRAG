# missing_info → clarify（自然反问 + 降级声明假设）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `missing_info` 从「模板拼 reason 的生硬反问 + 预算耗尽时拿模糊 query 硬检索」升级为「LLM 产出的自然反问句（L1）+ 预算耗尽时声明假设、尽力答」。

**Architecture:** `QueryPreprocessor` 在判 `missing_info` 时同一次 judge 调用多产一个面向用户的 `clarify_question`（零额外 LLM）。`QaCapability.retrieve` 加可选 `preamble` 参数支持答案前声明。`DocQueryWorkflow` 接线：`clarify_branch` 用 `clarify_question`（缺失退回模板）；`allow_clarify=False` 时走 `retrieve_branch` 并把假设声明作为 preamble 传入。

**Tech Stack:** Python 3.12，LlamaIndex Workflow，DeepSeek（`OpenAILike`，json_object + Pydantic 校验），pytest + pytest-asyncio。

参考 spec：`docs/superpowers/specs/2026-06-12-missing-info-clarify-design.md`

---

## File Structure

- **Modify** `core/workflow/query_preprocess.py` — prompt 的 missing_info 返回加 `clarify_question`；`QueryJudgment` / `PreprocessResult` 加字段；`run` 透传。
- **Modify** `tests/test_query_preprocess.py` — 追加 clarify_question 解析 / 默认空测试。
- **Modify** `core/workflow/qa_capability.py` — `retrieve` 加可选 `preamble`。
- **Modify** `tests/test_qa_capability.py` — 追加 retrieve preamble 测试。
- **Modify** `core/workflow/doc_workflow.py` — `ClarifyEvent` / `RetrieveAgentEvent` 加字段；`preprocess` missing_info 两分支；`clarify_branch` / `retrieve_branch` 接线。
- **Modify** `tests/test_doc_workflow.py` — 更新 5 处 fake_retrieve 签名；追加自然反问 / 降级声明假设测试。

---

### Task 1: QueryPreprocessor 产出 clarify_question（L1）

**Files:**
- Modify: `core/workflow/query_preprocess.py`
- Test: `tests/test_query_preprocess.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_query_preprocess.py` 末尾追加（复用文件内已有的 `FakeLLM` / `_pp`）:

```python
async def test_run_missing_info_carries_clarify_question():
    llm = FakeLLM([
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明", "clarify_question": "你说的「这个索引」指哪一个？B+树索引还是全文索引？"}'
    ])
    result = await _pp(llm).run("这个索引的应用场景")
    assert result.category == "missing_info"
    assert result.clarify_question == "你说的「这个索引」指哪一个？B+树索引还是全文索引？"


async def test_run_clarify_question_defaults_empty_when_absent():
    # 非 missing_info / LLM 未给 → clarify_question 默认空
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "MySQL锁"}'])
    result = await _pp(llm).run("MySQL锁")
    assert result.clarify_question == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -k clarify_question -q`
Expected: FAIL，`AttributeError: 'PreprocessResult' object has no attribute 'clarify_question'`

- [ ] **Step 3: 改 query_preprocess.py**

(a) 在 `_JUDGE_PROMPT` 中，把 missing_info 那段：

```
- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索（多为指代不明且历史里也无从补全）。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
  返回 {"category":"missing_info","rewritten_query": "处理后的 query","reason": "需澄清的原因，如'这个索引'指代不明"}
```

替换为：

```
- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索（多为指代不明且历史里也无从补全）。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
  返回 {"category":"missing_info","rewritten_query": "处理后的 query","reason": "需澄清的原因，如'这个索引'指代不明","clarify_question": "一句自然、面向用户的反问，点明不明之处并引导补充，能列候选就列，如'你说的「这个索引」具体指哪一个？是 B+树索引、全文索引，还是其他？'"}
```

(b) `QueryJudgment` 类，在 `reason` 字段之后追加:

```python
    clarify_question: str = Field(
        default="", description="missing_info 专用：面向用户的自然反问句"
    )
```

(c) `PreprocessResult` dataclass，在 `reason: str = ""` 之后追加:

```python
    clarify_question: str = ""
```

(d) `run` 方法中，把:

```python
            return PreprocessResult(judgment.category, rewritten, judgment.reason)
```

替换为:

```python
            return PreprocessResult(
                judgment.category, rewritten, judgment.reason, judgment.clarify_question
            )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -q`
Expected: 全部 passed（原有 + 新增 2）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "feat(workflow): QueryPreprocessor 为 missing_info 产出自然反问句 clarify_question"
```

---

### Task 2: QaCapability.retrieve 加可选 preamble

**Files:**
- Modify: `core/workflow/qa_capability.py`
- Test: `tests/test_qa_capability.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_qa_capability.py` 末尾追加（复用文件内已有的 `_qa` / `FakeIndexManager` / `FakeCtx`）:

```python
# ── retrieve preamble：降级声明假设、尽力答 ──────────────────────────
async def test_retrieve_with_preamble_prepends_declaration_and_emits_delta():
    qa = _qa(FakeIndexManager(nodes=["n1"]))

    async def fake_synth(ctx, query, nodes):
        return "正文答案"

    qa._synthesize_stream = fake_synth
    ctx = FakeCtx()

    text, nodes = await qa.retrieve(ctx, "这个索引", None, preamble="（注：按最可能解读作答）")
    assert text == "（注：按最可能解读作答）正文答案"
    deltas = [e.delta for e in ctx.events if e.__class__.__name__ == "AnswerDeltaEvent"]
    assert "（注：按最可能解读作答）" in deltas


async def test_retrieve_empty_nodes_ignores_preamble():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()
    text, nodes = await qa.retrieve(ctx, "这个索引", ["书"], preamble="（注：声明）")
    assert nodes == []
    assert "（注：声明）" not in text   # 空命中只给范围提示，不带声明
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_capability.py -k preamble -q`
Expected: FAIL，`TypeError`（retrieve 不接受 preamble 关键字参数）

- [ ] **Step 3: 改 qa_capability.py**

把现有 `retrieve` 方法:

```python
    async def retrieve(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """直接检索 + 流式合成（绕开 agent/工具）。返回 (答案文本, source_nodes)。"""
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        nodes = await self._retrieve_nodes(query, book_titles)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(nodes)))
        if not nodes:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []
        answer = await self._synthesize_stream(ctx, query, nodes)
        return answer, nodes
```

替换为:

```python
    async def retrieve(
        self,
        ctx: Context,
        query: str,
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """直接检索 + 流式合成（绕开 agent/工具）。返回 (答案文本, source_nodes)。

        preamble 非空 → 进入答案阶段后先推一个 AnswerDeltaEvent，并拼在答案最前
        （供 missing_info 预算耗尽降级时声明"按最可能解读作答"）。空命中不带声明。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        nodes = await self._retrieve_nodes(query, book_titles)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(nodes)))
        if not nodes:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
        answer = await self._synthesize_stream(ctx, query, nodes)
        return (preamble + answer if preamble else answer), nodes
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_capability.py -q`
Expected: 全部 passed（原有 10 + 新增 2 = 12）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(workflow): retrieve 加可选 preamble（降级声明假设用）"
```

---

### Task 3: DocQueryWorkflow 接线（clarify_question + 降级声明假设）

**Files:**
- Modify: `core/workflow/doc_workflow.py`
- Test: `tests/test_doc_workflow.py`

- [ ] **Step 1: 改 doc_workflow.py 实现**

(a) `RetrieveAgentEvent` 类:

```python
class RetrieveAgentEvent(Event):
    """retrievable / other / 降级 → 直接检索 + 合成。"""

    rewritten_query: str
```

替换为:

```python
class RetrieveAgentEvent(Event):
    """retrievable / other / 降级 → 直接检索 + 合成。

    assumption_note 非空（missing_info 预算耗尽降级）→ 答案前声明所作假设。
    """

    rewritten_query: str
    assumption_note: str = ""
```

(b) `ClarifyEvent` 类:

```python
class ClarifyEvent(Event):
    """missing_info → 反问用户，本轮终止等补充。"""

    rewritten_query: str
    clarify_reason: str = ""
```

替换为:

```python
class ClarifyEvent(Event):
    """missing_info → 反问用户，本轮终止等补充。"""

    rewritten_query: str
    clarify_reason: str = ""
    clarify_question: str = ""
```

(c) `preprocess` step 的 missing_info 分支:

```python
            case "missing_info":
                if await ctx.store.get("allow_clarify"):
                    return ClarifyEvent(
                        rewritten_query=rewritten, clarify_reason=result.reason
                    )
                return RetrieveAgentEvent(rewritten_query=rewritten)  # 预算耗尽降级
```

替换为:

```python
            case "missing_info":
                if await ctx.store.get("allow_clarify"):
                    return ClarifyEvent(
                        rewritten_query=rewritten,
                        clarify_reason=result.reason,
                        clarify_question=result.clarify_question,
                    )
                # 预算耗尽降级：不反问，声明假设、尽力答
                note = (
                    f"（注：原问题信息不足（{result.reason}），"
                    f"以下按最可能的解读作答。）\n"
                )
                return RetrieveAgentEvent(
                    rewritten_query=rewritten, assumption_note=note
                )
```

(d) `retrieve_branch` step:

```python
    @step
    async def retrieve_branch(self, ctx: Context, ev: RetrieveAgentEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

替换为:

```python
    @step
    async def retrieve_branch(self, ctx: Context, ev: RetrieveAgentEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self.qa.retrieve(
            ctx, ev.rewritten_query, book_titles, ev.assumption_note
        )
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

(e) `clarify_branch` step:

```python
    @step
    async def clarify_branch(self, ctx: Context, ev: ClarifyEvent) -> FinalizeEvent:
        question = f"为了更准确地回答，请补充：{ev.clarify_reason}"
        # 反问句经 finalize 作为 assistant turn 进会话记忆，
        # 下一轮门口才能同时看到「原问题 + 反问 + 用户补充」一起消解。
        return FinalizeEvent(answer=question, source_nodes=[])
```

替换为:

```python
    @step
    async def clarify_branch(self, ctx: Context, ev: ClarifyEvent) -> FinalizeEvent:
        # 优先用 LLM 产出的自然反问句；缺失则退回模板拼 reason（绝不阻塞）
        question = ev.clarify_question or f"为了更准确地回答，请补充：{ev.clarify_reason}"
        # 反问句经 finalize 作为 assistant turn 进会话记忆，
        # 下一轮门口才能同时看到「原问题 + 反问 + 用户补充」一起消解。
        return FinalizeEvent(answer=question, source_nodes=[])
```

- [ ] **Step 2: 更新现有 fake_retrieve 签名（避免 TypeError）**

`retrieve_branch` 现在传第 4 个位置参数。`tests/test_doc_workflow.py` 中所有三参 stub 会 TypeError。把全部 5 处:

```python
    async def fake_retrieve(ctx, query, book_titles):
```

替换为（replace_all）:

```python
    async def fake_retrieve(ctx, query, book_titles, preamble=""):
```

- [ ] **Step 3: 追加新测试**

在 `tests/test_doc_workflow.py` 末尾追加:

```python
# ── missing_info：自然反问 / 预算耗尽降级声明假设 ──────────────────────
async def test_missing_info_uses_natural_clarify_question():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明", "clarify_question": "你说的「这个索引」指哪一个？B+树还是全文索引？"}',
    ])
    wf = _wf(llm)
    result = await wf.run(query="这个索引的应用场景", memory=FakeMemory())
    assert "你说的「这个索引」指哪一个" in str(result.response)


async def test_missing_info_budget_exhausted_assumes_and_answers():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "这个索引的应用场景"}',
        '{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明"}',
    ])
    wf = _wf(llm)

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["query"] = query
        captured["preamble"] = preamble
        return preamble + "尽力答", ["n1"]

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(
        query="这个索引的应用场景", memory=FakeMemory(), allow_clarify=False
    )
    assert "按最可能的解读作答" in captured["preamble"]   # 声明假设
    assert "尽力答" in str(result.response)
    assert result.source_nodes == ["n1"]                   # 确实检索了（未反问）
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -q`
Expected: 全部 passed（原有 6 + 新增 2 = 8）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): missing_info 自然反问 + 预算耗尽降级声明假设接线"
```

---

### Task 4: 全量回归 + 编译 + 分层守卫

**Files:** 无（验证 only）

- [ ] **Step 1: 编译改动模块**

Run: `.venv/Scripts/python.exe -m py_compile core/workflow/query_preprocess.py core/workflow/qa_capability.py core/workflow/doc_workflow.py`
Expected: 无输出（成功）

- [ ] **Step 2: 跑新链路全部测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py tests/test_qa_capability.py tests/test_doc_workflow.py tests/test_query_dimension.py tests/test_query_decompose.py tests/test_chapter_tree.py tests/test_intent_router.py tests/test_doc_query_service.py tests/test_chat_router.py -q`
Expected: 全部 passed

- [ ] **Step 3: 分层守卫**

Run: `.venv/Scripts/python.exe scripts/check_layering.py`
Expected: 通过（core/ 未 import api/）

- [ ] **Step 4: 全量（跳过遗留 book_rag 采集错误）**

Run: `.venv/Scripts/python.exe -m pytest -q --continue-on-collection-errors`
Expected: 新增测试全 passed；仅 `test_book_rag_workflow.py` / `test_book_search_tool.py` 2 errors（遗留 book_rag 语法错，本计划不处理）

- [ ] **Step 5: Commit（如有未提交的验证性微调）**

```bash
git add -A
git commit -m "test(workflow): missing_info clarify 全量回归通过"
```

---

## Self-Review Notes

- **Spec coverage:** L1 自然反问 → Task 1（preprocess 产 clarify_question）+ Task 3 (e)（clarify_branch 用之，缺失退回模板）；降级声明假设 → Task 2（retrieve preamble）+ Task 3 (c)(d)（missing_info+not allow_clarify 拼 note、retrieve_branch 传入）；流式声明在合成前 → Task 2 retrieve 先推 preamble delta + 测试断言；空命中不带声明 → Task 2 测试；测试连锁（5 处 fake_retrieve 签名）→ Task 3 Step 2 明确处理；降级（LLM 未给 clarify_question → 模板兜底）→ Task 3 (e) `or` 兜底 + 现有 `test_missing_info_clarifies_without_retrieval` 守护。L2/L3 → spec §7 明确不做。
- **Type consistency:** `PreprocessResult(category, rewritten_query, reason="", clarify_question="")`、`QueryJudgment.clarify_question`、`retrieve(ctx, query, book_titles, preamble="") -> tuple[str,list]`、`ClarifyEvent.clarify_question`、`RetrieveAgentEvent.assumption_note`，各 Task 引用一致。
- **No placeholders:** 所有步骤含完整代码与确切命令。
- **风险点:** ① Task 3 改 retrieve_branch 传 4 参，若漏改 fake_retrieve 签名则 5 个现有测试 TypeError——Step 2 专门处理；② 现有 `test_missing_info_clarifies_without_retrieval` mock 未给 clarify_question，clarify_branch 走 `or` 退回模板，断言 reason 仍在 response，不破；③ 降级 note 文案是产品措辞，可后续按真实效果调整，不影响结构。
```
