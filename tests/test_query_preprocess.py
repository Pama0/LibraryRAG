"""QueryPreprocessor（QA capability 内部 step）单测：降噪 + 难度分类。

瘦身后只接收 clean_query（门口已做指代消解 + 规范化），不再消指代、不再读历史。
mock LLM 控返回，验证：分类解析 / Pydantic 校验 / 失败降级 / 不含历史段。
"""
from core.workflow.query_preprocess import PreprocessResult, QueryPreprocessor


class _Resp:
    def __init__(self, text: str):
        self._t = text

    def __str__(self) -> str:
        return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


def _pp(llm):
    return QueryPreprocessor(llm)


async def test_run_denoises_and_marks_retrievable():
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "MySQL有哪些锁"}'])
    result = await _pp(llm).run("MySQL有哪些锁")
    assert isinstance(result, PreprocessResult)
    assert result.category == "retrievable"
    assert result.rewritten_query == "MySQL有哪些锁"


async def test_run_classifies_pending_split():
    llm = FakeLLM(['{"category": "pending_split", "rewritten_query": "讲讲MySQL", "reason": "需罗列子项"}'])
    result = await _pp(llm).run("讲讲MySQL")
    assert result.category == "pending_split"
    assert result.reason == "需罗列子项"


async def test_run_classifies_ambiguous_with_reason():
    llm = FakeLLM(['{"category": "ambiguous", "rewritten_query": "Vue和React哪个好", "reason": "缺评价维度"}'])
    result = await _pp(llm).run("Vue和React哪个好")
    assert result.category == "ambiguous"
    assert result.reason == "缺评价维度"


async def test_run_classifies_missing_info():
    llm = FakeLLM(['{"category": "missing_info", "rewritten_query": "这个索引的应用场景", "reason": "指代不明"}'])
    result = await _pp(llm).run("这个索引的应用场景")
    assert result.category == "missing_info"
    assert result.reason == "指代不明"


async def test_run_falls_back_to_retrievable_on_parse_failure():
    llm = FakeLLM(["这不是JSON"])
    result = await _pp(llm).run("讲讲数据库")
    assert result.category == "retrievable"   # 解析失败 → 可检索，不阻塞
    assert result.rewritten_query == "讲讲数据库"


async def test_run_rejects_invalid_category():
    # 枚举外的 category（如 clear）应被 Pydantic 拒，降级为 retrievable + 原 query
    llm = FakeLLM(['{"category": "clear", "rewritten_query": "x"}'])
    result = await _pp(llm).run("讲讲数据库")
    assert result.category == "retrievable"
    assert result.rewritten_query == "讲讲数据库"


async def test_run_prompt_has_no_history_section():
    # 瘦身铁律：QA 内部不再消指代，prompt 不得再带对话历史段
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "B+树"}'])
    await _pp(llm).run("B+树")
    assert "对话历史" not in llm.prompts[0]


async def test_run_takes_only_clean_query():
    # 签名应是 run(clean_query)，不再接收 memory
    import inspect

    params = list(inspect.signature(QueryPreprocessor.run).parameters)
    assert params == ["self", "clean_query"]


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


async def test_run_classifies_other_for_complex_open_question():
    llm = FakeLLM([
        '{"category": "other", "rewritten_query": "综合对比 openclaw 与传统方案的架构取舍", "reason": "跨主题综合 + 开放权衡"}'
    ])
    result = await _pp(llm).run("综合对比 openclaw 与传统方案的架构取舍")
    assert result.category == "other"
    assert result.reason == "跨主题综合 + 开放权衡"
