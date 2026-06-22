"""Admitter（可答性判定单元）单测：mock LLM 控返回，验解析/降级/证据进 prompt。"""
from llama_index.core.schema import NodeWithScore, TextNode

from core.workflow.admitter import Admitter, AdmitVerdict


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


def _adm(llm):
    return Admitter(llm)


async def test_run_parses_ok():
    llm = FakeLLM(['{"verdict":"ok","reason":"主体在库且相关"}'])
    v = await _adm(llm).run("MySQL有哪些锁", ["片段A"])
    assert isinstance(v, AdmitVerdict)
    assert v.verdict == "ok"
    assert v.reason == "主体在库且相关"
    assert v.clarify_question == ""


async def test_run_parses_missing_info_with_clarify():
    llm = FakeLLM([
        '{"verdict":"missing_info","reason":"指代不明","clarify_question":"你说的「这个索引」指哪一个？B+树还是全文索引？"}'
    ])
    v = await _adm(llm).run("这个索引的应用场景", ["片段A"])
    assert v.verdict == "missing_info"
    assert v.clarify_question == "你说的「这个索引」指哪一个？B+树还是全文索引？"


async def test_run_parses_out_of_scope():
    llm = FakeLLM([
        '{"verdict":"out_of_scope","reason":"PostgreSQL 不在库，召回全是 MySQL"}'
    ])
    v = await _adm(llm).run("PostgreSQL的MVCC", ["MySQL 片段"])
    assert v.verdict == "out_of_scope"
    assert v.reason == "PostgreSQL 不在库，召回全是 MySQL"


async def test_run_injects_passages_into_prompt():
    llm = FakeLLM(['{"verdict":"ok"}'])
    await _adm(llm).run("openclaw 是什么", ["片段甲", "片段乙"])
    assert "片段甲" in llm.prompts[0]
    assert "片段乙" in llm.prompts[0]
    assert "openclaw 是什么" in llm.prompts[0]
    assert "json_object" not in llm.prompts[0]   # 不进 prompt 正文


async def test_run_empty_passages_still_works():
    llm = FakeLLM(['{"verdict":"out_of_scope","reason":"召回空，主体缺席"}'])
    v = await _adm(llm).run("Cassandra分片", [])
    assert v.verdict == "out_of_scope"


async def test_run_parse_failure_degrades_to_ok():
    llm = FakeLLM(["这不是JSON"])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"            # 失败 → 放行，不误拒


async def test_run_empty_content_degrades_to_ok():
    llm = FakeLLM([""])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"


async def test_run_invalid_verdict_rejected_to_ok():
    # 枚举外的 verdict 应被 Pydantic 拒 → 降级 ok
    llm = FakeLLM(['{"verdict":"maybe"}'])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"


async def test_run_strips_fenced_json():
    llm = FakeLLM(['```json\n{"verdict":"ok"}\n```'])
    v = await _adm(llm).run("MySQL锁", ["片段"])
    assert v.verdict == "ok"


def _nodes(books):
    return [
        NodeWithScore(node=TextNode(text="x", id_=str(i), metadata={"book_title": b}))
        for i, b in enumerate(books)
    ]


async def test_scope_single_dominant_book():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("讲讲MySQL", ["片段"], nodes=_nodes(["MySQL"] * 6 + ["X"] * 2))
    assert v.verdict == "ok"
    assert v.scope == ["MySQL"]


async def test_scope_two_books_when_concept_spans():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("讲讲索引", ["片段"], nodes=_nodes(["A"] * 4 + ["B"] * 3 + ["C"] * 1))
    assert v.scope == ["A", "B"]


async def test_scope_none_when_diffuse():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"], nodes=_nodes(["A"] * 3 + ["B"] * 3 + ["C"] * 2))
    assert v.scope is None


async def test_scope_none_when_no_nodes():
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"], nodes=[])
    assert v.scope is None


async def test_scope_none_when_nodes_arg_omitted():
    # 旧调用方不传 nodes → scope=None，行为不变（回归）
    llm = FakeLLM(['{"verdict":"ok"}'])
    v = await _adm(llm).run("q", ["片段"])
    assert v.scope is None


async def test_scope_none_on_verdict_parse_failure():
    # LLM 坏 → 降级 ok，scope 仍可从 nodes 算出（verdict 与 scope 解耦）
    llm = FakeLLM(["这不是JSON"])
    v = await _adm(llm).run("讲讲MySQL", ["片段"], nodes=_nodes(["MySQL"] * 6))
    assert v.verdict == "ok"
    assert v.scope == ["MySQL"]
