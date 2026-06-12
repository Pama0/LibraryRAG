from eval.sut import map_doc_result, map_workflow_result, RagOutput


class _Node:
    def __init__(self, text): self._t = text
    def get_content(self): return self._t


class _NodeWithScore:
    def __init__(self, text): self.node = _Node(text)


class _Response:
    """模拟 llama-index Response。"""
    def __init__(self, response, source_nodes):
        self.response = response
        self.source_nodes = source_nodes


class _ClarifyResult:
    """类名须为 ClarifyResult 以触发分流分支。"""
    def __init__(self, query, clarify_reason):
        self.query = query
        self.clarify_reason = clarify_reason


# 让伪类的类名匹配映射逻辑
_ClarifyResult.__name__ = "ClarifyResult"


def test_answered_extracts_text_and_contexts():
    resp = _Response("MVCC 通过 undo log 实现", [_NodeWithScore("片段A"), _NodeWithScore("片段B")])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "answered"
    assert out.response == "MVCC 通过 undo log 实现"
    assert out.retrieved_contexts == ["片段A", "片段B"]


def test_empty_when_no_nodes():
    resp = _Response("", [])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "empty"
    assert out.retrieved_contexts == []


def test_clarify_branch():
    cr = _ClarifyResult("这个索引", "指代不明")
    out = map_workflow_result(cr, response_cls=_Response)
    assert out.outcome == "clarify"
    assert out.response == ""


# ── map_doc_result：当前 DocQueryWorkflow 的 Response（带 metadata.category）──
class _RespMeta:
    """模拟带 metadata 的 DocQueryWorkflow Response。"""
    def __init__(self, response, source_nodes, metadata):
        self.response = response
        self.source_nodes = source_nodes
        self.metadata = metadata


def test_doc_answered_with_category():
    r = _RespMeta("答案", [_NodeWithScore("片段")], {"category": "retrievable", "intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
    assert out.category == "retrievable"
    assert out.retrieved_contexts == ["片段"]


def test_doc_empty_when_no_nodes():
    r = _RespMeta("反问句", [], {"category": "missing_info", "intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "empty"
    assert out.category == "missing_info"


def test_doc_handles_missing_metadata():
    r = _RespMeta("答案", [_NodeWithScore("片段")], None)
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
    assert out.category == ""   # 无 metadata → category 空，不报错
