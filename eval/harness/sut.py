"""被测系统（SUT）抽象：协议 + DocQueryWorkflow 适配器。

map_doc_result 把 DocQueryWorkflow.run() 的返回值归一成 RagOutput，是纯函数便于单测；
DocQueryWorkflowSystem 负责实际运行与异常兜底（按决策 flag 构造，供 ablation 用）。
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | clarify | split | empty | error
    category: str = ""  # judge 判的 category（评测分类准确率用）


@runtime_checkable
class RagSystem(Protocol):
    async def answer(self, query: str) -> RagOutput: ...


# ── 当前系统（DocQueryWorkflow）适配器 ──────────────────────────────
def map_doc_result(result, response_cls=None) -> RagOutput:
    """DocQueryWorkflow.run() 的 Response → RagOutput（读 metadata.category）。"""
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813
    meta = getattr(result, "metadata", None) or {}
    category = meta.get("category", "") or ""
    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(text, [], "empty", category)
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(text, contexts, "answered", category)
    return RagOutput(str(result), [], "empty", category)


class DocQueryWorkflowSystem:
    """包装当前 DocQueryWorkflow，按决策 flag 构造，实现 RagSystem（评测 ablation 用）。"""

    def __init__(self, index_manager, llm, flags: dict | None = None,
                 similarity_top_k: int = 5, timeout: float = 120.0):
        self._index_manager = index_manager
        self._llm = llm
        self._flags = flags or {}
        self._similarity_top_k = similarity_top_k
        self._timeout = timeout

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from core.workflow.doc_workflow import DocQueryWorkflow

        wf = DocQueryWorkflow(
            index_manager=self._index_manager, llm=self._llm,
            similarity_top_k=self._similarity_top_k, timeout=self._timeout,
            **self._flags,
        )
        try:
            result = await wf.run(query=query, book_titles=book_titles)
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error", "")
        return map_doc_result(result)
