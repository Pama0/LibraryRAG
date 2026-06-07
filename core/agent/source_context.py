"""请求级 source 收集容器（领域层）

工具调用时把检索到的 source_nodes 通过 contextvar 写入；
Web handler 在调用 Agent 前 begin_collection()，调用后 get_sources()。
contextvar 默认 task-local，子任务自动继承。

SourceRef 作为领域值对象定义在此，api 层从这里导入用于响应 DTO。
"""
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel


class SourceRef(BaseModel):
    book_title: str
    chapter: str
    page: int
    excerpt: str  # 引用片段


_current_sources: ContextVar[Optional[list]] = ContextVar(
    "current_sources", default=None
)

CLARIFY_MAX = 1  # 每个用户回合最多澄清次数，防止 Agent↔workflow 反复澄清死循环

# 用单元素 list 作共享可变容器（而非 int）：consume_clarify 在工具子任务里调用，
# 必须靠"原地改对象"跨 context 传播；若 .set() 重新绑定 int 则不回流父/兄弟任务。
# 与 _current_sources 用 extend 同理。
_clarify_budget: ContextVar[Optional[list]] = ContextVar("clarify_budget", default=None)

# 请求级查询范围：用户手动选择的书名列表（硬约束，是唯一的书籍过滤来源）。
# None / 空 表示全库；Agent 不再自行指定书名。
_scope_books: ContextVar[Optional[list]] = ContextVar("scope_books", default=None)


def begin_collection() -> None:
    """请求开始时调用，重置收集器、澄清预算与查询范围（按用户回合自动归位）"""
    _current_sources.set([])
    _clarify_budget.set([CLARIFY_MAX])
    _scope_books.set(None)  # 由 handler 随后 set_scope 显式注入


def set_scope(books: Optional[list]) -> None:
    """请求级：写入用户手动选择的查询范围。空/None 视为全库。"""
    _scope_books.set(list(books) if books else None)


def get_scope() -> Optional[list]:
    """读取当前请求的用户选定查询范围；None 表示未选（全库）。"""
    return _scope_books.get()


def can_clarify() -> bool:
    """当前回合是否还有澄清预算。未 begin_collection 的路径默认 False（倒向检索，不阻塞）。"""
    bucket = _clarify_budget.get()
    return bool(bucket) and bucket[0] > 0


def consume_clarify() -> None:
    """消费一次澄清预算（原地改，跨子任务 context 共享）。"""
    bucket = _clarify_budget.get()
    if bucket:
        bucket[0] = max(0, bucket[0] - 1)


def add_sources(refs: list) -> None:
    """工具内调用，追加 SourceRef 列表"""
    bucket = _current_sources.get()
    if bucket is None:
        return
    bucket.extend(refs)


def get_sources() -> list:
    """请求结束时取出收集到的 sources（去重保序）"""
    bucket = _current_sources.get()
    if not bucket:
        return []
    seen = set()
    unique = []
    for s in bucket:
        key = (s.book_title, s.chapter, s.page, s.excerpt[:50])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def node_to_source_ref(node) -> SourceRef:
    """LlamaIndex node -> SourceRef"""
    meta = node.metadata or {}
    return SourceRef(
        book_title=meta.get("book_title", "未知"),
        chapter=meta.get("chapter", ""),
        page=meta.get("page", meta.get("page_start", 0)) or 0,
        excerpt=(node.get_content() if hasattr(node, "get_content") else node.text)[:300],
    )
