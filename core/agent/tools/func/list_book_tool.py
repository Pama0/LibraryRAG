from llama_index.core.tools import FunctionTool

from core.agent.tools import register_tool, ToolContext


@register_tool
class ListBooksTool:
    """列出当前已入库书籍清单（按 book_title 计数）。"""

    name = "list_books"
    description = "列出当前已入库书籍清单。"
    prompt_usage = "list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def __call__(self) -> str:
        data = self.ctx.index_manager.chroma_collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas", []) or []:
            title = (meta or {}).get("book_title")
            if not title:
                continue
            counts[title] = counts.get(title, 0) + 1
        if not counts:
            return "知识库当前为空。"
        return "已入库书籍：\n" + "\n".join(
            f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )
