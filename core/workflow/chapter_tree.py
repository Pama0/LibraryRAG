"""章节树纯逻辑：把带编号的 chapter 标题（如 "1.2.1 工具A"）解析成可遍历的树。

供 split_branch 的"建骨架"用：从命中 chunk 的 chapter 推主导子树，再取该子树的
直接子节点标题作为拆解骨架。纯函数，无 LLM / chroma 依赖，便于单测。
"""
import re
from typing import Optional

_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def chapter_number(heading: str) -> Optional[tuple[int, ...]]:
    """'1.2.1  工具A' -> (1, 2, 1)；无前导编号 -> None。"""
    if not heading:
        return None
    m = _NUM_RE.match(heading)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def unique_chapters(metadatas: list, book_title: Optional[str] = None) -> list[str]:
    """从元数据列表抽该书去重 chapter（保序，去空）。book_title=None 不过滤。"""
    seen: set = set()
    out: list[str] = []
    for m in metadatas or []:
        if not m:
            continue
        if book_title is not None and m.get("book_title") != book_title:
            continue
        ch = (m.get("chapter") or "").strip()
        if not ch or ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def dominant_prefix(
    hit_chapters: list[str], threshold: float = 0.5
) -> Optional[tuple[int, ...]]:
    """命中 chapter 的主导编号前缀：被 >=threshold 命中共享的最深前缀。

    逐层下钻：每层取出现最多的前缀，若其占比 >= threshold*总数 且延续上层前缀，
    则继续下钻；否则停。命中散乱 / 无编号占多 -> None（信号：取顶层）。
    """
    paths = [p for p in (chapter_number(c) for c in hit_chapters) if p]
    if not paths:
        return None
    total = len(paths)
    prefix: tuple[int, ...] = ()
    depth = 1
    while True:
        counts: dict = {}
        for p in paths:
            if len(p) >= depth:
                key = p[:depth]
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            break
        best, cnt = max(counts.items(), key=lambda kv: kv[1])
        if cnt < threshold * total:
            break
        if prefix and best[: len(prefix)] != prefix:
            break
        prefix = best
        depth += 1
    return prefix or None


def children(all_chapters: list[str], prefix: Optional[tuple[int, ...]]) -> list[str]:
    """prefix 下的直接子节点标题（按编号排序）。

    prefix=None / () -> 顶层骨架：每个一级编号分组取 path 最浅的标题。
    """
    numbered = [(chapter_number(c), c) for c in all_chapters]
    numbered = [(p, c) for p, c in numbered if p]

    if not prefix:
        by_top: dict = {}
        for p, c in numbered:
            top = p[:1]
            if top not in by_top or len(p) < len(by_top[top][0]):
                by_top[top] = (p, c)
        return [c for _, (_, c) in sorted(by_top.items())]

    depth = len(prefix) + 1
    kids = [
        (p, c) for p, c in numbered if len(p) == depth and p[: len(prefix)] == prefix
    ]
    return [c for _, c in sorted(kids)]
