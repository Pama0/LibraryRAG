"""章节树纯逻辑单测：编号解析 / 去重 / 主导前缀 / 子节点。"""
from core.workflow.chapter_tree import (
    chapter_number,
    children,
    dominant_prefix,
    unique_chapters,
)


def test_chapter_number_parses_dotted_prefix():
    assert chapter_number("1.2.1  工具A") == (1, 2, 1)
    assert chapter_number("3.2 工具系统") == (3, 2)


def test_chapter_number_none_when_no_leading_number():
    assert chapter_number("(messages/prompt + tool") is None
    assert chapter_number("") is None
    assert chapter_number("前言") is None


def test_unique_chapters_filters_book_dedups_and_drops_empty():
    metas = [
        {"book_title": "A", "chapter": "1.1 X"},
        {"book_title": "A", "chapter": "1.1 X"},   # 重复
        {"book_title": "A", "chapter": ""},          # 空
        {"book_title": "B", "chapter": "9.9 别的书"},  # 别的书
        {"book_title": "A", "chapter": "1.2 Y"},
        None,                                          # 脏
    ]
    assert unique_chapters(metas, "A") == ["1.1 X", "1.2 Y"]


def test_dominant_prefix_returns_deepest_majority_prefix():
    # 多数命中聚在 3.2.* 下
    hits = ["3.2.1 a", "3.2.2 b", "3.2.3 c", "3.5 别处"]
    assert dominant_prefix(hits) == (3, 2)


def test_dominant_prefix_none_when_scattered():
    hits = ["1.1 a", "2.3 b", "5.1 c", "(噪声"]
    assert dominant_prefix(hits) is None


def test_children_under_prefix_returns_direct_children_sorted():
    all_ch = ["3.2 工具系统", "3.2.1 工具A", "3.2.2 工具B", "3.2.1.1 细节", "3.3 别节"]
    assert children(all_ch, (3, 2)) == ["3.2.1 工具A", "3.2.2 工具B"]


def test_children_none_prefix_returns_top_level_per_group():
    all_ch = ["1.1 概述", "1.2 细节", "2.1 进阶", "2.2 更深"]
    # 每个一级分组取最浅标题
    assert children(all_ch, None) == ["1.1 概述", "2.1 进阶"]
