from core.agent.agent import BOOK_SYSTEM_PROMPT


def test_system_prompt_has_coreference_rewrite_rule():
    # 必须提示 LLM 在调用工具前，把含指代词的问题改写为自包含 query
    assert "指代" in BOOK_SYSTEM_PROMPT
    assert "book_search" in BOOK_SYSTEM_PROMPT
