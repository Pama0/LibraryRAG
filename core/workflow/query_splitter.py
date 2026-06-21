"""QuerySplitter：QA 入口的多主体拆分器（降噪 + 拆分二合一）。

职责：把【已净化的 clean_query】→【≥1 个降噪后的自包含子问题】。
- 降噪：去口语/礼貌/请求词，留实体/技术名词/限定词（原 QueryGate 的降噪职责并入）。
- 拆分：仅拆"显式并列、话题独立、无比较词、无依赖"的多主体问题；比较/多跳/广度
  发散/话题共享的居中句式一律【不拆】，返回单元素，交下游 classifier 判类型。

只看问题文本，不检索。解析失败/空 → 单元素（原 query），绝不阻塞。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import logging
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入，避免 JSON 示例花括号被 str.format 误当占位符。
_SPLIT_PROMPT = """你是检索 query 预处理器。下面的 query 已净化（指代已消解、错别字已纠正）。做两件事：先降噪，再判断是否需要拆成多个独立子问题。

第一步 降噪：去掉口语化/礼貌/请求词，保留关键词、实体、技术名词、限定词。已干净则不动，不要强行改写。

第二步 拆分（只以"多主体"为判据，宁可不拆）：
【拆】同时满足：① 显式并列（A和B、A与B、A、B分别…）；② 两侧话题不同（"A的x和B的y"）或带"分别/各自"标记；③ 无比较/对比/区别词；④ 无依赖（后半不靠前半的答案）。把每个子问题写成降噪后、能独立检索的自包含短句。
【不拆】（任一即整体作为单元素返回）：
  · 比较/评价："A和B的区别""A和B哪个好"——不拆。
  · 多跳依赖：后半要先知道前半的答案——不拆。
  · 单主题广度发散："怎么优化X""讲懂X的核心概念"——不拆。
  · 话题共享且无"分别"标记的居中句式："讲讲A和B的缓存机制"——默认不拆。
铁律：拆是不可逆的（拆开就回不到跨主体对照），拿不准一律不拆，返回单元素。

无论拆不拆，sub_queries 都是降噪后的自包含短句；不拆时只含 1 个元素。

只返回 JSON，不要其它任何内容：
{"sub_queries": ["子问题1", "子问题2", ...]}

query：{query}"""


class SplitResult(BaseModel):
    """LLM 拆分结果的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QuerySplitter:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, clean_query: str) -> list[str]:
        prompt = _SPLIT_PROMPT.replace("{query}", clean_query)
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            result = SplitResult.model_validate_json(text)
            subs = [s.strip() for s in result.sub_queries if s and s.strip()]
            if not subs:
                raise ValueError("empty sub_queries")
            logger.info("splitter: %d 个子问题 %r", len(subs), subs)
            return subs
        except Exception as exc:
            logger.warning("splitter 解析失败，降级不拆（原 query）：%s", exc)
            return [clean_query]
