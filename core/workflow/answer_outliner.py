"""AnswerOutliner：据【宽 hybrid 覆盖召回的片段】把"答案"拆成并列概念子主题（骨架）。

explain 专用。骨架对齐【库里实际覆盖】（喂宽召回片段），不绑章节树、不靠模型世界知识。
尺寸自适应：原子概念 1~2 节、宽主题多节，下限 1。空/失败 → []，由 qa.explain 落 agent 兜底。
设计见 docs/superpowers/specs/2026-06-20-explain-intent-workflow-design.md。
"""
import logging
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

_OUTLINE_PROMPT = """你是答案大纲规划器。下面是一个用户想"讲清楚"的问题，以及在知识库里宽召回到的相关片段。请【只依据召回片段覆盖到的内容】，把"这个问题的答案"拆成若干并列子主题，每个子主题是答案的一个方面/一节，便于逐个检索后分节讲解。

铁律：
- 子主题只能来自召回片段真实覆盖的内容，严禁凭世界知识编库里没有的子主题。
- 数量按概念复杂度【自适应】：原子概念 1~2 个即可，宽主题可多个（最多 {max} 个）。下限 1，不强凑。
- 每个子主题写成一个能独立检索的【完整短句】，含主体技术实体（别只写"应用场景"这种裸限定）。

只返回 JSON，不要其它任何内容：
{"sub_queries":["子主题1","子主题2", ...]}

问题：{query}

召回片段：
{passages}"""


class Outline(BaseModel):
    """LLM 列骨架的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class AnswerOutliner:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self, query: str, passages: List[str], max_items: int = 8
    ) -> List[str]:
        prompt = (
            _OUTLINE_PROMPT.replace("{query}", query)
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{max}", str(max_items))
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            data = Outline.model_validate_json(text)
            subs = [s.strip() for s in data.sub_queries if s and s.strip()][:max_items]
            logger.info("outline: 列出 %d 个子主题：%s", len(subs), " | ".join(subs))
            return subs
        except Exception as exc:
            logger.warning("outline 解析失败，返回空（explain 将落 agent 兜底）：%s", exc)
            return []
