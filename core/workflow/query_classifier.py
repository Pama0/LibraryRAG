"""QueryClassifier：可答性闸放行后的类型分类器（替代 gate.intent + 旧 4 类 judge）。

吃 query + probe 召回证据，判 4 类【答案形状/难度】：
- explain：想理解/讲透一个概念（"什么是X""讲讲X""X的原理"）。
- compare：比较/评价（"A和B的区别""A和B哪个好""X做缓存好吗"）——继承原 ambiguous 路线。
- simple：单一信息需求，一条检索能集中命中（原 retrievable）。
- complex：多跳依赖 / 单主题广度发散 / 开放综合权衡（原 other + pending_split），交有界 agent。

可答性（out_of_scope/missing_info）已由前置 Admitter 判完、非 ok 不会走到这里。
解析失败/空/非法 → simple（最便宜确定路径），绝不阻塞。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import logging
from dataclasses import dataclass
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入；稳定指令在前、证据/query 在末尾命中缓存。
_CLASSIFY_PROMPT = """你是检索 query 类型分类器。下面的 query 已确认可答（库内有相关内容）。请判定它属于哪一类【答案形状/难度】。

【铁律·必读】判定以末尾【知识库探测召回】为准，绝不以"你是否认识问题中的词"为准。库里全是你训练时没见过的专名（书名/工具名/项目名），其含义由检索决定。绝不要因为不认识某个词就判 complex。

四类（先判 explain/compare，再在其余里分 simple/complex）：

- explain：用户想【理解 / 讲清楚 / 讲透】一个概念或主题（"什么是X""讲讲X""讲懂X""X的原理是什么""X是怎么回事"）。
  返回 {"category":"explain","reason":"理由"}

- compare：【比较 / 评价 / 选型】两个或多个对象，或对一个对象求某种立场/角度的评价。
  如「Vue和React的区别」「Vue和React哪个好」「Redis做缓存好吗」「MySQL大表查询慢怎么优化」（有多个角度可选）。
  返回 {"category":"compare","reason":"理由"}

- simple：单一信息需求，**一条检索 query 就能集中命中**——哪怕答案要枚举若干项，只要集中在同一片区域。
  如「MySQL有哪些锁」（锁列在同一节，一次命中）。旁证：末尾召回命中集中在 1 个章节、有明显主导章 → 倾向 simple。
  返回 {"category":"simple","reason":"理由"}

- complex：需要【多跳依赖检索、单一大主题铺成多个子领域、或开放设计/权衡】，单轮答不全，须多轮检索+推理。
  · 多跳依赖：后一步查什么要看前一步检索回的答案（如「MySQL默认隔离级别会有哪些并发问题」——先查默认级别是RR，再查RR的并发问题）。
  · 广度发散：单一大主题散在多个互不重叠子领域（如「怎么优化MySQL」索引/查询/配置/架构散在多章）；旁证：末尾召回跨多个章节、无明显主导章。
  返回 {"category":"complex","reason":"理由"}

- explain是强制复杂回答，如果能判断用户不需要复杂性回答，可判为simple，比如（“简单讲讲X”）

category 仅为 [explain|compare|simple|complex]，结果只返回 JSON，不要其它任何内容。

系统已用该 query 在知识库做了一次探测检索：
【知识库探测召回】
{evidence}

query：{query}"""


@dataclass
class ClassifyResult:
    """类型分类产出。"""

    category: str
    reason: str = ""


class ClassifyJudgment(BaseModel):
    """LLM 判定目标 schema（代码侧 Pydantic 校验）。category 用 Literal 锁枚举。"""

    category: Literal["explain", "compare", "simple", "complex"]
    reason: str = Field(default="")


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryClassifier:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, query: str, evidence: str = "") -> ClassifyResult:
        prompt = (
            _CLASSIFY_PROMPT.replace("{query}", query)
            .replace("{evidence}", evidence or "（系统未能探测知识库，请仅依据问题文本判定）")
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            j = ClassifyJudgment.model_validate_json(text)
            logger.info("classifier: category=%s reason=%s", j.category, j.reason)
            return ClassifyResult(j.category, j.reason)
        except Exception as exc:
            logger.warning("classifier 解析失败，降级 simple：%s", exc)
            return ClassifyResult("simple")
