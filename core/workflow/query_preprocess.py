"""query 预处理（QA capability 内部的第一步，从 workflow 编排中抽离）。

职责单一：把【已净化的 clean_query】→【降噪后的检索 query + 难度分类】。
- 降噪（去口语/礼貌/请求词，留实体、技术名词、限定词）
- 难度分类 → retrievable / pending_split / missing_info / ambiguous / other

【边界】指代消解 + 规范化已在门口（intent_router）完成，这里只收 clean_query，
不再读历史、不再消指代。"为检索而降噪 + 难度路由"是检索专属，故留在 QA 内部。

不持有 memory，不碰 ctx，不做路由——这些是 workflow 编排层的事。
解析失败一律降级为可检索（retrievable）用原 query，绝不阻塞。
"""
from dataclasses import dataclass
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

# 两步 JUDGE：clean_query 已自包含 + 规范，这里只做降噪 + 难度分类。
# 用 .replace 注入，避免 prompt 内 JSON 示例的花括号被 str.format 误当占位符。
_JUDGE_PROMPT = """你是检索 query 处理器。下面的 query 已经过净化（指代已消解、错别字已纠正、形式已规范），你只需做两步：先降噪，再判定该 query 能否直接进入检索。
要求：如果问题已经足够清晰适合检索，降噪可不改动，不要强行改写。

第一步 降噪（去除口语化、礼貌性、请求词、无信息量的词，保留关键词语、实体、技术名词）
如（原始问题：小E,我想请问一下,MySQL有哪些锁啊?
改写后的查询：MySQL有哪些锁）
降噪只删与检索无关的冗余措辞，严禁删除任何技术限定词、修饰语、实体、版本号或承载意图的词（如"聚簇""行级""全文""有哪些""区别""第3章"）。判据：一个词删掉后若会改变检索命中，就必须保留。
反例：「MySQL聚簇索引和二级索引的区别」不可降成「MySQL 索引」——删掉"聚簇""二级""区别"会毁掉意图。

第二步 判定该 query 能否直接进入检索（基于降噪后的 query）：
【可以】可确定指向具体的技术概念/章节/问题，能检索到精准、集中的内容。
特征：仅检索问题即可
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}

【不可以】归入以下四类之一：

- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索（多为指代不明且历史里也无从补全）。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
  返回 {"category":"missing_info","rewritten_query": "处理后的 query","reason": "需澄清的原因，如'这个索引'指代不明","clarify_question": "一句自然、面向用户的反问，点明不明之处并引导补充，能列候选就列，如'你说的「这个索引」具体指哪一个？是 B+树索引、全文索引，还是其他？'"}

- ambiguous（角度不定）：话题已具体、能集中命中，但用户想要的维度/立场未给，有多个合理答法不知道选哪个。
  特征：答案就一个主题，但有几种角度/立场可选。
  如「Vue和React哪个好」(缺选型维度)「Redis做缓存好吗」(缺评判角度)
  返回 {"category":"ambiguous","rewritten_query": "处理后的 query","reason": "角度不定的原因，比如vue和React哪个好缺少评价维度"}

- pending_split （需要拆分）：问题显式包括多个实体。或话题大到要覆盖文档一整片内容，检索会命中大量分散结果。
  特征：答案需要罗列并列子项才完整。
  如「讲讲MySQL」「讲讲功能A和功能B」
  返回 {"category":"pending_split","rewritten_query": "处理后的 query","reason": "需要拆分的原因，如MySQL需要罗列子项，功能A和功能B需要分开拆解"}

- other（高难度/开放复杂问题）：需要【跨多个主题综合、多步推理，或开放设计/权衡比较】，单轮检索难以一次答全，更适合多轮检索+推理逐步求解。
  特征：要综合多处证据、需要分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  倾向（积极）：当问题明显偏复杂综合、又不属于前三类（缺信息/角度不定/单纯并列罗列）时，判为 other 交由 agent 多轮处理；仅当问题其实能单轮检索集中命中时才回到 retrievable。
  返回 {"category":"other","rewritten_query": "处理后的 query", "reason":"判为高难度的原因，如'需跨主题综合+权衡比较'"}

【不可以】归类的优先级：先判断信息是否不足(missing_info)，再判断是否角度不定(ambiguous)，再判断是否单纯并列罗列(pending_split)；若以上都不是、但问题需要跨主题综合/多步推理/开放权衡，则判 other（积极）；其余能单轮集中命中的归 retrievable。

对照：
  「怎么优化MySQL」→ pending_split（优化是一整片：索引/查询/配置/架构）
  「MySQL大表查询慢怎么优化」→ ambiguous（场景已具体，仍有索引/分区/改SQL几个角度）
  「Vue和React哪个好」→ ambiguous（缺"好"的维度，虽然两个实体，但仍为ambiguous）
  「Vue和React的区别」→ pending_split（不缺维度，需要拆分）

category 仅为[retrievable|pending_split|missing_info|ambiguous|other]不允许有其他词，rewritten_query 始终返回处理后的 query，reason返回对应的原因，结果只返回 JSON，不要其他任何内容。

query：{query}"""


@dataclass
class PreprocessResult:
    """QA 内部 step1 产出：category 决定 workflow 路由，rewritten_query 进检索。"""

    category: str
    rewritten_query: str
    reason: str = ""
    clarify_question: str = ""


class QueryJudgment(BaseModel):
    """LLM 判定的目标 schema。

    DeepSeek 稳定端点只有 json_object 模式（保语法不保 schema），故本模型不发给
    模型做约束，而用于【代码侧】对返回 JSON 做 Pydantic 校验。category 用 Literal
    锁枚举，非法值会在 model_validate 阶段被拒。
    """

    category: Literal["retrievable", "pending_split", "missing_info", "ambiguous", "other"]
    rewritten_query: str = Field(..., min_length=1, description="降噪后的检索 query")
    reason: str = Field(default="", description="对应分类的原因说明")
    clarify_question: str = Field(
        default="", description="missing_info 专用：面向用户的自然反问句"
    )


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryPreprocessor:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控分类输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, clean_query: str) -> PreprocessResult:
        prompt = _JUDGE_PROMPT.replace("{query}", clean_query)
        try:
            # json_object 模式保 JSON 语法合法（DeepSeek 稳定端点能力）；
            # 只能按调用传，别塞进全局 llm，否则 agent/synthesizer 也被迫 json 模式。
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                # DeepSeek json 模式偶发空 content，专门兜底
                raise ValueError("empty content")
            # schema 校验交给 Pydantic（json_object 不保 schema，这步才是约束）
            judgment = QueryJudgment.model_validate_json(text)
            rewritten = (judgment.rewritten_query or clean_query).strip() or clean_query
            return PreprocessResult(
                judgment.category, rewritten, judgment.reason, judgment.clarify_question
            )
        except Exception:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）都降级为可检索，绝不阻塞
            return PreprocessResult("retrievable", clean_query, "")
