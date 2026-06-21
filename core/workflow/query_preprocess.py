"""query 预处理（QA capability 内部的第一步，从 workflow 编排中抽离）。

职责单一：把【已净化的 clean_query】→【降噪后的检索 query + 难度分类】。
- 降噪（去口语/礼貌/请求词，留实体、技术名词、限定词）
- 难度分类 → retrievable / pending_split / ambiguous / other（可答性轴 out_of_scope/missing_info 已上移到 Admitter 前置闸）

【边界】指代消解 + 规范化已在门口（front_door）完成，这里只收 clean_query，
不再读历史、不再消指代。"为检索而降噪 + 难度路由"是检索专属，故留在 QA 内部。

不持有 memory，不碰 ctx，不做路由——这些是 workflow 编排层的事。
解析失败一律降级为可检索（retrievable）用原 query，绝不阻塞。
"""
import logging
from dataclasses import dataclass
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 两步 JUDGE：clean_query 已自包含 + 规范，这里只做降噪 + 难度分类。
# 用 .replace 注入，避免 prompt 内 JSON 示例的花括号被 str.format 误当占位符。
# 【prompt 顺序约定】稳定指令（含分类定义）在前、每轮变化的输入（retrieval/query）在末尾，
# 让前缀命中 DeepSeek 上下文缓存。【铁律】里"末尾的【知识库探测召回】"依赖此顺序，勿乱挪。
_JUDGE_PROMPT = """你是检索 query 处理器。下面的 query 已经过净化（指代已消解、错别字已纠正、形式已规范），你只需做两步：先降噪，再判定该 query 的检索结构难度。
要求：如果问题已经足够清晰适合检索，降噪可不改动，不要强行改写。

第一步 降噪（去除口语化、礼貌性、请求词、无信息量的词，保留关键词语、实体、技术名词）
如（原始问题：小E,我想请问一下,MySQL有哪些锁啊?
改写后的查询：MySQL有哪些锁）
降噪只删与检索无关的冗余措辞，严禁删除任何技术限定词、修饰语、实体、版本号或承载意图的词（如"聚簇""行级""全文""有哪些""区别""第3章"）。判据：一个词删掉后若会改变检索命中，就必须保留。
反例：「MySQL聚簇索引和二级索引的区别」不可降成「MySQL 索引」——删掉"聚簇""二级""区别"会毁掉意图。

第二步 判定该 query 的检索结构难度（基于降噪后的 query）。

【铁律·必读】判定必须以末尾的【知识库探测召回】为准，【绝不以你是否认识问题中的词为准】。知识库里全是你训练时没见过的专有名词（书名/工具名/项目名），其含义由检索决定、不由你的世界知识判断。绝不要因为"我不认识这个词"或"这个词可能多义/很难"就判 other——只要召回到与问题相关且集中的内容，就是 retrievable。

【可答性由前置闸判定，本步不再判】本步只判检索结构难度。若召回片段与问题主体实体明显不相关（库外）或缺关键限定（信息不足），那已由前置可答性闸处理，本步不重复判，只关注召回相关的前提下"怎么查"。

【可以】retrievable：召回片段与问题相关，且**一条检索 query 就能集中命中**——单一信息需求，哪怕答案正文要拼几段、要枚举若干项，只要这些内容集中在同一片区域即可。
特征：仅检索问题即可。**枚举集中也算这类**——如「MySQL 有哪些锁」，锁都列在同一节，一次命中即可；枚举本身不等于要拆分。
旁证：参考末尾【知识库探测召回】的「跨 N 个章节」——命中集中在 1 个章节、有明显主导章 → 倾向 retrievable。
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}

【不可以】归入以下三类之一：

- ambiguous（角度不定）：话题已具体、能集中命中，但用户想要的维度/立场未给，有多个合理答法不知道选哪个。
  特征：答案就一个主题，但有几种角度/立场可选。
  如「Vue和React哪个好」(缺选型维度)「Redis做缓存好吗」(缺评判角度)
  返回 {"category":"ambiguous","rewritten_query": "处理后的 query","reason": "角度不定的原因，比如vue和React哪个好缺少评价维度"}

- pending_split（需要拆分）：判据=**一条检索 query 覆盖不全，必须扇出多个彼此独立的子查询**（这些子查询在**一轮粗召回定位后**就能一次性规划全、并行检索，彼此不依赖对方的检索结果）。触发于以下二者之一：
  · 多主体（**只看问题文本，与 probe 形状无关**）：问题显式含 ≥2 个并列主体、且带比较/对比/区别/异同意图（「A和B的区别」「A和B有什么不同」「A、B、C分别…」），需各自检索再综合。**即便 probe 召回看着集中在一处，这类结构也判 pending_split**——扇出各检索一侧再综合，覆盖比单轮 top-k 全，单轮易只命中泛化的上位概念而丢掉某一侧。如「Vue和React的区别」「聚簇索引和二级索引的区别」。
  · 广度分散（看 probe 形状）：单一大主题铺成多个互不重叠的子领域，单轮 top-k 覆盖不全，且末尾【知识库探测召回】显示命中**跨多个章节、无明显主导章**佐证。如「怎么优化MySQL」（索引/查询/配置/架构散在多章）。
  特征：答案需罗列/综合多个并列子项才完整，且子查询互相独立、不依赖彼此的检索结果。
  与 other 的边界：若子查询之间有**依赖**（后一个要等前一个**检索回的答案**才写得出来，即多跳），不归这里，归 other。
  返回 {"category":"pending_split","rewritten_query": "处理后的 query","reason": "需要拆分的原因，如'MySQL优化跨多章需扇出'、'Vue和React两主体需分别检索'"}

- other（高难度/开放复杂问题）：**召回到了相关内容，但**需要【多跳依赖检索、跨主题综合多步推理，或开放设计/权衡比较】，单轮、甚至一次性并行扇出都答不全，必须多轮检索+推理逐步求解。
  特征二者之一：
  · 多跳依赖：子查询之间有依赖，后一跳的 query 要等前一跳**检索回的答案**才写得出来（一轮定位后规划不完，得边检索边定下一步）。如「MySQL 默认隔离级别会有哪些并发问题」——先查出默认级别是 RR，才能去查 RR 的并发问题。
  · 开放综合/权衡：要综合多处证据分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  与 pending_split 的边界：一轮定位后就能一次产出全部子查询、彼此独立可并行（不依赖中间检索结果）→ pending_split；做不到（下一步查什么要看上一步检索的答案，或步骤集合都预定不了）→ other。
  铁律：other 看的是【问题结构是否需多跳/多步综合】，不是【你认不认识其中的词】。「X是什么 / 讲讲X / 讲明白X」这类即便 X 是你不认识的专名，只要召回到相关内容，就归 retrievable（单一概念）或 pending_split（X 是大主题需罗列），**绝不因不认识 X 而判 other**。
  返回 {"category":"other","rewritten_query": "处理后的 query", "reason":"判为高难度的原因，如'多跳依赖：需先定位默认级别再查其并发问题'、'需跨主题综合+权衡比较'"}

【不可以】归类的优先级：在召回相关的前提下，先判断是否角度不定(ambiguous)，再判断是否需扇出独立子查询(pending_split)；若以上都不是、但问题需要多跳依赖/跨主题综合/开放权衡，则判 other（积极）；其余一条 query 能集中命中的归 retrievable。

对照：
  「怎么优化MySQL」→ pending_split（优化是一整片：索引/查询/配置/架构）
  「给我讲懂MySQL的核心概念」→ pending_split（主体在库、问得宽，概念散在多章需扇出）
  「MySQL大表查询慢怎么优化」→ ambiguous（场景已具体，仍有索引/分区/改SQL几个角度）
  「Vue和React哪个好」→ ambiguous（缺"好"的维度，虽然两个实体，但仍为ambiguous）
  「Vue和React的区别」→ pending_split（不缺维度，两主体需分别检索再比，子查询独立可并行）
  「MySQL默认隔离级别会有哪些并发问题」→ other（多跳依赖：先查出默认级别，才能查该级别的并发问题，子查询有先后依赖）

category 仅为[retrievable|pending_split|ambiguous|other]不允许有其他词，rewritten_query 始终返回处理后的 query，reason返回对应的原因，结果只返回 JSON，不要其他任何内容。

系统已用该 query 在知识库做了一次探测检索：
【知识库探测召回】
{retrieval}

query：{query}"""


@dataclass
class PreprocessResult:
    """QA 内部 step1 产出：只判 category（降噪/rewritten_query 已上移到 QueryGate/Call A）。"""

    category: str
    reason: str = ""
    clarify_question: str = ""


class QueryJudgment(BaseModel):
    """LLM 判定的目标 schema。

    DeepSeek 稳定端点只有 json_object 模式（保语法不保 schema），故本模型不发给
    模型做约束，而用于【代码侧】对返回 JSON 做 Pydantic 校验。category 用 Literal
    锁枚举，非法值会在 model_validate 阶段被拒。
    """

    category: Literal[
        "retrievable", "pending_split", "ambiguous", "other"
    ]
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

    async def run(
        self, clean_query: str, retrieval_context: str = ""
    ) -> PreprocessResult:
        prompt = (
            _JUDGE_PROMPT.replace("{query}", clean_query)
            .replace(
                "{retrieval}",
                retrieval_context or "（系统未能探测知识库，请仅依据问题文本判定）",
            )
        )
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
            result = PreprocessResult(
                judgment.category, judgment.reason, judgment.clarify_question
            )
            logger.info("preprocess: category=%s reason=%s", result.category, result.reason)
            return result
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）都降级为可检索，绝不阻塞
            logger.warning("preprocess 解析失败，降级 retrievable：%s", exc)
            return PreprocessResult("retrievable")
