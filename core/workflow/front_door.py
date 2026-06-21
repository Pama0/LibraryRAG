"""对话准入节点（Layer 1 门口）：净化 + 四出口决策。

把门口从"只分意图的 IntentRouter"升级成"带会话记忆、能应付对话表层的准入决策"。
职责：把【用户原始 query + 会话历史 + 选中的书】→ 一个有界决策：
- dispatch_qa：内容提问 → clean_query 下沉 QA 流程（红线：绝不在此自答内容）
- dispatch_study_plan：学习计划请求
- converse：寒暄/元问题/对上一轮的反馈不满 → reply 直接回复（不检索）
- clarify：指会话里某物但历史定不出所指 → reply 反问

单次 LLM 调用的结构化决策单元（非工具循环 agent）。沿用 IntentRouter/QueryPreprocessor
模式：注入 LLM、json_object、Pydantic 校验、失败降级、对外只暴露一个 run。

scope（库外）不在此判——内容问题一律 dispatch_qa，库外由下游 QueryPreprocessor 按
probe 召回证据判。设计见 docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md。
"""
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer

from core.workflow.summarizer import SUMMARY_MARKER
from core.rag.inventory import list_books_text

logger = logging.getLogger(__name__)

# 门口消指代只取最近几轮历史，别灌全量（省 token，也避免远古上下文误导）
MAX_HISTORY_MSGS = 6

# 兜底回复：converse/clarify 万一返回空 reply 时用，绝不给用户空答复
_FALLBACK_REPLY = "你好！我是文档知识库助手，可以问我已入库书籍/文档里的内容～"

# 【prompt 顺序约定】稳定指令在前、每轮变化输入（history/scope/query）在末尾，命中 DeepSeek 缓存。
# 用 .replace 注入，避免 JSON 示例花括号被 str.format 误当占位符。
_FRONT_DOOR_PROMPT = """你是知识库助手的对话门口。对下面的 query 做两件事：先净化，再决定交给哪个出口。

第一步 净化（产出 clean_query，自包含、规范）：
1) 指代消解：用【对话历史】+【当前选中的书】把"它/这个/上面说的/前面提到的/那个/这本书"等补全成不依赖上文、能独立成立的句子。无指代则不动。
2) 规范化：纠错别字/同音形近字、统一全半角、仅展开无歧义缩写（如 K8s→Kubernetes）。只改形式不改意图。
   【铁律·专名保护】知识库里全是你训练时没见过的专名（书名/工具名/项目名），它们常长得像生僻词或英文缩写的形近错字。这类不认识的 token【一律原样保留】，绝不要"纠正"成你认识的相近词——如用户写 openclaw，绝不可改成 OpenCL；写 nanoclaw 绝不可改成 NanoClaw 之外的任何词。只有在你高度确信是常见错字（如 myaql→MySQL）时才改；拿不准是不是专名，一律不动。
已自包含且规范则原样保留。

第二步 选出口（四选一，基于会话状态判断，不要自己回答任何知识内容）：
- dispatch_qa：对已入库书籍/文档内容的【具体知识提问】。把净化后的自包含问句放进 clean_query。
  铁律：凡承载知识的具体提问，哪怕你自己知道答案，也绝不在这里作答——一律 dispatch_qa 交检索系统按知识库回答。
- dispatch_study_plan：要求基于某本书生成学习计划/学习路线。clean_query 放净化后的请求。
- converse：寒暄/问候/致谢/闲聊、问你是谁或能做什么这类元问题，以及【对上一轮回答的反馈、质疑、不满、调侃】（如"你逗我呢""为什么答不了""不对吧"——参考对话历史里上一轮系统的回复来判断）。reply 放面向用户的自然回复；若上一轮是拒答/没答好而本轮是不满，先如实承认再引导。
  【元工具（仅 converse 路径可用）】若本轮是关于知识库藏书的元查询（"库里有什么""有 MySQL 的书吗""多少本"等），设 tool="list_books" + tool_filter（书名子串，大小写不敏感，如"mysql"；无过滤留空）+ tool_count_only（只要计数时 true，列清单时 false），reply 留空（系统查库后另行组织）。纯寒暄/反馈/无需库藏数据时 tool="" 照常填 reply。
  【红线】tool 只能是 list_books。绝不可用于答书里的内容问题——内容问题一律 dispatch_qa 下沉检索。
- clarify：本轮明显在指会话里的某个东西，但你无法从历史中确定所指（落在很早、或有歧义）。reply 放一句自然反问，点明不明之处，能列候选就列。

判断本轮与上一轮的关系，以【对话历史】为准，别只看这句话的字面。

只返回 JSON，不要其它任何内容：
{"action":"dispatch_qa / dispatch_study_plan / converse / clarify","clean_query":"净化后的自包含 query（dispatch 时填）","reply":"面向用户的话（converse/clarify 且无需工具时填）","reason":"简短理由","tool":"list_books 或空串（仅 converse 元查询时填 list_books）","tool_filter":"书名子串过滤（tool=list_books 时填，无过滤留空）","tool_count_only":false}

对话历史：
{history}

当前选中的书：{scope}

query：{query}"""


# 2nd LLM：converse+tool 路径用工具结果组自然回复。非 json_object，自然文本。
_COMPOSE_PROMPT = """用户问了关于知识库藏书的问题。下面是系统从知识库元数据查到的真实结果。请据此用一句自然、面向用户的话回复，不要机械复述数据。

铁律：
- 只能基于下面的【库藏数据】答，不得编造未列出的书。
- 简短自然，别寒暄一堆。

用户问题：{query}

库藏数据：
{data}"""


@dataclass
class FrontDoorDecision:
    """门口产出：action 决定 dispatch；dispatch_* 带 clean_query，converse/clarify 带 reply。

    converse+tool 时 reply 由系统查库 + 2nd LLM 组回复后填入。
    """

    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""
    tool: str = ""
    tool_filter: str = ""
    tool_count_only: bool = False


class FrontDoorDecisionModel(BaseModel):
    """LLM 判定的目标 schema（json_object 不保 schema，这步 Pydantic 校验才是约束）。

    action 用 Literal 锁枚举，非法值在 model_validate 阶段被拒、走降级。
    tool 仅 converse 路径用，锁枚举 list_books / 空串；绝不可加 book_search（红线）。
    """

    action: Literal["dispatch_qa", "dispatch_study_plan", "converse", "clarify"]
    clean_query: str = Field(default="", description="dispatch_* 的自包含 query")
    reply: str = Field(default="", description="converse/clarify 面向用户的回复")
    reason: str = Field(default="", description="简短理由")
    tool: Literal["list_books", ""] = Field(default="", description="converse 元工具，仅 list_books")
    tool_filter: str = Field(default="", description="书名子串过滤，大小写不敏感")
    tool_count_only: bool = Field(default=False, description="只要计数时 true")


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def format_history(
    memory: Optional[ChatMemoryBuffer], max_msgs: int = MAX_HISTORY_MSGS
) -> str:
    """取最近几轮历史拼成文本，喂给门口做指代消解 + 对话判断。

    若首条是摘要消息（SUMMARY_MARKER 前缀），【永远保留】它再接最近 max_msgs 条——
    摘要承载被压缩掉的远期上下文，落窗口外被截断则压缩白做。
    """
    if memory is None:
        return ""
    msgs = memory.get()
    if not msgs:
        return ""
    head: list = []
    rest = msgs
    first = msgs[0]
    if first.content and str(first.content).startswith(SUMMARY_MARKER):
        head = [first]
        rest = msgs[1:]
    rest = rest[-max_msgs:]
    return "\n".join(f"{m.role}: {m.content}" for m in (head + rest))


def format_scope(book_titles: Optional[list[str]]) -> str:
    """把用户选中的书拼成文本，喂给门口消解"这本书"类指代。"""
    if not book_titles:
        return "（用户未选择特定书籍，范围为全部已入库书籍）"
    return "".join(f"《{t}》" for t in book_titles)


class FrontDoorAgent:
    """注入 LLM + index_manager，对外只暴露一个 run。单次结构化决策 + converse 元工具路径。

    index_manager 供 converse+list_books 查库藏元数据；None 时元工具路径降级占位文本。
    """

    def __init__(self, llm: LLM, index_manager=None):
        self.llm = llm
        self.index_manager = index_manager

    async def run(
        self,
        original: str,
        memory: Optional[ChatMemoryBuffer] = None,
        book_titles: Optional[list[str]] = None,
    ) -> FrontDoorDecision:
        history = format_history(memory)
        scope = format_scope(book_titles)
        prompt = (
            _FRONT_DOOR_PROMPT.replace("{query}", original)
            .replace("{history}", history)
            .replace("{scope}", scope)
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            d = FrontDoorDecisionModel.model_validate_json(text)
            if d.action in ("dispatch_qa", "dispatch_study_plan"):
                clean = (d.clean_query or original).strip() or original
                logger.info(
                    "front_door: action=%s clean_query=%r", d.action, clean[:80]
                )
                return FrontDoorDecision(d.action, clean_query=clean, reason=d.reason)
            if d.action == "clarify":
                reply = (d.reply or "").strip() or _FALLBACK_REPLY
                logger.info("front_door: action=clarify")
                return FrontDoorDecision("clarify", reply=reply, reason=d.reason)
            # converse
            if d.tool == "list_books":
                reply = await self._converse_with_tool(original, d)
                return FrontDoorDecision(
                    "converse", reply=reply, reason=d.reason,
                    tool=d.tool, tool_filter=d.tool_filter, tool_count_only=d.tool_count_only,
                )
            # converse 无 tool：reply 直接用（空 reply 兜底）
            reply = (d.reply or "").strip() or _FALLBACK_REPLY
            logger.info("front_door: action=converse")
            return FrontDoorDecision("converse", reply=reply, reason=d.reason)
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）→ 降级 dispatch_qa + 原 query，绝不阻塞
            logger.warning("front_door 解析失败，降级 dispatch_qa + 原 query：%s", exc)
            return FrontDoorDecision("dispatch_qa", clean_query=original)

    async def _converse_with_tool(self, original: str, d: FrontDoorDecisionModel) -> str:
        """converse + list_books：查库藏元数据 → 2nd LLM 组自然回复。

        工具失败 → 占位文本进 2nd；2nd 失败/空 → 裸 tool_result 当 reply。
        """
        try:
            tool_result = list_books_text(
                self.index_manager, d.tool_filter, d.tool_count_only
            )
        except Exception as exc:
            logger.warning("front_door list_books 查询失败，用占位文本：%s", exc)
            tool_result = "（未能读取库藏清单）"
        logger.info(
            "front_door: action=converse tool=list_books title_filter=%r count_only=%s",
            d.tool_filter, d.tool_count_only,
        )
        return await self._compose_tool_reply(original, tool_result)

    async def _compose_tool_reply(self, original: str, tool_result: str) -> str:
        """2nd LLM：用工具结果 + 原 query 组自然回复。失败降级裸 tool_result。"""
        prompt = (
            _COMPOSE_PROMPT.replace("{query}", original)
            .replace("{data}", tool_result)
        )
        try:
            resp = await self.llm.acomplete(prompt)   # 非 json_object，自然文本
            text = str(resp).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("front_door compose reply 失败，用裸 tool_result：%s", exc)
        return tool_result
