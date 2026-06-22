from typing import Literal, Optional

from llama_index.core.agent import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from core.prompts.template import load_prompt
from core.workflow.summarizer import SUMMARY_MARKER
import logging

logger = logging.getLogger(__name__)

CLEAN_PROMPT = load_prompt("clean_prompt")
SPLIT_QUERY_PROMPT = load_prompt("split_query_prompt")
ROUTE_PROMPT = load_prompt("route_prompt")

# 门口消指代只取最近几轮历史，别灌全量（省 token，也避免远古上下文误导）
MAX_HISTORY_MSGS = 6

class _CleanResultModel(BaseModel):
    is_missing_info: bool = Field(default=False,description="是否缺失信息")
    clean_query: str = Field(default="", description="净化后的自包含 query")
    missing_reason: str = Field(default="", description="信息缺失的原因")


class _RouteItemModel(BaseModel):
    """单个子问题的路由出口（schema 对齐 route_prompt.md）。"""
    query: str = Field(default="", description="原样回填的子问题")
    action: Literal["dispatch_qa", "study_plan", "converse", "clarify"] = Field(
        default="dispatch_qa", description="该子问题的出口"
    )


class _RouteResultModel(BaseModel):
    """批量路由产物：每个子问题一个出口。"""
    routes: list[_RouteItemModel] = Field(default_factory=list)



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


class FrontDoor:
    def __init__(self, llm: LLM, index_manager=None, probe_retriever=None, probe_k: int = 8):
        self.llm = llm
        self.index_manager = index_manager
        self.probe_retriever = probe_retriever
        self.probe_k = probe_k
        self._book_titles: Optional[list[str]] = None   # split_query 每轮绑定，供 probe 过滤
        self.split_agent = FunctionAgent(
            llm=self.llm,
            system_prompt=SPLIT_QUERY_PROMPT,
            tools=[
                FunctionTool.from_defaults(
                    async_fn=self.probe,
                    name="probe",
                    description=(
                        "探测某个挂法/概念是否在知识库中存在。传入存疑挂法"
                        "（如「MySQL的gateway」），返回知识库召回证据文本；"
                        "无召回即说明该挂法在库中不成立。"
                    ),
                )
            ],
        )

    async def _complete_json(self, prompt: str) -> str:
        """单次 json_object LLM 调用，去围栏，空返回抛错（交由各步降级）。"""
        resp = await self.llm.acomplete(prompt, response_format={"type": "json_object"})
        text = _strip_fences(str(resp)).strip()
        if not text:
            raise ValueError("empty content")
        return text

    async def clean(self, original: str, memory: Optional[ChatMemoryBuffer]) -> tuple[str, bool, str]:
        """original + history → clean_query。失败/空 → 原 query。"""
        history = format_history(memory)
        prompt = (
            CLEAN_PROMPT.replace("{query}", original)
            .replace("{history}", history)
        )
        is_missing_info = False
        missing_reason = ""
        try:
            text = await self._complete_json(prompt)
            c = _CleanResultModel.model_validate_json(text)
            clean_q = (c.clean_query or original).strip() or original
            is_missing_info = c.is_missing_info
            missing_reason = c.missing_reason
        except Exception as exc:
            logger.warning("front_door 净化失败，用原 query：%s", exc)
            clean_q = original
        logger.info("front_door clean: %r", clean_q[:80])
        return clean_q,is_missing_info,missing_reason

    async def split_query(
        self, clean_query: str, book_titles: Optional[list[str]]
    ) -> list[str]:
        fallback = [clean_query]
        self._book_titles = book_titles   # 绑定本轮选中的书，供 probe 工具按范围过滤召回
        try:
            result = await self.split_agent.run(clean_query)
            subs = [line.strip() for line in str(result).splitlines() if line.strip()]
            if not subs:
                raise ValueError("empty sub_queries")
            logger.info("front_door split: %d 子问题 | %s", len(subs), " || ".join(subs))
            return subs
        except Exception as exc:
            logger.warning("front_door 拆分失败，降级不拆：%s", exc)
            return fallback

    async def route(self, sub_texts: list[str]) -> _RouteResultModel:
        """子问题列表 → 每个一个出口。失败 → 全部 dispatch_qa。"""
        subqs_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_texts))
        prompt = ROUTE_PROMPT.replace("{subqs}", subqs_block)
        try:
            text = await self._complete_json(prompt)
            r = _RouteResultModel.model_validate_json(text)
            r.routes = [ri for ri in r.routes if ri.query and ri.query.strip()]
            if not r.routes:
                raise ValueError("empty routes")
            logger.info(
                "front_door route: %s",
                " || ".join(f"[{ri.action}] {ri.query}" for ri in r.routes),
            )
            return r
        except Exception as exc:
            logger.warning("front_door 路由失败，全部 dispatch_qa：%s", exc)
            return _RouteResultModel(
                routes=[
                           _RouteItemModel(query=q.strip(), action="dispatch_qa")
                           for q in sub_texts if q.strip()
                       ] or [_RouteItemModel(query="", action="dispatch_qa")]
            )

    async def probe(self, term: str) -> str:
        """探测存疑挂法 term 是否在知识库中存在，返回召回证据文本（供拆分 agent 消歧）。

        逻辑同 front_door._split 的 probe 段：按本轮选中的书检索 term，拼前若干条召回。
        无 probe 能力 / 检索失败 / 无召回都返回占位文本，由 agent 据此自行决定是否拆分。
        """
        if self.probe_retriever is None:
            return "（无探测能力，按你自己的判断决定是否拆分）"
        try:
            nodes = await self.probe_retriever.retrieve(
                term, index_manager=self.index_manager,
                book_titles=self._book_titles, top_k=self.probe_k,
            )
        except Exception as exc:
            logger.warning("front_door probe 探测失败：%s", exc)
            return "（探测失败，按你自己的判断决定是否拆分）"
        evidence = "\n".join(
            f"《{(getattr(n, 'metadata', None) or {}).get('book_title', '?')}》 "
            + (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:120]
            for n in nodes[:8]
        )
        return evidence or "（无召回，该挂法在库中不存在）"

