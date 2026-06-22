"""对话准入节点（Layer 1 门口）：净化 → 拆子问题 → 逐子问题选出口。

把门口拆成三步严格独立的 LLM 调用，互不知道对方的判据，各自独立降级：
1. 净化（_CLEAN_PROMPT）：原始 query + 会话历史 + 选中的书 → clean_query（指代消解 +
   规范化 + 专名保护）。【只净化，不判任何出口】。
2. 拆子问题（_SPLIT_PROMPT，必要时 + _RESPLIT_PROMPT）：clean_query → 子问题文本列表
   （多主体拆分 + 按需 probe 消歧）。【只拆分，不路由】，产出纯文本串。
3. 选出口（_ROUTE_PROMPT，一次批量）：子问题列表 + 历史 + scope → 给每个子问题判一个
   出口（dispatch_qa / converse / clarify / study_plan，list_books 元工具归 converse）。

三步产物在 _aggregate 收成对外契约 FrontDoorDecision（turn 级 action + clean_query +
reply + sub_queries），下游 doc_workflow / qa_capability 零改动：
- dispatch_qa：内容提问 → clean_query + sub_queries 下沉 QA 流程（红线：绝不在此自答内容）
- dispatch_study_plan：学习计划请求
- converse：寒暄/元问题/对上一轮的反馈不满 → reply 直接回复（不检索；list_books 走 compose）
- clarify：指会话里某物但历史定不出所指 → reply 反问

每步都是单次结构化 LLM 调用（json_object + Pydantic 校验 + 失败降级），非工具循环 agent。

scope（库外）不在此判——内容问题一律 dispatch_qa，库外由下游 QueryPreprocessor 按
probe 召回证据判。设计见 docs/superpowers/specs/2026-06-20-front-door-admission-node-design.md。
"""
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

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

# ── Step 1：净化（只产 clean_query，绝不判出口）─────────────────────────
_CLEAN_PROMPT = """你是知识库助手的净化器。对下面的 query 只做净化，产出 clean_query（自包含、规范）。【绝不判断交给哪个出口、绝不回答任何知识内容】。

1) 指代消解：用【对话历史】+【当前选中的书】把"它/这个/上面说的/前面提到的/那个/这本书"等补全成不依赖上文、能独立成立的句子。无指代则不动。
2) 规范化：纠错别字/同音形近字、统一全半角、仅展开无歧义缩写（如 K8s→Kubernetes）。只改形式不改意图。
规定：知识库里全是你训练时没见过的专名（书名/工具名/项目名），它们常长得像生僻词或英文缩写的形近错字。这类不认识的 token【一律原样保留】，绝不要"纠正"成你认识的相近词——如用户写 openclaw，绝不可改成 OpenCL；写 nanoclaw 绝不可改成 NanoClaw 之外的任何词。只有在你高度确信是常见错字（如 myaql→MySQL）时才改；拿不准是不是专名，一律不动。
已自包含且规范则原样保留。最新问题已完整则原样返回，不做任何改动。

## 示例

### 示例1
user: MySQL 的主从复制怎么工作的？
assistant: 分两阶段：binlog 写入 → relay log 回放……
user: 那它的延迟怎么监控？
输出：{"clean_query":"MySQL的主从复制的延迟怎么监控？"}

### 示例2
user: OpenClaw由什么组成
assistant: 由Gateway,Node,Agent,Tool,Session组成……
user:讲讲Gateway
输出：{"clean_query":"讲讲OpenClaw的Gateway"}

只返回 JSON，不要其它任何内容：{"clean_query":"净化后的自包含 query"}

对话历史：
{history}

当前选中的书：{scope}

query：{query}"""


# ── Step 2：拆子问题（只拆分 + 判歧义，产纯文本串，不路由）──────────────
_SPLIT_PROMPT = """你是知识库助手的子问题拆分器。下面的 query 已净化（指代已消解、错别字已纠正）。做两件事：按"多主体"拆分，再判断挂法是否存疑。【不判出口、不回答内容】。

第一步 拆分（只以"多主体"为判据，宁可不拆）：
【拆】同时满足：① 显式并列（A和B、A与B、A、B分别…）；② 两侧话题不同或带"分别/各自"标记；③ 无比较/对比/区别词；④ 无依赖。把每个子问题写成降噪后、能独立检索的自包含短句。
【不拆】（任一即整体作为单元素返回）：比较/评价（"A和B的区别/哪个好"）；多跳依赖；单主题广度发散（"怎么优化X"）；话题共享且无"分别"标记的居中句式（"讲讲A和B的缓存机制"）。
铁律：拆是不可逆的，拿不准一律不拆，返回单元素。

第二步 判歧义：若出现"A和B的X"这类修饰语作用域不定、且某挂法的存在性取决于知识（如 X 是否是 A 的概念），置 ambiguous=true，并把【存疑挂法】写进 probe_term（如 "MySQL的gateway"）；否则 ambiguous=false、probe_term 空。

只返回 JSON（sub_queries 是纯字符串数组）：
{"ambiguous":false,"probe_term":"","sub_queries":["子问题1","子问题2"]}

query：{query}"""


_RESPLIT_PROMPT = """你之前在拆分"{query}"时，对"{probe_term}"这个挂法拿不准。下面是它在知识库的探测召回。据此判断该挂法是否成立，重新给出最终子问题拆分（消歧后、降噪自包含）。若召回里找不到该挂法主体的相关内容，说明该挂法不成立，应改挂到真正拥有该概念的主体。

探测召回：
{evidence}

只返回 JSON（sub_queries 是纯字符串数组）：
{"sub_queries":["子问题1","子问题2"]}"""


# ── Step 3：选出口（一次批量给所有子问题判出口）──────────────────────────
_ROUTE_PROMPT = """你是知识库助手的出口路由器。下面是已净化、已拆分的若干子问题。给【每个】子问题判一个出口（四选一），并判断是否要求全库回答。基于【对话历史】+【当前选中的书】判断，绝不自己回答任何知识内容。

出口（四选一）：
- dispatch_qa：对已入库书籍/文档内容的【具体知识提问】（默认）。reply / tool 留空。
  铁律：凡承载知识的具体提问，哪怕你自己知道答案，也绝不在这里作答——一律 dispatch_qa 交检索系统按知识库回答。
- study_plan：要求基于某本书生成学习计划/学习路线。reply / tool 留空。
- converse：这个子问题根本不是知识提问——寒暄/问候/致谢/闲聊、问你是谁或能做什么这类元问题、对上一轮回答的反馈/质疑/不满（参考对话历史里上一轮系统的回复来判断），或要求你创作/写代码/编故事等本系统不做的事。reply 放面向用户的自然回复（婉拒/先承认再引导）。
  【元工具（仅 converse）】若是关于库藏的元查询（"库里有什么""有 MySQL 的书吗""多少本"等），设 tool="list_books" + tool_filter（书名子串，大小写不敏感，如"mysql"；无过滤留空）+ tool_count_only（只要计数 true，列清单 false），reply 留空（系统查库后另行组织）。
  【红线】tool 只能是 list_books，绝不可用于答书里的内容问题。
- clarify：明显在指会话里某个东西，但你无法从历史中确定所指（落在很早、或有歧义）。reply 放一句自然反问，能列候选就列。

纠偏：若用户明确要求【在所有书/全部书里】或【不要限定范围】回答，置 disable_scope=true（仅对 dispatch_qa 有意义；默认 false，不要随意置 true）。

判断本轮与上一轮的关系，以【对话历史】为准，别只看字面。query 字段原样回填对应子问题。

只返回 JSON：
{"disable_scope":false,"routes":[{"query":"原样回填的子问题","action":"dispatch_qa / study_plan / converse / clarify","reply":"converse/clarify 面向用户的话","tool":"list_books 或空串","tool_filter":"","tool_count_only":false}]}

对话历史：
{history}

当前选中的书：{scope}

子问题：
{subqs}"""


# 2nd LLM：converse+tool 路径用工具结果组自然回复。非 json_object，自然文本。
_COMPOSE_PROMPT = """用户问了关于知识库藏书的问题。下面是系统从知识库元数据查到的真实结果。请据此用一句自然、面向用户的话回复，不要机械复述数据。

铁律：
- 只能基于下面的【库藏数据】答，不得编造未列出的书。
- 简短自然，别寒暄一堆。

用户问题：{query}

库藏数据：
{data}"""


@dataclass
class RoutedSubQuery:
    """拆分后的一个子问题及其路由出口。"""
    query: str
    action: str = "dispatch_qa"      # dispatch_qa | converse
    reply: str = ""                  # converse 婉拒文案（dispatch_qa 时空）


@dataclass
class FrontDoorDecision:
    """门口产出：action 决定 dispatch；dispatch_* 带 clean_query，converse/clarify 带 reply。

    converse+tool 时 reply 由系统查库 + 2nd LLM 组回复后填入。
    sub_queries 仅 dispatch_qa 非空。
    """

    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""
    tool: str = ""
    tool_filter: str = ""
    tool_count_only: bool = False
    disable_scope: bool = False
    sub_queries: list = field(default_factory=list)   # list[RoutedSubQuery]，仅 dispatch_qa 非空


# ── 三步各自的 LLM 目标 schema（json_object 不保 schema，Pydantic 校验才是约束）──
class _CleanResultModel(BaseModel):
    """Step 1 净化产物。"""
    clean_query: str = Field(default="", description="净化后的自包含 query")


class _SplitResultModel(BaseModel):
    """Step 2 拆分产物：纯文本子问题 + 歧义信号（_RESPLIT 只填 sub_queries）。"""
    ambiguous: bool = False
    probe_term: str = ""
    sub_queries: List[str] = Field(default_factory=list)


class _RouteItemModel(BaseModel):
    """Step 3 单个子问题的路由。action 用 Literal 锁枚举，非法值在校验阶段被拒。"""
    query: str
    action: Literal["dispatch_qa", "converse", "clarify", "study_plan"] = "dispatch_qa"
    reply: str = ""
    tool: Literal["list_books", ""] = ""
    tool_filter: str = ""
    tool_count_only: bool = False


class _RouteResultModel(BaseModel):
    """Step 3 批量路由产物。disable_scope 是 turn 级（仅对 dispatch_qa 有意义）。"""
    disable_scope: bool = False
    routes: List[_RouteItemModel] = Field(default_factory=list)


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
    """注入 LLM + index_manager，对外只暴露一个 run。净化→拆分→选出口三步独立调用。

    index_manager 供 converse+list_books 查库藏元数据；None 时元工具路径降级占位文本。
    probe_retriever 供拆分阶段按需消歧；None 时跳过 probe 直接用趟1拆分。
    """

    def __init__(self, llm: LLM, index_manager=None, probe_retriever=None, probe_k: int = 8):
        self.llm = llm
        self.index_manager = index_manager
        self.probe_retriever = probe_retriever
        self.probe_k = probe_k

    async def run(
        self,
        original: str,
        memory: Optional[ChatMemoryBuffer] = None,
        book_titles: Optional[list[str]] = None,
    ) -> FrontDoorDecision:
        history = format_history(memory)
        scope = format_scope(book_titles)
        try:
            clean = await self._clean(original, history, scope)            # Step 1
            sub_texts = await self._split(clean, book_titles)              # Step 2
            route_result = await self._route(sub_texts, history, scope)    # Step 3
            return await self._aggregate(original, clean, route_result)
        except Exception as exc:
            # 流水线任何未捕获失败 → 降级 dispatch_qa + 原 query，绝不阻塞
            logger.warning("front_door 流水线失败，降级 dispatch_qa + 原 query：%s", exc)
            return FrontDoorDecision(
                "dispatch_qa", clean_query=original,
                sub_queries=[RoutedSubQuery(original, "dispatch_qa")],
            )

    async def _complete_json(self, prompt: str) -> str:
        """单次 json_object LLM 调用，去围栏，空返回抛错（交由各步降级）。"""
        resp = await self.llm.acomplete(prompt, response_format={"type": "json_object"})
        text = _strip_fences(str(resp)).strip()
        if not text:
            raise ValueError("empty content")
        return text

    # ── Step 1：净化 ────────────────────────────────────────────────────
    async def _clean(self, original: str, history: str, scope: str) -> str:
        """original + history + scope → clean_query。失败/空 → 原 query。"""
        prompt = (
            _CLEAN_PROMPT.replace("{query}", original)
            .replace("{history}", history)
            .replace("{scope}", scope)
        )
        try:
            text = await self._complete_json(prompt)
            c = _CleanResultModel.model_validate_json(text)
            clean = (c.clean_query or original).strip() or original
        except Exception as exc:
            logger.warning("front_door 净化失败，用原 query：%s", exc)
            clean = original
        logger.info("front_door clean: %r", clean[:80])
        return clean

    # ── Step 2：拆子问题（含按需 probe 消歧）─────────────────────────────
    async def _split(
        self, clean_query: str, book_titles: Optional[list[str]]
    ) -> list[str]:
        """clean_query → ≥1 个子问题文本。失败/空 → 单元素（不拆）。

        两趟：趟1 拆分 + 判歧义；若标 ambiguous 且有 probe 能力，探测存疑挂法
        （probe_term）一次，把召回证据喂趟2 重拆；任何环节失败都退回趟1 结果，不阻塞。
        """
        fallback = [clean_query]
        # 趟1：拆分 + 判歧义
        try:
            text = await self._complete_json(_SPLIT_PROMPT.replace("{query}", clean_query))
            r1 = _SplitResultModel.model_validate_json(text)
            subs1 = [s.strip() for s in r1.sub_queries if s and s.strip()]
            if not subs1:
                raise ValueError("empty sub_queries")
            logger.info("front_door split: %d 子问题 | %s", len(subs1), " || ".join(subs1))
        except Exception as exc:
            logger.warning("front_door 拆分趟1失败，降级不拆：%s", exc)
            return fallback

        # 无歧义 / 无 probe 能力 → 直接用趟1
        if not (r1.ambiguous and r1.probe_term and self.probe_retriever is not None):
            return subs1

        # 趟2：probe 存疑挂法 → 据证据重拆（任何失败退回趟1）
        try:
            nodes = await self.probe_retriever.retrieve(
                r1.probe_term, index_manager=self.index_manager,
                book_titles=book_titles, top_k=self.probe_k,
            )
            evidence = "\n".join(
                f"《{(getattr(n, 'metadata', None) or {}).get('book_title', '?')}》 "
                + (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:120]
                for n in nodes[:8]
            ) or "（无召回）"
            text2 = await self._complete_json(
                _RESPLIT_PROMPT.replace("{query}", clean_query)
                .replace("{probe_term}", r1.probe_term)
                .replace("{evidence}", evidence)
            )
            r2 = _SplitResultModel.model_validate_json(text2)
            subs2 = [s.strip() for s in r2.sub_queries if s and s.strip()]
            if subs2:
                logger.info("front_door resplit: %d 子问题 | %s", len(subs2), " || ".join(subs2))
            return subs2 or subs1
        except Exception as exc:
            logger.warning("front_door 消歧趟2失败，用趟1结果：%s", exc)
            return subs1

    # ── Step 3：选出口（一次批量）────────────────────────────────────────
    async def _route(
        self, sub_texts: list[str], history: str, scope: str
    ) -> _RouteResultModel:
        """子问题列表 → 每个一个出口 + turn 级 disable_scope。失败 → 全部 dispatch_qa。"""
        subqs_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_texts))
        prompt = (
            _ROUTE_PROMPT.replace("{history}", history)
            .replace("{scope}", scope)
            .replace("{subqs}", subqs_block)
        )
        try:
            text = await self._complete_json(prompt)
            r = _RouteResultModel.model_validate_json(text)
            r.routes = [ri for ri in r.routes if ri.query and ri.query.strip()]
            if not r.routes:
                raise ValueError("empty routes")
            logger.info(
                "front_door route: disable_scope=%s | %s",
                r.disable_scope,
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

    # ── 聚合：三步产物 → 对外契约 FrontDoorDecision ─────────────────────
    async def _aggregate(
        self, original: str, clean: str, rr: _RouteResultModel
    ) -> FrontDoorDecision:
        """按优先级把逐子问题路由收成 turn 级出口，保持下游契约。

        优先级：任一 dispatch_qa → dispatch_qa（带 sub_queries）；否则 study_plan →
        clarify → converse。dispatch_qa 与其它出口混排时，非 qa 子问题统一降为 converse
        装饰（qa.answer 只认 dispatch_qa / converse）。
        """
        routes = rr.routes
        if any(r.action == "dispatch_qa" for r in routes):
            subs = [
                RoutedSubQuery(r.query.strip(), "dispatch_qa")
                if r.action == "dispatch_qa"
                else RoutedSubQuery(r.query.strip(), "converse", (r.reply or "").strip())
                for r in routes
            ]
            logger.info("front_door: action=dispatch_qa subs=%d", len(subs))
            return FrontDoorDecision(
                "dispatch_qa", clean_query=clean,
                sub_queries=subs, disable_scope=rr.disable_scope,
            )

        study = next((r for r in routes if r.action == "study_plan"), None)
        if study is not None:
            logger.info("front_door: action=dispatch_study_plan")
            return FrontDoorDecision(
                "dispatch_study_plan", clean_query=(study.query.strip() or clean)
            )

        clarify = next((r for r in routes if r.action == "clarify"), None)
        if clarify is not None:
            logger.info("front_door: action=clarify")
            return FrontDoorDecision(
                "clarify", reply=(clarify.reply or "").strip() or _FALLBACK_REPLY
            )

        # 全 converse：list_books 元工具走 compose；否则拼 reply
        tool_route = next((r for r in routes if r.tool == "list_books"), None)
        if tool_route is not None:
            reply = await self._converse_with_tool(
                original, tool_route.tool_filter, tool_route.tool_count_only
            )
            return FrontDoorDecision(
                "converse", reply=reply, tool="list_books",
                tool_filter=tool_route.tool_filter, tool_count_only=tool_route.tool_count_only,
            )
        replies = [(r.reply or "").strip() for r in routes if (r.reply or "").strip()]
        reply = "\n\n".join(replies).strip() or _FALLBACK_REPLY
        logger.info("front_door: action=converse")
        return FrontDoorDecision("converse", reply=reply)

    async def _converse_with_tool(
        self, original: str, tool_filter: str, tool_count_only: bool
    ) -> str:
        """converse + list_books：查库藏元数据 → 2nd LLM 组自然回复。

        工具失败 → 占位文本进 2nd；2nd 失败/空 → 裸 tool_result 当 reply。
        """
        try:
            tool_result = list_books_text(self.index_manager, tool_filter, tool_count_only)
        except Exception as exc:
            logger.warning("front_door list_books 查询失败，用占位文本：%s", exc)
            tool_result = "（未能读取库藏清单）"
        logger.info(
            "front_door: action=converse tool=list_books title_filter=%r count_only=%s",
            tool_filter, tool_count_only,
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
