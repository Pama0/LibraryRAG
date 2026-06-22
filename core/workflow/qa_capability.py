"""QA capability：文档库问答的检索 + 合成实质逻辑（从 DocQueryWorkflow 抽出）。

`DocQueryWorkflow` 是顶层 Router 编排，把 `intent=qa` dispatch 到这里。本类是
注入式协作单元（与 `IntentRouter` / `QuerySplitter` / `QueryClassifier` 同一
模式），【不依赖 LlamaIndex Workflow step 机制】，便于独立单测。

持有：
- 检索（`index_manager`）+ 合成（`llm`）。
- 拆分 `QuerySplitter`（一句话 → ≥1 个降噪自包含子问题）。
- 可答性闸 `Admitter`（probe 证据判 ok/missing_info/out_of_scope）+ 瘦身分类
  `QueryClassifier`（ok 子问题判 explain/compare/simple/complex）。

对外主入口：`answer(ctx, clean_query, book_titles, probe)` —— 拆分 → 并行逐子问题判定
（`_decide_subq`：probe → admit → classify）→ 按序执行 ok 子问题（`_execute_subq` 按
category 分派）→ 合并装饰非 ok 子问题。单问题退化为旧单路径（无分节标题）。
分支执行体仍独立可调：`retrieve`/`explain`/`split`/`assume`（均接收 query/book_titles，
返回 (answer, source_nodes)）。

流式：检索/合成进度通过 `ctx.write_event_to_stream` 推【流式专用事件】（本模块定义，
`doc_workflow` re-export、api 层映射成前端 SSE）。这些事件不参与 workflow step 图。
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context, Event

from core.retrieval.rerank import Reranker
from core.retrieval.retrieve import Retriever, VectorRetriever
from core.workflow.admitter import Admitter, AdmitVerdict
from core.workflow.chapter_tree import children, dominant_prefix, unique_chapters
from core.workflow.query_classifier import QueryClassifier
from core.workflow.query_decompose import QueryDecomposer
from core.workflow.query_dimension import DimensionExtractor
from core.workflow.query_splitter import QuerySplitter
from core.workflow.answer_outliner import AnswerOutliner

logger = logging.getLogger(__name__)

# explain 整合教学写作 prompt（用 .replace 注入，避免 JSON/花括号被 str.format 误解）
# 要的是讲师的【思维方式】（拆解、选高度、做减法、先总览后分述），不是师生【角色扮演】。
_TEACH_PROMPT = """请把"{query}"讲清楚、讲透。下面给你一份讲解骨架（要讲的几个维度，按顺序）和一批从知识库检索到的资料片段。请用讲师剖析一个概念时的思路——先拆出主干、选对讲解高度、把握详略——据此写一篇连贯的讲解。

身份与口吻：
- 你是问答系统，对方是提问的用户。【不要】用"同学们""大家好""今天我们要讲""这节课"等课堂或师生口吻，也不要自称老师、把对方当学生。直接作为解答给出，像一篇清晰的技术讲解文，而非一场课堂表演。

写作要求：
- 先用一段话总起：点出这个主题整体是什么、要从哪几个方面来看；然后按骨架分节展开；最后一两句收束。开场直接进入主题，不要寒暄或宣布"开始上课"。
- 分节用轻量小标题：每个维度一个「## 维度名」小标题，只写骨架里列出的维度；节与节之间要有承接，不要各写各的。
- 【做减法、选高度】目标是讲透而非堆资料：只讲帮助理解主题的内容；资料片段里与当前维度无关的零碎细节（具体字段名、内部常量等）一律略去，除非它直接支撑某个维度的论点。

铁律（grounding，不可违反）：
- 事实只能来自下面的【资料片段】，严禁用你自己的训练知识或常识补充片段里没有的事实。
- 片段没覆盖到的维度，如实说"资料中未涉及"，不要编造。

讲解骨架（按此顺序分节）：
{plan}

资料片段：
{passages}"""


class EmptySkeleton(Exception):
    """AnswerOutliner 列不出骨架 → 由 explain_branch 落 agent 兜底。"""


class OutOfScope(Exception):
    """explain 路 admit 判库外 → 由 explain_branch 接住拒答。镜像 EmptySkeleton 的异常驱动控制流。"""


class MissingInfo(Exception):
    """explain 路 admit 判信息不足 → 由 explain_branch 接住反问。

    clarify_question 由 Admitter 产；缺时 explain_branch 用 REFUSAL_FALLBACK 兜底。
    """

    def __init__(self, clarify_question: str = ""):
        super().__init__(clarify_question or "")
        self.clarify_question = clarify_question or ""


# ── 拒答话术共享常量（库外分支与 explain OutOfScope catch 共用，避免分叉）──
REFUSAL_TEXT = (
    "这个问题知识库里暂未收录相关内容，我没法基于现有资料回答。"
    "你可以换个已入库主题问我，或把问题换个角度再试试～"
)
# missing_info 缺 clarify_question 时的兜底反问
REFUSAL_FALLBACK = "为了更准确地回答，能不能把问题再说具体一点？"


# ── 流式专用事件（仅 write_event_to_stream，不参与 step 图）──────────
class RetrievalStartEvent(Event):
    """开始检索（→ 前端 tool_call）。"""

    query: str


class RetrievalDoneEvent(Event):
    """检索完成（→ 前端 tool_result，触发答案阶段）。"""

    count: int


class AnswerDeltaEvent(Event):
    """合成阶段逐 token（→ 前端 delta）。"""

    delta: str


@dataclass
class _SubDecision:
    """单个子问题的判定结果：可答性 verdict + （ok 时）类型 category。"""

    query: str
    verdict: str = "ok"          # ok / missing_info / out_of_scope
    category: str = ""           # explain/compare/simple/complex（仅 ok）
    reason: str = ""
    clarify_question: str = ""


class QaCapability:
    """文档库问答能力：降噪分类 → 检索 → 流式合成 / 拆解汇总。注入式，独立可测。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        reranker: "Reranker | None" = None,
        rerank_candidate_k: int = 20,
        retriever: "Retriever | None" = None,
        probe_retriever: "Retriever | None" = None,
        probe_reranker: "Reranker | None" = None,
        explain_retriever: "Retriever | None" = None,
        explain_recall_k: int = 12,
        simple_escalate_min_score: float = 0.0,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        # 检索不可跳过，基线=具体 VectorRetriever（不传即基线）
        self.retriever = retriever or VectorRetriever()
        # probe（探测召回判类）的检索与答案检索解耦：probe 求覆盖信号、答案求精排。
        # 默认 vector + 不重排（rerank 会收敛召回、压扁 pending_split 依赖的章节 spread 信号）。
        self.probe_retriever = probe_retriever or VectorRetriever()
        self.probe_reranker = probe_reranker
        self.decomposer = QueryDecomposer(llm)
        self.dimensioner = DimensionExtractor(llm)
        # explain 专用：宽覆盖召回（hybrid + 大 top_k + 不重排，求覆盖不求精）
        self.explain_retriever = explain_retriever or VectorRetriever()
        self.explain_recall_k = explain_recall_k
        self.simple_escalate_min_score = simple_escalate_min_score
        self.admitter = Admitter(llm)
        self.outliner = AnswerOutliner(llm)
        self._retrieve_concurrency = 4  # 扇出检索并发上限，防 embedding/BM25/rerank 打爆
        self.splitter = QuerySplitter(llm)
        self.classifier = QueryClassifier(llm)
        # 有界 agent 由 doc_workflow 构造后注入（complex / simple 升级用）；None → 降级单轮
        self.qa_agent = None

    async def _decide_subq(
        self, q: str, book_titles: Optional[list[str]], probe: bool = True
    ) -> "_SubDecision":
        """单子问题判定：probe → admit（非 ok 短路）→ classify。失败一律放行/降级。"""
        evidence = ""
        if probe:
            try:
                located = await self._probe_retrieve(q, book_titles)
                evidence = self._format_probe(located, book_titles)
            except Exception as exc:
                logger.warning("_decide_subq probe 失败，纯文本判定：%s", exc)
        try:
            verdict = await self.admitter.run(q, [evidence])
        except Exception as exc:
            logger.warning("_decide_subq admit 抛错，降级 ok：%s", exc)
            verdict = None
        if verdict is not None and verdict.verdict == "out_of_scope":
            return _SubDecision(q, "out_of_scope", reason=verdict.reason)
        if verdict is not None and verdict.verdict == "missing_info":
            return _SubDecision(
                q, "missing_info", reason=verdict.reason,
                clarify_question=verdict.clarify_question,
            )
        result = await self.classifier.run(q, evidence)
        return _SubDecision(q, "ok", category=result.category, reason=result.reason)

    def _evidence_weak(self, nodes: list) -> bool:
        """simple 安全网触发判据：召回空 / top-1 分数低于阈值（complex 误判成 simple 时升级）。"""
        if not nodes:
            return True
        top = max((getattr(n, "score", 0) or 0) for n in nodes)
        return top < self.simple_escalate_min_score

    async def _execute_subq(
        self, ctx: Context, q: str, category: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """按 category 分派执行；simple 证据不足升级 agent，complex agent 异常降级单轮。"""
        if category == "explain":
            return await self.explain(ctx, q, book_titles)
        if category == "compare":
            return await self.assume(ctx, q, book_titles)
        if category == "complex":
            if self.qa_agent is None:
                return await self.retrieve(ctx, q, book_titles)
            try:
                return await self.qa_agent.run(ctx, q, book_titles)
            except Exception as exc:
                logger.warning("complex agent 失败，降级单轮：%s", exc)
                return await self.retrieve(ctx, q, book_titles)
        # simple（含分类降级）：先检索一次，证据不足且有 agent → 升级；否则复用节点合成
        nodes = await self._retrieve_nodes(q, book_titles)
        if self._evidence_weak(nodes) and self.qa_agent is not None:
            logger.info("simple 证据不足，升级 agent：%r", q[:60])
            try:
                return await self.qa_agent.run(ctx, q, book_titles)
            except Exception as exc:
                logger.warning("simple 升级 agent 失败，回落单轮：%s", exc)
        return await self.retrieve(ctx, q, book_titles, nodes=nodes)

    async def split_query(self, clean_query: str) -> list[str]:
        """委托 QuerySplitter：clean_query → ≥1 个降噪自包含子问题。"""
        return await self.splitter.run(clean_query)

    async def answer(
        self,
        ctx: Context,
        clean_query: str,
        book_titles: Optional[list[str]],
        probe: bool = True,
    ) -> tuple[str, list, dict]:
        """顶层编排：拆分 → 并行逐子问题判定 → 按序执行 ok 子问题 → 合并装饰。

        - 并行只用于判定阶段（无用户可见输出）；执行/合成按子问题顺序串行（保流式顺序）。
        - 单问题：无分节标题、无合并装饰，等价旧单路径。
        - 部分非 ok：先答 ok 的，末尾追加 missing_info 反问 / out_of_scope "不在库" 提示。
        - 全非 ok：纯拒答（out_of_scope→REFUSAL_TEXT）/反问（missing_info）。
        """
        sub_qs = await self.split_query(clean_query)
        decisions = await asyncio.gather(
            *(self._decide_subq(q, book_titles, probe=probe) for q in sub_qs)
        )
        oks = [d for d in decisions if d.verdict == "ok"]
        missing = [d for d in decisions if d.verdict == "missing_info"]
        oos = [d for d in decisions if d.verdict == "out_of_scope"]
        multi = len(sub_qs) > 1
        meta = {
            "categories": [d.category for d in oks],
            "sub_count": len(sub_qs),
            "category": (oks[0].category if oks else "out_of_scope")
            if len(sub_qs) == 1 else "multi",
        }

        # 全非 ok：退化纯拒答/反问（单条复用原话术）
        if not oks:
            if missing:
                q = missing[0].clarify_question or REFUSAL_FALLBACK
                return q, [], meta
            return REFUSAL_TEXT, [], meta

        # 执行 ok 子问题（按序流式）。多问题加分节标题；单问题裸答。
        parts: list[str] = []
        all_nodes: list = []
        for d in oks:
            if multi:
                heading = f"\n## {d.query}\n"
                ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
                parts.append(heading)
            ans, nodes = await self._execute_subq(ctx, d.query, d.category, book_titles)
            parts.append(ans)
            all_nodes.extend(nodes)

        # 末尾装饰：out_of_scope / missing_info 子问题（仅多问题且存在时）
        tail = self._compose_tail(oos, missing) if multi else ""
        if tail:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=tail))
            parts.append(tail)

        return "".join(parts).strip(), all_nodes, meta

    @staticmethod
    def _compose_tail(oos: list, missing: list) -> str:
        """合并末尾提示：库外子问题如实告知 + 信息不足子问题反问。"""
        lines: list[str] = []
        if oos:
            names = "、".join(f"「{d.query}」" for d in oos)
            lines.append(f"另外，{names} 知识库里暂未收录相关内容，无法作答。")
        for d in missing:
            lines.append(d.clarify_question or f"关于「{d.query}」，能再说具体一点吗？")
        return ("\n\n" + "\n".join(lines)) if lines else ""

    def _format_probe(self, nodes: list, book_titles) -> str:
        """探测召回 → 喂 judge 的信号：命中数 + 章节分布 + top 截断片段。"""
        if not nodes:
            return "知识库未召回到任何相关内容。"
        dist: list[str] = []
        seen: set = set()
        for n in nodes:
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            if tag not in seen:
                seen.add(tag)
                dist.append(tag)
        lines: list[str] = []
        for i, n in enumerate(nodes[:5], 1):
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            content = (
                n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            )[:150]
            lines.append(f"{i}. [{tag}] {content}")
        return (
            f"共命中 {len(nodes)} 段，跨 {len(dist)} 个章节：{'、'.join(dist)}\n"
            + "\n".join(lines)
        )

    # ── 分支：单轮检索 + 流式合成 ────────────────────────────────────
    async def retrieve(
        self,
        ctx: Context,
        query: str,
        book_titles: Optional[list[str]],
        preamble: str = "",
        nodes: Optional[list] = None,
    ) -> tuple[str, list]:
        """直接检索 + 流式合成（绕开 agent/工具）。返回 (答案文本, source_nodes)。

        preamble 非空 → 进入答案阶段后先推一个 AnswerDeltaEvent，并拼在答案最前
        （供 missing_info 预算耗尽降级时声明"按最可能解读作答"）。空命中不带声明。
        nodes 非空 → 复用已检索节点，跳过内部检索（供 simple 安全网"检索一次→判定→合成"
        复用，避免二次检索）。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        if nodes is None:
            nodes = await self._retrieve_nodes(query, book_titles)
        logger.info("retrieve: 命中 %d 段 scope=%s", len(nodes), book_titles)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(nodes)))
        if not nodes:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
        answer = await self._synthesize_stream(ctx, query, nodes)
        return (preamble + answer if preamble else answer), nodes

    async def assume(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """角度不定：定位 → LLM 归纳评判维度 → 声明所选角度 → 逐维度检索分节合成。

        归纳不出维度 → 降级为单轮合成（复用已定位结果，绝不阻塞）。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿正文供归纳维度
        located = await self._retrieve_nodes(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 2) 归纳维度：从「问题 + 召回正文」产 (label, query) 维度对
        dimensions = await self.dimensioner.run(query, passages, self.max_sub_queries)

        # 降级：归纳不出维度 → 整句单轮合成
        if not dimensions:
            logger.info("assume: 无维度，降级单轮检索")
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 3) 声明所选角度（透明 + 可纠偏）
        labels = "、".join(d.label for d in dimensions)
        preamble = f"「{query}」可以从以下角度来看：{labels}。下面分别说明——\n"

        # 4) 逐维度检索 + 分节合成（与 split 共用 helper）
        sections = [(d.label, d.query) for d in dimensions]
        return await self._retrieve_and_concat(ctx, sections, book_titles, preamble)

    async def explain(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """讲清楚：宽覆盖召回 → 教学维度教案(吃 TOC) → 每维度检索 → 合并截断 → 一次整合教学写作。

        空教案 → raise EmptySkeleton（由 explain_branch 落 agent 兜底）。
        结构来自教学 schema + 书的 TOC（自上而下、不被召回碎片带偏）；事实只来自检索 pool。
        """
        # 1. 宽覆盖召回（内部，不发流事件——空教案时要静默落 agent，别先污染 UI）
        located = await self._explain_recall(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 1.5 可答性闸：吃宽召回片段判 ok/missing_info/out_of_scope；非 ok 抛异常
        # 由 explain_branch 接住拒答/反问。admit 失败降级 ok（放行），不阻塞。
        try:
            verdict = await self.admitter.run(query, passages)
        except Exception as exc:
            logger.warning("explain admit 抛错，降级 ok 放行：%s", exc)
            verdict = AdmitVerdict(verdict="ok")
        if verdict.verdict == "out_of_scope":
            raise OutOfScope(query)
        if verdict.verdict == "missing_info":
            raise MissingInfo(verdict.clarify_question)

        # 2. 出教案：教学维度词表 + 书的 TOC 提示（单书才有，多书/未选 → []）
        toc_hint = self._book_chapters(book_titles)
        outline = await self.outliner.run(query, passages, toc_hint)
        if not outline:
            raise EmptySkeleton(query)

        # 3. 每维度检索扇出 → 去重合并 → 截断/重排到上下文预算（此时才发 RetrievalStart）
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        retrieved = await self._retrieve_all([d.query for d in outline], book_titles)
        pool = self._merge_pool(retrieved)
        if self.reranker:
            pool = await self.reranker.rerank(query, pool, self.rerank_candidate_k)
        else:
            pool = sorted(
                pool, key=lambda n: getattr(n, "score", 0) or 0, reverse=True
            )[: self.rerank_candidate_k]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
        if not pool:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []

        # 4. 一次整合教学写作（教案当脚手架、讲师 prompt 立 grounding + 做减法）
        answer = await self._teach_synthesize(ctx, query, outline, pool)
        return answer, pool

    async def split(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """定位 → 建骨架(结构主+内容辅) → 逐项检索 → map-reduce 汇总。

        拆解失败/空 → 降级为单轮检索+合成（等同 retrieve），绝不阻塞。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿命中 chunk 的 chapter
        located = await self._retrieve_nodes(query, book_titles)

        # 2) 建骨架：章节子树标题（结构）+ 召回正文（内容）→ 子查询
        all_chapters = self._book_chapters(book_titles)
        hit_chapters = [(n.metadata or {}).get("chapter", "") for n in located]
        prefix = dominant_prefix(hit_chapters)
        headings = children(all_chapters, prefix)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]
        sub_queries, mode = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )

        # 降级：拆不出子查询 → 整句单轮合成
        if not sub_queries:
            logger.info("split: 无子查询，降级单轮检索")
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 综合型：扇出检索 → 去重合并 → 对原始问题一次整合合成（单子查询天然退化单轮）
        if mode == "synthesize":
            logger.info("进入整合路线")
            return await self._retrieve_and_synthesize(ctx, query, sub_queries, book_titles)

        # 罗列型：逐项检索 + map-reduce 汇总（与 assume 共用同一 helper）
        logger.info("进入罗列路线")
        sections = [(sq, sq) for sq in sub_queries]
        return await self._retrieve_and_concat(ctx, sections, book_titles)

    @staticmethod
    def _node_id(n) -> object:
        """稳定去重键：优先 NodeWithScore.node.node_id，退回 node_id，再退回对象 id。"""
        node = getattr(n, "node", None)
        return getattr(node, "node_id", None) or getattr(n, "node_id", None) or id(n)

    def _merge_pool(self, lists: list[list]) -> list:
        """多路检索结果按 node_id 去重合并，保首次出现顺序。"""
        seen: set = set()
        out: list = []
        for ns in lists:
            for n in ns:
                k = self._node_id(n)
                if k in seen:
                    continue
                seen.add(k)
                out.append(n)
        return out

    async def _retrieve_and_synthesize(
        self,
        ctx: Context,
        original_query: str,
        sub_queries: list[str],
        book_titles: Optional[list[str]],
    ) -> tuple[str, list]:
        """synthesize 模式：扇出检索（并发）→ 去重合并 → 对原始问题一次整合合成。

        子查询只为拓宽召回面；合成用【原始问题】，让 LLM 同时看到所有子项原始片段去比较/讲关系。
        """
        retrieved = await self._retrieve_all(sub_queries, book_titles)
        pool = self._merge_pool(retrieved)
        if self.reranker:
            # 拿原始问题（非子查询）对合并池重排，截到上下文预算
            pool = await self.reranker.rerank(original_query, pool, self.rerank_candidate_k)
        else:
            pool = sorted(pool, key=lambda n: getattr(n, "score", 0) or 0, reverse=True)[
                : self.rerank_candidate_k
            ]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
        if not pool:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{original_query}」相关的内容。", []
        answer = await self._synthesize_stream(ctx, original_query, pool)
        return answer, pool

    # ── 公共流水线：逐项检索 → 一次 RetrievalDone →（可选声明）→ 逐节合成拼接 ──
    async def _retrieve_and_concat(
        self,
        ctx: Context,
        sections: list[tuple[str, str]],
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """sections: [(分节标题, 检索/合成用子查询)]。list 模式：逐节裸拼（split / assume 共用）。

        - 扇出检索并发；逐节合成仍串行（保分节流式顺序）。
        - 先全检索（只发一次 RetrievalDone）。preamble 非空 → 答案阶段先推一个 AnswerDeltaEvent。
        """
        retrieved = await self._retrieve_all([sq for _h, sq in sections], book_titles)
        all_nodes: list = [n for ns in retrieved for n in ns]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        parts: list[str] = []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
            parts.append(preamble)
        for (heading, sub_query), ns in zip(sections, retrieved):
            h = f"\n## {heading}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=h))
            body = (
                await self._synthesize_stream(ctx, sub_query, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(h + body)
        return "".join(parts).strip(), all_nodes

    # ── helpers：章节 / 检索 / 流式合成 ──────────────────────────────
    def _book_chapters(self, book_titles: Optional[list[str]]) -> list[str]:
        """取单一选定书的去重 chapter 列表；未选或多选 → []（结构缺失，倒向内容主导）。"""
        if not book_titles or len(book_titles) != 1:
            return []
        data = self.index_manager.chroma_collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        return unique_chapters(metas, book_titles[0])

    async def _retrieve_with(self, query, book_titles, retriever, reranker):
        # 检索策略可插拔（默认 VectorRetriever=基线）；有 reranker 时过召回候选池再重排截断
        fetch_k = self.rerank_candidate_k if reranker else self.similarity_top_k
        nodes = await retriever.retrieve(
            query, index_manager=self.index_manager,
            book_titles=book_titles, top_k=fetch_k,
        )
        if reranker:
            nodes = await reranker.rerank(query, nodes, self.similarity_top_k)
        return nodes

    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        """答案检索：用注入的 retriever/reranker。"""
        return await self._retrieve_with(
            query, book_titles, self.retriever, self.reranker
        )

    async def _probe_retrieve(self, query: str, book_titles: Optional[list[str]]):
        """probe 探测召回：用独立的 probe_retriever/probe_reranker（默认 vector / 不重排）。"""
        return await self._retrieve_with(
            query, book_titles, self.probe_retriever, self.probe_reranker
        )

    async def _explain_recall(self, query: str, book_titles: Optional[list[str]]):
        """explain 宽覆盖召回：hybrid + 大 top_k + 不重排（求"有哪几块"，不求精）。"""
        return await self.explain_retriever.retrieve(
            query, index_manager=self.index_manager,
            book_titles=book_titles, top_k=self.explain_recall_k,
        )

    async def _retrieve_all(
        self, sub_queries: list[str], book_titles: Optional[list[str]]
    ) -> list[list]:
        """并发扇出检索：对每个子查询各检索一次，返回与入参同序的 node 列表的列表。"""
        sem = asyncio.Semaphore(self._retrieve_concurrency)

        async def _one(q: str):
            async with sem:
                return await self._retrieve_nodes(q, book_titles)

        return await asyncio.gather(*(_one(q) for q in sub_queries))

    async def _stream_tokens(self, query: str, nodes: list):
        """流式合成的 token 源。单独成方法便于单测替身。"""
        synthesizer = get_response_synthesizer(llm=self.llm, streaming=True)
        resp = await synthesizer.asynthesize(query=query, nodes=nodes)
        async for token in resp.async_response_gen():
            yield token

    async def _synthesize_stream(self, ctx: Context, query: str, nodes: list) -> str:
        """逐 token 合成：每 token 推一个 AnswerDeltaEvent，最后拼成完整答案。"""
        parts: list[str] = []
        async for token in self._stream_tokens(query, nodes):
            parts.append(token)
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=token))
        return "".join(parts)

    async def _teach_tokens(self, prompt: str):
        """讲师整合写作的 token 源：直接对 prompt 流式 complete。单独成方法便于单测替身。"""
        handle = await self.llm.astream_complete(prompt)
        async for chunk in handle:
            yield chunk.delta or ""

    async def _teach_synthesize(
        self, ctx: Context, query: str, outline: list, pool: list
    ) -> str:
        """教案(维度顺序) + (已由调用方截断的) pool → 讲师 prompt → 一次流式整合写作。

        结构来自教案（教学先验/TOC，安全元知识）；事实只来自 pool 片段（prompt 立铁律）。
        逐 token 发 AnswerDeltaEvent，前端零改动。
        """
        plan = "\n".join(f"- {d.label}：{d.query}" for d in outline)
        passages = "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))
            for n in pool
        )
        prompt = (
            _TEACH_PROMPT.replace("{query}", query)
            .replace("{plan}", plan)
            .replace("{passages}", passages or "（无）")
        )
        parts: list[str] = []
        async for tok in self._teach_tokens(prompt):
            parts.append(tok)
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=tok))
        return "".join(parts)
