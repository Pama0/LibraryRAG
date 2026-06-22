"""Admitter：可答性判定单元（正交于难度分类器）。

把"能不能答"这条轴从难度六分类器里抽出来：吃 query + 召回片段，只判
ok / missing_info / out_of_scope。判据原样搬自 QueryPreprocessor 的那两段
（含"只看主体实体在不在库""深度/角度不匹配≠库外"等已调细的铁律）。

- 证据由调用方喂，不自检索（explain 喂宽召回片段、classify 喂 probe 格式化证据）。
- 沿用决策单元约定：注入 LLM、只暴露 run、json_object + Pydantic 校验、失败降级 ok、
  自带 _strip_fences 副本。
- 降级方向=放行（ok）：判定器坏了不该误拒正常问题。残留风险由 QaAgent 库外拒答补丁接住。

设计见 docs/superpowers/specs/2026-06-21-answerability-pregate-design.md。
"""
import logging
from collections import Counter
from typing import Literal, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
# 判据原样搬自 QueryPreprocessor._JUDGE_PROMPT 的 out_of_scope / missing_info 两段
# + 铁律 preamble + 优先级段（只保留可答性轴，删掉难度分类器的 retrievable/ambiguous/
# pending_split/other 判据——那些不归本单元）。
_ADMIT_PROMPT = """你是知识库问答的【可答性判定器】。下面给你一个问题和一批从知识库召回到的片段。你只判一件事：这个问题能不能基于这些片段答——能答(ok)、信息不足要反问(missing_info)、还是库外(out_of_scope)。

【铁律·必读】判定必须以末尾的【召回片段】为准，【绝不以你是否认识问题中的词为准】。知识库里全是你训练时没见过的专有名词（书名/工具名/项目名），其含义由检索决定、不由你的世界知识判断。绝不要因为"我不认识这个词"就判 missing_info 或 out_of_scope——只要召回到与问题相关且集中的内容，就是 ok。

判据（三选一）：

- ok：召回片段与问题相关（主体实体在库），且信息足够作答；或片段虽不完全覆盖但与问题主题一致、不算库外也不缺关键限定。其余皆 ok。
  返回 {"verdict":"ok","reason":"判定理由"}

- missing_info（信息不足）：**末尾【召回片段】到了与问题相关的主题**，但缺了检索必需的关键限定/指代不明，补充后才能精确命中。多为指代不明。
  如「这个索引的应用场景是什么」——库里有索引内容，但"这个索引"指代不明（全文索引？B+树索引？其他？），补充后即可检索。
  注意：若召回到了相关内容，即便问题里有你不认识的专名，也不是 missing_info。
  返回 {"verdict":"missing_info","reason":"需澄清的原因，如'这个索引'指代不明","clarify_question":"一句自然、面向用户的反问，点明不明之处并引导补充，能列候选就列，如'你说的「这个索引」具体指哪一个？是 B+树索引、全文索引，还是其他？'"}

- out_of_scope（库外）：**问题的主体技术实体根本不在知识库里**——末尾【召回片段】里找不到该实体的任何内容（召回片段讲的全是另一套系统）。判据【只看主体实体在不在库】，与问题是否完整、是否缺限定无关；因为库里没有的内容，反问也补不出来。
  判断以问题的【主体技术实体】为准（如 PostgreSQL、MongoDB、Oracle、Cassandra 这类系统/产品名）：若召回片段讲的是另一套系统，即便与问题里的通用术语（如"一致性""分片""架构""集群""事务"）字面重合，也属主体实体缺席 → out_of_scope。
  【铁律·别把"不匹配"误当库外】只要召回里出现了问题的主体实体，就【绝不是】库外——哪怕召回的【深度/角度/粒度/广度】跟用户想要的不一致（用户要"入门概念"、库里是"高阶内核细节"；用户问得很宽、库里是细节散在多章；用户要某个角度、库里是另一角度）。这类深度/角度/广度/完整性不匹配【一律不判库外】，判 ok。
  特征：召回片段讲的全是另一个系统、主体实体缺席。如「PostgreSQL的MVCC怎么实现」「MongoDB分片」「Oracle RAC」「Cassandra的一致性级别」——本库召回到的都是别的系统（如 MySQL）。
  反例（不是库外）：「给我讲懂MySQL的核心概念」——MySQL 在库，只是问得宽、要得浅，召回是高阶内核细节也无妨，判 ok（绝不判库外）。
  返回 {"verdict":"out_of_scope","reason":"库外原因，如'Cassandra 不在本库主题范围，召回片段是 MySQL 内容、主体实体缺席'"}

【判据优先级】**最先看末尾【召回片段】里问题的主体技术实体在不在库——只有主体实体根本缺席（召回全是另一套系统）才判 out_of_scope（最优先，无论问题是否完整、是否缺限定）；若主体实体在库、只是召回的深度/角度/广度与用户所求不符，不算库外，判 ok；在召回相关的前提下，再判断信息是否不足(missing_info)；其余皆 ok**。

只返回 JSON，不要其它任何内容：
{"verdict":"ok / missing_info / out_of_scope","reason":"判定理由","clarify_question":"missing_info 专用：面向用户的反问句；其余为空字符串"}

【召回片段】
{passages}

问题：{query}"""


class AdmitVerdict(BaseModel):
    """LLM 判定目标 schema（代码侧 Pydantic 校验）。

    verdict 用 Literal 锁枚举，非法值会在 model_validate 阶段被拒 → 降级 ok。
    默认 ok：构造失败兜底时直接用 AdmitVerdict() 即放行。
    """

    verdict: Literal["ok", "missing_info", "out_of_scope"] = "ok"
    reason: str = Field(default="", description="判定理由（日志/调试）")
    clarify_question: str = Field(
        default="", description="missing_info 专用：面向用户的自然反问句"
    )
    scope: Optional[list[str]] = Field(
        default=None, description="从召回 nodes 算的主导书集合；None=不收窄/全库"
    )


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _book_of(node) -> str:
    """从 NodeWithScore/TextNode 取 book_title（缺失返回空串）。"""
    meta = getattr(node, "metadata", None) or {}
    return meta.get("book_title") or ""


class Admitter:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(
        self,
        llm: LLM,
        dominant_share: float = 0.60,
        dominant_ratio: float = 2.0,
        cover_share: float = 0.80,
        max_books: int = 2,
        min_count: int = 2,
    ):
        self.llm = llm
        self.dominant_share = dominant_share
        self.dominant_ratio = dominant_ratio
        self.cover_share = cover_share
        self.max_books = max_books
        self.min_count = min_count

    def _decide_scope(self, nodes: list) -> Optional[list[str]]:
        """命中 nodes → 主导书集合 or None（不收窄）。搬自旧 ConversationScoper._decide。"""
        titles = [t for t in (_book_of(n) for n in nodes) if t]
        if not titles:
            return None
        counts = Counter(titles).most_common()          # [(book, n), ...] 降序
        total = sum(n for _b, n in counts)
        top_book, top_n = counts[0]
        second_n = counts[1][1] if len(counts) > 1 else 0
        if (
            top_n / total >= self.dominant_share
            and top_n >= self.dominant_ratio * second_n
            and top_n >= self.min_count
        ):
            return [top_book]
        prefix: list[str] = []
        acc = 0
        for book, n in counts:
            if n < self.min_count:
                break
            prefix.append(book)
            acc += n
            if len(prefix) > self.max_books:
                return None
            if acc / total >= self.cover_share:
                tail = counts[len(prefix):]
                if all(tn < self.min_count for _tb, tn in tail):
                    return prefix
                return None
        return None

    async def run(
        self, query: str, passages: list[str], nodes: Optional[list] = None
    ) -> AdmitVerdict:
        scope = self._decide_scope(nodes) if nodes else None
        passages_text = "\n---\n".join(passages) or "（无召回片段）"
        prompt = (
            _ADMIT_PROMPT.replace("{query}", query)
            .replace("{passages}", passages_text)
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            verdict = AdmitVerdict.model_validate_json(text)
            verdict.scope = scope
            logger.info("admit: verdict=%s scope=%s reason=%s", verdict.verdict, scope, verdict.reason)
            return verdict
        except Exception as exc:
            # 任何失败 → 放行（ok），绝不阻塞；判定器坏了不该误拒正常问题
            logger.warning("admit 解析失败，降级 ok（放行）：%s", exc)
            return AdmitVerdict(verdict="ok", scope=scope)
