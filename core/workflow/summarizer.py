"""会话历史的持久化增量摘要。

本项目每请求从 SQLite 全量重建 memory（无状态后端），故摘要必须【落盘 + 增量更新】，
不能像 LlamaIndex 自带 ChatSummaryMemoryBuffer 那样每轮从头重摘（每轮多烧一次 LLM）。

本模块只做两件纯粹的事，不碰 DB：
- plan_overflow：给定全部消息 + 已摘要水位，算出本轮该折叠哪些旧消息（无副作用、无 LLM）。
- fold_summary：把【旧摘要 + 溢出消息】交给 LLM 折成一段新摘要（唯一的 LLM 调用）。

DB 读写与触发编排留在装配层（api/routers/chat.py），与既有 _persist_pair 一致。
借鉴 AncientAgent 的 Condenser（keep_last + 摘要溢出），但持久化、增量；且本项目
memory 只含 user/assistant 文本（无 tool 消息），故无需 Condenser 的孤儿 tool safe-cut。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 摘要消息在 memory 里的前缀标记：build_memory 用它前置摘要，front_door.format_history
# 用它识别并【永远保留】摘要头（否则摘要落在最近窗口之外会被截断，压缩等于白做）。
SUMMARY_MARKER = "[早期对话摘要]"

# 触发与保留阈值（装配层引用）：未摘要消息 > trigger 才压缩；压缩时保留最近 keep_last 条原文。
SUMMARY_TRIGGER_MSGS = 20
SUMMARY_KEEP_LAST_MSGS = 10

# 稳定指令在前、变量（prev/history）在后——前缀缓存友好，见 [[project_deepseek_prompt_caching]]。
_SUMMARIZE_PROMPT = """你是对话历史压缩器。下面给出【已有摘要】与一段更早的【新增对话】（用户与助手交替）。\
请把两者合并，用简洁中文输出一段更新后的摘要，保留对后续对话仍然重要的信息：\
用户身份/偏好、已确认的结论、讨论过的主题与未决事项。只输出摘要正文，不要寒暄或解释。

已有摘要：
{prev}

新增对话：
{history}"""


def plan_overflow(messages, summarized_upto_id: int, *, trigger: int, keep_last: int):
    """决定本轮要折进摘要的旧消息（纯函数，无 LLM、无副作用）。

    messages: 按 id 升序的全部消息（鸭子类型读 .id / .role / .content）。
    summarized_upto_id: 已折入摘要的最大消息 id（水位）。
    返回 (overflow, new_upto_id)；无需压缩时返回 (None, summarized_upto_id)。
    """
    unsummarized = [m for m in messages if m.id > summarized_upto_id]
    if len(unsummarized) <= trigger:
        return None, summarized_upto_id
    cut = len(unsummarized) - keep_last  # 留最近 keep_last 条原文，其余折叠
    if cut <= 0:  # 配置异常兜底（keep_last ≥ 未摘要数）：不产生空折叠
        return None, summarized_upto_id
    overflow = unsummarized[:cut]
    return overflow, overflow[-1].id


def _render_history(messages) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)


async def fold_summary(llm, prev_summary: Optional[str], overflow) -> str:
    """把旧摘要 + 溢出消息折成新摘要。

    失败【抛给调用方】决定（不在此静默吞）：压缩在答复送出后才跑，调用方负责
    try/except 兜底，绝不因压缩失败阻塞对话。
    """
    prompt = (
        _SUMMARIZE_PROMPT
        .replace("{prev}", prev_summary or "（无）")
        .replace("{history}", _render_history(overflow))
    )
    resp = await llm.acomplete(prompt)
    return str(resp).strip()
