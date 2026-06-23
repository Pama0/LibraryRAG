"""检索层指标（纯函数，零依赖，可离线单测）。

retrieved：按检索序的 chunk_id 列表；relevant：相关 chunk_id 集合。
relevant 为空的 query 不该进来（评测层已过滤），各函数对空 relevant 返回 None。
"""
import math

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def _hit_count(retrieved: list[str], relevant: set[str], k: int) -> int:
    return sum(1 for rid in retrieved[:k] if rid in relevant)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """命中相关数 / 相关总数。relevant 为空 → None（不计入均值）。"""
    if not relevant:
        return None
    return _hit_count(retrieved, relevant, k) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """命中相关数 / k（标准 P@k，分母恒为 k）。k<=0 → None。"""
    if k <= 0:
        return None
    return _hit_count(retrieved, relevant, k) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """第一个相关命中的 1/rank；无命中 → 0.0。"""
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """二元增益 nDCG@k = DCG / IDCG。relevant 为空 → None。"""
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved[:k])
        if rid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else None
