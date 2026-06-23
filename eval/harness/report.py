"""评测结果展示 + 落盘（compare 用）。

纯展示与持久化逻辑，不依赖 run_eval / compare，故两者都能 import 而不成环：
- render_delta_table：5 ragas + 成本列的 Markdown 表（单行时无 delta）。
- write_detail_csv：每条明细 CSV（含 expected_category 列供人工查阅）。
- default_result_paths：带时间戳的缺省落盘路径（默认为 compare）。
"""
import csv
import os
from datetime import datetime

# 对比表展示的列（5 ragas 质量 + 2 成本）
_COLS = [
    ("context_precision", lambda rep: rep.get("metric_means", {}).get("context_precision")),
    ("context_recall", lambda rep: rep.get("metric_means", {}).get("context_recall")),
    ("factual_correctness", lambda rep: rep.get("metric_means", {}).get("factual_correctness")),
    ("faithfulness", lambda rep: rep.get("metric_means", {}).get("faithfulness")),
    ("answer_relevancy", lambda rep: rep.get("metric_means", {}).get("answer_relevancy")),
    # 成本列：越低越好——delta 为正＝更贵（与上面质量列符号相反）
    ("时延(s/条)", lambda rep: rep.get("cost", {}).get("mean_latency_s")),
    ("tokens/条", lambda rep: rep.get("cost", {}).get("mean_total_tokens")),
]


def _fmt(val, base):
    """单元格：值 + 相对 baseline 的 delta（baseline 自身或无值不带 delta）。"""
    if val is None:
        return "—"
    if base is None or val == base:
        return f"{val:.2f}"
    return f"{val:.2f} ({val - base:+.2f})"


def render_delta_table(variants: list[dict], baseline: str) -> str:
    """variants: [{"name", "report"(aggregate 输出)}]。→ Markdown delta 表。

    单行（run_eval 单系统）时 baseline 即该行自身，各列无 delta。
    """
    base_rep = next((v["report"] for v in variants if v["name"] == baseline), None)
    if base_rep is None:
        raise ValueError(f"baseline {baseline!r} 不在 variants 中：{[v['name'] for v in variants]}")
    header = "| 配置 | " + " | ".join(c[0] for c in _COLS) + " |"
    sep = "|" + "---|" * (len(_COLS) + 1)
    lines = [header, sep]
    for v in variants:
        cells = [_fmt(getter(v["report"]), getter(base_rep)) for _, getter in _COLS]
        lines.append(f"| {v['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 明细 CSV 列顺序
_DETAIL_COLS = [
    "variant", "user_input", "expected_category", "outcome",
    "reference", "response", "num_contexts",
    "faithfulness", "answer_relevancy", "context_precision",
    "context_recall", "factual_correctness",
    "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
]

_RESULT_DIR = os.path.join("eval", "results")


def default_result_paths(prefix: str = "compare", now: "datetime | None" = None) -> tuple[str, str]:
    """缺省落盘路径：eval/results/<时间戳>/<prefix>.{md,_detail.csv}。

    每次运行一个秒级时间戳子文件夹 → 不覆盖上一次，且 md 对比表 + csv 明细同处便于归档。
    prefix 区分来源：compare（多变体）/ run_eval（单系统）。
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_RESULT_DIR, stamp)
    return (
        os.path.join(run_dir, f"{prefix}.md"),
        os.path.join(run_dir, f"{prefix}_detail.csv"),
    )


def write_detail_csv(detail: list[dict], path: str) -> None:
    """每条明细写 CSV（utf-8-sig，Excel 直开）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_DETAIL_COLS, extrasaction="ignore")
        w.writeheader()
        for d in detail:
            w.writerow(d)
