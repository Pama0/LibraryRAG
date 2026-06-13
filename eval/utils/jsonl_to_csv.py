"""把 testset JSONL 转成 CSV，方便人工查看与标注。

用法：
    python -m eval.utils.jsonl_to_csv eval/dataset/testset.draft.jsonl
    python -m eval.utils.jsonl_to_csv eval/dataset/testset.draft.jsonl -o eval/dataset/testset.draft.csv

输出列顺序为便于标注设计：先放空的 `category`/`scope` 让你边看边填，
再放 user_input / reference 等只读参考列。CSV 用 utf-8-sig（带 BOM），
Excel 直接双击打开不乱码。
"""
import argparse
import csv
import json
from pathlib import Path

# 输出列：前两列留空给人工标注，其余为生成内容（只读参考）
FIELDS = [
    "idx",            # 原始行号，方便和 jsonl 对应
    "category",       # ← 待标注（判定树定）
    "scope",          # ← 待标注（全库留空 / 填书名）
    "user_input",     # 用户问题
    "reference",      # 自动生成的参考答案（可复用）
    "query_style",
    "query_length",
    "persona_name",
    "reference_contexts",  # 检索片段（拼接，仅供肉眼核对）
]


def to_csv(src: Path, dst: Path) -> int:
    rows = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ctxs = d.get("reference_contexts") or []
            rows.append({
                "idx": i,
                "category": "",
                "scope": "",
                "user_input": d.get("user_input", ""),
                "reference": d.get("reference", ""),
                "query_style": d.get("query_style", ""),
                "query_length": d.get("query_length", ""),
                "persona_name": d.get("persona_name", ""),
                "reference_contexts": "\n---\n".join(ctxs),
            })

    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="testset JSONL → CSV")
    p.add_argument("src", help="输入 jsonl 路径")
    p.add_argument("-o", "--out", help="输出 csv 路径（默认同名 .csv）")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.out) if args.out else src.with_suffix(".csv")
    n = to_csv(src, dst)
    print(f"已写出 {n} 行 → {dst}")


if __name__ == "__main__":
    main()
