"""合并已验证候选 → golden.jsonl 第一版。

来源：
  - split_candidates.jsonl          （pending_split / other / retrievable）
  - ambiguous_missing_candidates.jsonl（ambiguous / missing_info）

输出每行最小档：{user_input, category, scope}（category=构造意图金标准，compare 读作
expected_category 算分类准确率）。同时打印类别配额 + 与 SUT 不一致清单（供人工复核）。

运行：python -m eval.datagen.merge_golden
"""
import json
import os
from collections import Counter

from eval.config import DATASET_DIR

SOURCES = ["split_candidates.jsonl", "ambiguous_missing_candidates.jsonl"]
GOLDEN = os.path.join(DATASET_DIR, "golden.jsonl")


def load(name):
    path = os.path.join(DATASET_DIR, name)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    rows = []
    review = []
    for name in SOURCES:
        for d in load(name):
            cat = d["suggested_category"]
            rows.append({"user_input": d["user_input"], "category": cat, "scope": d.get("scope")})
            if d["suggested_category"] != d["sut_category"]:
                review.append((cat, d["sut_category"], d["user_input"]))

    with open(GOLDEN, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = Counter(r["category"] for r in rows)
    print(f"已写 {GOLDEN} （共 {len(rows)} 条）")
    print("类别配额（checklist 要求每类 ≥3）:")
    for cat in ["retrievable", "pending_split", "other", "ambiguous", "missing_info"]:
        n = by.get(cat, 0)
        print(f"  {cat:14} {n}  {'✓' if n >= 3 else '✗ 不足'}")
    print(f"\n金标准 vs SUT 不一致 {len(review)} 条（人工复核：标签对则保留=暴露SUT误判）:")
    for gold, sut, q in review:
        print(f"  金标={gold:13} SUT={sut:13} | {q[:34]}")


if __name__ == "__main__":
    main()
