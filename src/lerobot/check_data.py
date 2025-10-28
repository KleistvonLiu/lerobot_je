#!/usr/bin/env python3
# validate_stats.py ----------------------------------------------------------
# 校验 episodes_stats.jsonl 中 action / observation.state 的 min‒max
# 是否落在“硬编码”的合法区间。不合法时打印 episode_index 及异常维度
# python3 src/lerobot/check_data.py /home/kleist/Documents/Database/test_0928/meta/episodes_stats.jsonl
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

# ---------- 1. 允许区间（硬编码，7 维） ------------------------------------
ALLOW_RANGE: Dict[str, Dict[str, List[float]]] = {
    # action 各维：       0        1        2         3        4        5        6
    "action": {
        "min": [-150000, -3000, -170000, -102_000, -70000, -120000, -1_000],
        "max": [ 150000, 180000,    3000,    100000,  77000,   120000,  80_000],
    },
    # observation.state 若区间相同直接复用；如有差异可单独改
    "state": {
        "min": [-150000, -3000, -170000, -100_000, -70000, -120000, -1_000],
        "max": [ 150000, 180000,    3000,    100000,  77000,   120000,  80_000],
    },
}
# ---------------------------------------------------------------------------

def bad_dims(vec_min, vec_max, allow_min, allow_max) -> List[int]:
    """返回超出区间的维度索引（0-based）。"""
    return [
        i for i, (vmin, vmax, lo, hi)
        in enumerate(zip(vec_min, vec_max, allow_min, allow_max))
        if vmin < lo or vmax > hi
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate min/max of action & observation.state in episodes_stats.jsonl"
    )
    parser.add_argument("jsonl", type=Path, help="Path to episodes_stats.jsonl")
    args = parser.parse_args()

    total, bad = 0, 0
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            epi = rec["episode_index"]
            stats = rec["stats"]

            a_min, a_max = stats["action"]["min"], stats["action"]["max"]
            s_min, s_max = stats["observation.state"]["min"], stats["observation.state"]["max"]

            bad_a = bad_dims(a_min, a_max, ALLOW_RANGE["action"]["min"], ALLOW_RANGE["action"]["max"])
            bad_s = bad_dims(s_min, s_max, ALLOW_RANGE["state"]["min"], ALLOW_RANGE["state"]["max"])

            if bad_a or bad_s:
                bad += 1
                print(f"Episode {epi} 超界：")
                if bad_a:
                    print(f"  • action 维度 {bad_a} 超界  (min={a_min}, max={a_max})")
                if bad_s:
                    print(f"  • state  维度 {bad_s} 超界  (min={s_min}, max={s_max})")

    print(f"\n完成检查：{total} 条记录，{bad} 条超出范围.")


if __name__ == "__main__":
    main()
