#!/usr/bin/env python3
# validate_stats.py ----------------------------------------------------------
# 校验 episodes_stats.jsonl 中 action / observation.state 的 min‒max
# 是否落在“硬编码”的合法区间。不合法时打印 episode_index 及异常维度
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

# ---------- 单一“真源”：7 维基础阈值（只改这里） ----------------------------
_BASE7: Dict[str, Dict[str, List[float]]] = {
    # action 各维：       0        1        2         3         4        5        6
    "action": {
        "min": [-1500000,  -30000, -1700000, -102_0000,  -700000, -1200000,  -1_0000],
        "max": [ 1500000,  1800000,    30000,  1000000,   770000,  1200000,   80_0000],
    },
    # observation.state 若区间不同可与 action 分开维护
    "state": {
        "min": [-1500000,  -30000, -1700000, -100_0000,  -700000, -1200000,  -1_0000],
        "max": [ 1500000,  1800000,    30000,  1000000,   770000,  1200000,   80_0000],
    },
}

def _repeat(vals: List[float], times: int) -> List[float]:
    return vals * times

# ---------- 自动派生：7/14/21 三套区间（14=7×2，21=7×3） --------------------
ALLOW_RANGE_BY_DIM: Dict[int, Dict[str, Dict[str, List[float]]]] = {
    7: {
        "action": {
            "min": _BASE7["action"]["min"],
            "max": _BASE7["action"]["max"],
        },
        "state": {
            "min": _BASE7["state"]["min"],
            "max": _BASE7["state"]["max"],
        },
    },
    14: {
        "action": {
            "min": _repeat(_BASE7["action"]["min"], 2),
            "max": _repeat(_BASE7["action"]["max"], 2),
        },
        "state": {
            "min": _repeat(_BASE7["state"]["min"], 2),
            "max": _repeat(_BASE7["state"]["max"], 2),
        },
    },
    21: {
        "action": {
            "min": _repeat(_BASE7["action"]["min"], 3),
            "max": _repeat(_BASE7["action"]["max"], 3),
        },
        "state": {
            "min": _repeat(_BASE7["state"]["min"], 3),
            "max": _repeat(_BASE7["state"]["max"], 3),
        },
    },
    42: {
        "action": {
            "min": _repeat(_BASE7["action"]["min"], 6),
            "max": _repeat(_BASE7["action"]["max"], 6),
        },
        "state": {
            "min": _repeat(_BASE7["state"]["min"], 6),
            "max": _repeat(_BASE7["state"]["max"], 6),
        },
    },
}
# ---------------------------------------------------------------------------

def _allow_for_dim(kind: str, dim: int) -> Tuple[List[float], List[float]]:
    """根据输入维度（7/14/21）选择对应的允许区间。"""
    if dim not in ALLOW_RANGE_BY_DIM:
        raise ValueError(f"Unsupported dimensionality {dim}; only 7, 14 or 21 are supported.")
    lo = ALLOW_RANGE_BY_DIM[dim][kind]["min"]
    hi = ALLOW_RANGE_BY_DIM[dim][kind]["max"]
    if not (len(lo) == len(hi) == dim):
        raise ValueError(f"ALLOW_RANGE[{dim}]['{kind}'] length mismatch: min={len(lo)}, max={len(hi)}, dim={dim}")
    return lo, hi

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

            if len(a_min) != len(a_max):
                raise ValueError(f"Episode {epi}: action min/max length mismatch ({len(a_min)} vs {len(a_max)}).")
            if len(s_min) != len(s_max):
                raise ValueError(f"Episode {epi}: state  min/max length mismatch ({len(s_min)} vs {len(s_max)}).")

            allow_a_min, allow_a_max = _allow_for_dim("action", len(a_min))
            allow_s_min, allow_s_max = _allow_for_dim("state",  len(s_min))

            bad_a = bad_dims(a_min, a_max, allow_a_min, allow_a_max)
            bad_s = bad_dims(s_min, s_max, allow_s_min, allow_s_max)

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
