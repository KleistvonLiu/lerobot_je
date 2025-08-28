#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
对比保存成视频前和从视频解析出来的深度图的数据
python compare_depth_data.py raw_data.txt compressed_data.txt --show 20
'''
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


HEADER_RE = re.compile(r"#\s*BEGIN\b.*\bshape=(\d+)x(\d+)\b.*(?:\bdtype=([A-Za-z0-9_]+))?", re.IGNORECASE)
END_RE = re.compile(r"#\s*END\b")


@dataclass
class Block:
    key: str
    h: int
    w: int
    dtype: str  # e.g. 'uint16'
    data: np.ndarray  # shape (H, W), dtype uint16 (or int64 if无法解析就保持整数)


def _parse_key(header_line: str) -> str:
    # 从 "# BEGIN key=xxx ..." 中尽量取出 key= 值；取不到就返回空串
    m = re.search(r"\bkey=([^\s]+)", header_line)
    return m.group(1) if m else ""


def read_depth_txt(path: Path) -> List[Block]:
    """
    读取由 append_depth_arrays_to_txt 写出的文件：
      # BEGIN key=... shape=HxW dtype=uint16
      <H 行，每行 W 个整数>
      # END key=...
    返回 Block 列表（按出现顺序）。
    """
    blocks: List[Block] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        i += 1
        if not line.startswith("#"):
            continue
        m = HEADER_RE.search(line)
        if not m:
            continue

        H, W = int(m.group(1)), int(m.group(2))
        dtype = (m.group(3) or "uint16").lower()
        key = _parse_key(line)

        rows: List[np.ndarray] = []
        # 读取直到遇到 END
        while i < n:
            line2 = lines[i].rstrip("\n")
            i += 1
            if END_RE.match(line2):
                break
            if line2.startswith("#") or line2.strip() == "":
                continue
            row = np.fromstring(line2, sep=" ", dtype=np.int64)
            if row.size == 0:
                continue
            rows.append(row)

        if len(rows) != H:
            raise ValueError(f"{path}: 期望 {H} 行数据，实际 {len(rows)} 行（key={key}）")

        arr = np.stack(rows, axis=0)
        if arr.shape != (H, W):
            raise ValueError(f"{path}: 期望形状 {(H, W)}，实际 {arr.shape}（key={key}）")

        # 转目标 dtype（默认 uint16）
        if "uint16" in dtype.lower():
            arr = arr.astype(np.uint16, copy=False)
        else:
            # 保守：保持整数（如需要你可扩展其它 dtype）
            arr = arr.astype(np.int64, copy=False)

        blocks.append(Block(key=key, h=H, w=W, dtype=dtype, data=arr))
    return blocks


def compare_blocks(b1: Block, b2: Block, show: int = 10) -> Tuple[int, int]:
    """
    比较两块，返回 (mismatch_count, max_abs_diff)
    同时打印前 show 条差异 (i, j, v1, v2, diff)。
    """
    if (b1.h, b1.w) != (b2.h, b2.w):
        raise ValueError(f"形状不一致：{b1.h}x{b1.w} vs {b2.h}x{b2.w}（key1={b1.key}, key2={b2.key}）")

    # 转为有符号整型做差，避免 uint16 下溢
    a = b1.data.astype(np.int64, copy=False)
    b = b2.data.astype(np.int64, copy=False)
    diff = b - a
    mask = diff != 0
    count = int(mask.sum())
    max_abs = int(np.abs(diff).max()) if count > 0 else 0

    print(f"块对比  key1='{b1.key}'  key2='{b2.key}': 形状={b1.h}x{b1.w}, 不同点={count}, 最大绝对差={max_abs}")
    if count and show > 0:
        ys, xs = np.where(mask)
        take = min(show, count)
        print(f"  前 {take} 条差异 (y, x, v1, v2, diff)：")
        for k in range(take):
            y, x = int(ys[k]), int(xs[k])
            v1 = int(a[y, x])
            v2 = int(b[y, x])
            d = int(diff[y, x])
            print(f"    ({y:5d}, {x:5d}) : {v1:6d}  ->  {v2:6d}  (Δ={d:+d})")
    return count, max_abs


def main():
    ap = argparse.ArgumentParser(description="对比两个深度txt文件（按块、按像素），并打印第一个文件的最小/最大值。")
    ap.add_argument("file1", type=Path, help="第一个txt文件（基准）")
    ap.add_argument("file2", type=Path, help="第二个txt文件（对比）")
    ap.add_argument("--show", type=int, default=10, help="每块显示的差异条数（默认10，设为0可关闭）")
    args = ap.parse_args()

    f1, f2 = args.file1, args.file2
    if not f1.exists() or not f2.exists():
        raise FileNotFoundError("输入文件不存在。")

    blocks1 = read_depth_txt(f1)
    blocks2 = read_depth_txt(f2)

    if not blocks1:
        print(f"{f1} 无数据块。")
        return
    if not blocks2:
        print(f"{f2} 无数据块。")
        return

    # 遍历第一个文件，统计全局 min/max
    all_vals_1 = np.concatenate([b.data.reshape(-1) for b in blocks1])
    global_min_1 = int(all_vals_1.min())
    global_max_1 = int(all_vals_1.max())
    print(f"\n== 第一个文件全局统计 ==")
    print(f"块数：{len(blocks1)}")
    print(f"全局最小值：{global_min_1}")
    print(f"全局最大值：{global_max_1}")
    for i, b in enumerate(blocks1):
        print(f"  块 {i}: key='{b.key}', shape={b.h}x{b.w}, dtype={b.dtype}, "
              f"min={int(b.data.min())}, max={int(b.data.max())}")

    # 逐块比较（按出现顺序对齐）
    n = min(len(blocks1), len(blocks2))
    total_mismatch = 0
    global_max_abs = 0
    print(f"\n== 逐块对比（对齐前 {n} 块）==")
    for i in range(n):
        c, m = compare_blocks(blocks1[i], blocks2[i], show=args.show)
        total_mismatch += c
        global_max_abs = max(global_max_abs, m)

    # 如果两侧块数不同，提醒
    if len(blocks1) != len(blocks2):
        print(f"\n[注意] 两文件块数不同：{len(blocks1)} vs {len(blocks2)}。仅对比了前 {n} 块。")

    print(f"\n== 对比汇总 ==")
    print(f"总不同点：{total_mismatch}")
    print(f"全局最大绝对差：{global_max_abs}")


if __name__ == "__main__":
    main()
