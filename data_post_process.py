#!/usr/bin/env python3
"""
data_post_process.py

功能
------
1. 遍历 <root>/data/**/* .parquet
   - 读取为 pyarrow.Table，保持原始 schema
   - 用 observation.state 前 6 元素覆盖 action 前 6 元素，保留第 7 元素
   - 写出 *_updated.parquet；若 --overwrite 则原子覆写
2. 修改 <root>/meta/episodes_stats.jsonl
   - 用 observation.state 的 {min,max,mean,std} 前 6 项覆盖 action
   - 写出 episodes_stats.updated.jsonl；或 --overwrite 覆写
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------- Parquet 处理 ----------
def _patched_action_array(action_col: pa.ListArray,
                          obs_col: pa.ListArray) -> pa.ListArray:
    """
    返回新的 action 列：前 6 元素来自 obs，最后 1 元素保留原 action[6]
    保持元素类型、list 长度不变。
    """
    # 将两列解包成 Python list[ list[float] ]
    acts = action_col.to_pylist()
    obss = obs_col.to_pylist()

    # 构造新的 action 列
    new_act = [
        obs[:6] + [act[6]] for obs, act in zip(obss, acts)
    ]
    # 推断元素类型
    value_type = action_col.type.value_type
    return pa.array(new_act, type=pa.list_(value_type))  # List<float32>


def patch_one_parquet(src: Path, overwrite: bool):
    tbl = pq.read_table(src)
    if {"action", "observation.state"} - set(tbl.column_names):
        print(f"[SKIP] {src.relative_to(src.parents[2])} 缺 action/observation.state")
        return

    act_col = tbl["action"]
    obs_col = tbl["observation.state"]

    # 生成修补后的列
    new_act_col = _patched_action_array(act_col, obs_col)

    # 用新的列替换
    tbl = tbl.set_column(
        tbl.schema.get_field_index("action"),
        "action",
        new_act_col
    )

    # 写入
    if overwrite:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=src.parent, suffix=".tmp")
        os.close(tmp_fd)
        pq.write_table(tbl, tmp_path, compression="snappy")
        os.replace(tmp_path, src)          # 原子覆写
        tag = "覆写"
        out_path = src
    else:
        out_path = src.with_name(f"{src.stem}_updated{src.suffix}")
        pq.write_table(tbl, out_path, compression="snappy")
        tag = "写入"

    print(f"[OK ] {tag} {out_path.relative_to(src.parents[2])}")


def patch_parquet_tree(data_root: Path, overwrite: bool):
    for pf in data_root.rglob("*.parquet"):
        patch_one_parquet(pf, overwrite)


# ---------- stats 处理 ----------
def patch_episode_stats(meta: Path, overwrite: bool):
    # 先读入全部内容，避免覆写截断
    with meta.open() as fin:
        lines = list(fin)

    # 准备输出路径
    dst = meta if overwrite else meta.with_name("episodes_stats.updated.jsonl")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dst.parent, suffix=".tmp")
    os.close(tmp_fd)

    with open(tmp_path, "w") as fout:
        for line in lines:
            obj = json.loads(line)
            stats = obj["stats"]
            for key in ("min", "max", "mean", "std"):
                stats["action"][key][:6] = stats["observation.state"][key][:6]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_path, dst)  # 原子写
    tag = "覆写" if overwrite else "写入"
    print(f"[OK ] {tag} {dst.relative_to(meta.parent)}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Replace action[0:6] with observation.state[0:6] in all parquet "
                    "files and sync stats jsonl; use --overwrite to modify in-place.")
    ap.add_argument("root", help="数据集根目录，需包含 data/ 与 meta/")
    ap.add_argument("-o", "--overwrite", action="store_true",
                    help="直接覆写原 parquet / jsonl（会进行原子替换）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    data_root = root / "data"
    meta_file = root / "meta" / "episodes_stats.jsonl"
    if not data_root.is_dir() or not meta_file.is_file():
        sys.exit("[ERR] 路径不正确（需包含 data/ 和 meta/episodes_stats.jsonl）")

    patch_parquet_tree(data_root, args.overwrite)
    patch_episode_stats(meta_file, args.overwrite)


if __name__ == "__main__":
    main()
