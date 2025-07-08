#!/usr/bin/env python3
# rename_and_patch.py
# ------------------------------------------------------------
# 把数据集里最新 N 个 episode 改名并修改 parquet 内部 episode_index
# 依赖: pyarrow >= 8.0
# ------------------------------------------------------------
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

EP_RE_TEMPLATE = "episode_{:06d}"
PARQUET_SUFFIX = ".parquet"
MP4_SUFFIX = ".mp4"


# --------------------------- parquet patch helper --------------------------- #
def _rep_column(value: int, length: int, dtype: pa.DataType) -> pa.Array:
    """Return an Arrow array filled with `value`."""
    try:
        return pa.repeat(pa.scalar(value, dtype), length)
    except AttributeError:
        return pa.array([value] * length, type=dtype)


def patch_episode_index_to_file(
    src: Path,
    dst: Path,
    new_value: int,
) -> None:
    """
    Read `src` parquet, replace episode_index column with `new_value`,
    write to `dst` (could be same as src).
    """
    table = pq.read_table(src)
    schema = table.schema
    col_idx = schema.get_field_index("episode_index")
    epi_dtype = schema.field(col_idx).type
    n_rows = table.num_rows

    patched_table = table.set_column(
        col_idx, "episode_index", _rep_column(new_value, n_rows, epi_dtype)
    )

    # 保留原列压缩格式（若读取失败则退化为 snappy）
    col_comp = {}
    try:
        md = pq.read_metadata(src)
        for i in range(md.num_columns):
            col_comp[schema.names[i]] = md.column(i).compression.lower()
    except Exception:
        col_comp = {name: "snappy" for name in schema.names}

    pq.write_table(
        patched_table,
        dst,
        compression=col_comp,
        version="2.6",
        data_page_version="2.0",
        row_group_size=n_rows,           # 保持 1 个 row-group
        use_dictionary=False,
    )


# --------------------------- rename helpers --------------------------------- #
def scan_episode_numbers(files: List[Path]) -> List[int]:
    return sorted(int(p.stem.split("_")[1]) for p in files)


def build_mapping(existing: List[int], new_indices: List[int]) -> Dict[int, int]:
    if len(existing) < len(new_indices):
        raise RuntimeError("现有 episode 数不足以重命名！")
    old = existing[-len(new_indices):]  # 取最后 N 个
    if len(set(new_indices)) != len(new_indices):
        raise ValueError("new_indices 含重复值！")
    if any(x in existing[:-len(new_indices)] for x in new_indices):
        raise ValueError("new_indices 与数据集中较早的 episode 冲突！")
    return dict(zip(old, new_indices))


def rename_mp4(video_root: Path, mapping: Dict[int, int]) -> None:
    pattern = str(video_root / "**" / f"episode_*{MP4_SUFFIX}")
    for path_str in glob.glob(pattern, recursive=True):
        p = Path(path_str)
        old_num = int(p.stem.split("_")[1])
        if old_num in mapping:
            new_name = EP_RE_TEMPLATE.format(mapping[old_num]) + MP4_SUFFIX
            p.rename(p.with_name(new_name))


def patch_jsonl(path: Path, mapping: Dict[int, int], patch_stats: bool = False) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            ep = obj.get("episode_index")
            if ep in mapping:
                obj["episode_index"] = mapping[ep]
                if patch_stats and "stats" in obj and "episode_index" in obj["stats"]:
                    for k in obj["stats"]["episode_index"]:
                        obj["stats"]["episode_index"][k] = [mapping[ep]]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------- main ------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Rename and patch latest episodes.")
    parser.add_argument("root", type=Path, help="Dataset root")
    parser.add_argument("new_indices", type=int, nargs="+", help="New episode_index values")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    data_dir = root / "data" / "chunk-000"
    video_root = root / "videos"
    meta_dir = root / "meta"

    parquet_files = sorted(data_dir.glob(f"episode_*{PARQUET_SUFFIX}"))
    existing = scan_episode_numbers(parquet_files)
    mapping = build_mapping(existing, args.new_indices)
    print("Mapping (old ➔ new):", mapping)

    # --- 1. 处理 parquet (重写文件 + 改名) ------------------------------------
    for old, new in mapping.items():
        src = data_dir / f"{EP_RE_TEMPLATE.format(old)}{PARQUET_SUFFIX}"
        dst = data_dir / f"{EP_RE_TEMPLATE.format(new)}{PARQUET_SUFFIX}"
        if args.dry_run:
            print(f"[dry] patch {src.name} -> {dst.name}")
            continue
        patch_episode_index_to_file(src, dst, new)
        src.unlink()  # 删除原文件

    # --- 2. 处理视频 ----------------------------------------------------------
    rename_mp4(video_root, mapping)

    # --- 3. 处理 meta JSONL ---------------------------------------------------
    if not args.dry_run:
        patch_jsonl(meta_dir / "episodes.jsonl", mapping, patch_stats=False)
        patch_jsonl(meta_dir / "episodes_stats.jsonl", mapping, patch_stats=True)

    print("✅ Done.")


if __name__ == "__main__":
    main()
