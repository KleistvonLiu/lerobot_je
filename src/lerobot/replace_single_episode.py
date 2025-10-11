#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配合check_data.py使用
check_data.py检测哪些数据有问题，最简单的方法就是直接使用其他数据集替换，a是被替换的index，b是替换的index
python replace_episode.py --root /home/kleist/Documents/Database/test_0928_100_v2 \
  --a 3 --b 1 --chunk chunk-000 --dry-run
"""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

def copy_file(src: Path, dst: Path, dry: bool):
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[DRY] copy {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")

def backup_file(p: Path, dataset_root: Path, backup_root: Path, dry: bool):
    if not p.exists():
        return
    rel = p.relative_to(dataset_root)   # 修正：以数据集根目录为基准
    bak = backup_root / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[DRY] backup {p} -> {bak}")
    else:
        shutil.copy2(p, bak)
        print(f"backup  {p} -> {bak}")

def load_jsonl(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"JSONL 不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl_atomic(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def replace_meta_episode(jsonl_path: Path, a: int, b: int, backup_root: Path, dataset_root: Path, dry: bool):
    rows = load_jsonl(jsonl_path)
    row_a_idx = next((i for i, r in enumerate(rows) if int(r.get("episode_index")) == a), None)
    row_b_idx = next((i for i, r in enumerate(rows) if int(r.get("episode_index")) == b), None)
    if row_b_idx is None:
        raise ValueError(f"{jsonl_path.name}: 找不到 episode_index={b}")

    if row_a_idx is None:
        insert_pos = 0
        for i, r in enumerate(rows):
            if int(r.get("episode_index")) > a:
                break
            insert_pos = i + 1
        new_row = json.loads(json.dumps(rows[row_b_idx]))
        new_row["episode_index"] = a
        new_rows = rows[:insert_pos] + [new_row] + rows[insert_pos:]
    else:
        new_rows = list(rows)
        new_row = json.loads(json.dumps(rows[row_b_idx]))
        new_row["episode_index"] = a
        new_rows[row_a_idx] = new_row

    if dry:
        print(f"[DRY] 将 {jsonl_path.name} 中 episode_index={a} 用 {b} 的记录替换/插入")
    else:
        backup_file(jsonl_path, dataset_root, backup_root, dry=False)
        save_jsonl_atomic(jsonl_path, new_rows)
        print(f"updated {jsonl_path.name}: episode {a} ← {b}")

def main():
    ap = argparse.ArgumentParser(description="把索引 a 的内容全部用索引 b 的内容替换")
    ap.add_argument("--root", required=True, help="数据集根目录（包含 data/, videos/, meta/）")
    ap.add_argument("--a", type=int, required=True, help="目标 episode 索引（被覆盖）")
    ap.add_argument("--b", type=int, required=True, help="来源 episode 索引（作为模板）")
    ap.add_argument("--chunk", default="chunk-000", help="分块目录名，默认 chunk-000")
    ap.add_argument("--no-backup", action="store_true", help="不做备份（默认会在 .backup_replace_episode 下备份）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不执行")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    chunk = args.chunk
    a, b = args.a, args.b

    data_dir = root / "data" / chunk
    videos_dir = root / "videos" / chunk
    meta_dir = root / "meta"

    ep_a = f"episode_{a:06d}"
    ep_b = f"episode_{b:06d}"

    backup_root = root / ".backup_replace_episode" / (datetime.now().strftime("%Y%m%d_%H%M%S") if not args.no_backup else "NO_BACKUP")

    print(f"ROOT: {root}")
    print(f"操作: {a} <- {b}  （覆盖 A 的内容为 B）")
    print(f"dry-run: {args.dry_run}, backup: {not args.no_backup}")

    # 1) parquet
    src_parquet = data_dir / f"{ep_b}.parquet"
    dst_parquet = data_dir / f"{ep_a}.parquet"
    if not args.no_backup:
        backup_file(dst_parquet, root, backup_root, dry=args.dry_run)
    copy_file(src_parquet, dst_parquet, dry=args.dry_run)

    # 2) videos
    if videos_dir.exists():
        for cam_dir in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
            src_mp4 = cam_dir / f"{ep_b}.mp4"
            dst_mp4 = cam_dir / f"{ep_a}.mp4"
            if src_mp4.exists():
                if not args.no_backup:
                    backup_file(dst_mp4, root, backup_root, dry=args.dry_run)
                copy_file(src_mp4, dst_mp4, dry=args.dry_run)
            else:
                print(f"[WARN] 源视频缺失（跳过）: {src_mp4}")
    else:
        print(f"[WARN] 未找到视频目录: {videos_dir}")

    # 3) meta
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        path = meta_dir / name
        replace_meta_episode(path, a, b, backup_root, root, dry=args.dry_run)

    print("完成。建议检查：")
    print(f"  ls {data_dir}/{ep_a}.parquet")
    print(f"  ls {videos_dir}/**/{ep_a}.mp4")
    print(f"  grep '\"episode_index\": {a}' {meta_dir}/episodes*.jsonl")

if __name__ == "__main__":
    main()