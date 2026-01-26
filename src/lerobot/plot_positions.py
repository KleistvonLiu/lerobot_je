#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘画出所有的je数据集的关节位置曲线

python3 ./src/lerobot/plot_positions.py /media/kleist/NewNTFS1/test_1112 --start-episode-idx 313

python3 ./src/lerobot/plot_positions.py /media/kleist/NewNTFS1/test_1128

"""
import argparse
import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def find_meta_files(root: str) -> List[str]:
    """
    在 root 目录下递归查找所有 meta.jsonl 文件，并按路径排序。
    典型结构：
        root/
          episode_000000/meta.jsonl
          episode_000001/meta.jsonl
          ...
    """
    meta_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "meta.jsonl" in filenames:
            meta_paths.append(os.path.join(dirpath, "meta.jsonl"))
    meta_paths.sort()
    return meta_paths


def load_positions_from_meta(meta_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单个 meta.jsonl 文件中读取 frame_index 和 7 维 position。

    返回：
        frame_indices: shape [N]
        positions:     shape [N, 7]
    """
    frame_indices = []
    positions: List[List[float]] = []

    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            frame_idx = data.get("frame_index", None)
            # 如果没有 frame_index，就用当前长度代替
            if frame_idx is None:
                frame_idx = len(frame_indices)
            frame_indices.append(frame_idx)

            joints_list = data.get("joints", [])
            if not joints_list:
                # 没有 joints 数据就跳过这一帧
                continue

            # 如果有多个 joints 条目，优先选 topic 为 /robot/joint_states 的，否则用第一个
            joint_entry: Optional[dict] = None
            for j in joints_list:
                if j.get("topic") == "/robot/joint_states":
                    joint_entry = j
                    break
            if joint_entry is None:
                joint_entry = joints_list[0]

            pos = joint_entry.get("position", [])
            if len(pos) < 7:
                # 不是 7 维就跳过，避免画图出错
                continue

            positions.append(pos[:7])

    if not positions:
        raise ValueError(f"{meta_path} 中没有有效的 position 数据")

    frame_indices_arr = np.array(frame_indices[: len(positions)], dtype=np.int64)
    positions_arr = np.array(positions, dtype=np.float64)
    return frame_indices_arr, positions_arr


def plot_episode(frame_indices: np.ndarray, positions: np.ndarray, title: str) -> None:
    """
    画出单个 episode 的 7 维 position 曲线。
    每个维度单独一个子图，7 个子图在同一个窗口中。

    x 轴：frame_index
    y 轴：对应 joint 的 position
    """
    num_joints = positions.shape[1]
    assert num_joints == 7, f"期望 7 维关节，实际为 {num_joints}"

    # 创建 7 行 1 列的子图，共享 x 轴
    fig, axes = plt.subplots(
        num_joints, 1, sharex=True, figsize=(10, 2.0 * num_joints)
    )

    # 防止 num_joints == 1 时 axes 不是列表
    if num_joints == 1:
        axes = [axes]

    for j in range(num_joints):
        ax = axes[j]
        ax.plot(frame_indices, positions[:, j])
        ax.set_ylabel(f"joint{j}")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("frame_index")

    # 总标题
    fig.suptitle(title, fontsize=14)
    # 给总标题留点空间
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 阻塞式显示，窗口关闭后函数返回
    plt.show()
    plt.close(fig)


def extract_episode_idx_from_path(meta_path: str) -> Optional[int]:
    """
    从路径中解析 episode_idx，假设目录名类似 'episode_000002'。

    例如：
        /xxx/episode_000002/meta.jsonl -> 2
    """
    episode_dir = os.path.basename(os.path.dirname(meta_path))
    if not episode_dir.startswith("episode_"):
        return None
    try:
        idx_str = episode_dir.split("_", 1)[1]
        return int(idx_str)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "遍历数据集目录，依次弹窗显示每个 episode 的 7 维关节 position 曲线"
            "（一个窗口 7 个子图），支持从指定 episode_idx 开始处理。"
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="数据集根目录，例如：/media/kleist/NewNTFS1/test_1112",
    )
    parser.add_argument(
        "--start-episode-idx",
        type=int,
        default=None,
        help=(
            "只处理 episode_idx >= 该值 的 episode，例如 2 表示只从 episode_000002 开始；"
            "默认不限制，从最小的 episode 开始。"
        ),
    )
    args = parser.parse_args()
    root = args.root
    start_episode_idx = args.start_episode_idx

    meta_files = find_meta_files(root)
    if not meta_files:
        print(f"在目录 {root} 下没有找到任何 meta.jsonl 文件")
        return

    print(f"共找到 {len(meta_files)} 个 episode 的 meta.jsonl 文件")
    if start_episode_idx is not None:
        print(f"将从 episode_idx >= {start_episode_idx} 的 episode 开始处理")

    for idx, meta_path in enumerate(meta_files):
        episode_idx_from_path = extract_episode_idx_from_path(meta_path)

        # 如果设置了起始 episode_idx 且当前 episode 小于该值，则跳过
        if (
            start_episode_idx is not None
            and episode_idx_from_path is not None
            and episode_idx_from_path < start_episode_idx
        ):
            print(
                f"[{idx+1}/{len(meta_files)}] 跳过: {meta_path} "
                f"(episode_idx={episode_idx_from_path} < {start_episode_idx})"
            )
            continue

        print(f"[{idx+1}/{len(meta_files)}] 处理: {meta_path}")
        try:
            frame_indices, positions = load_positions_from_meta(meta_path)
        except ValueError as e:
            print(f"  跳过：{e}")
            continue

        # 从 meta.jsonl 的第一行里取 episode_idx，当作标题的一部分
        episode_idx_str = "unknown"
        try:
            with open(meta_path, "r") as f:
                first_line = f.readline()
                if first_line:
                    first_data = json.loads(first_line)
                    episode_idx_str = str(first_data.get("episode_idx", "unknown"))
        except Exception:
            # 如果从内容里没读到，就退回到路径解析出来的
            if episode_idx_from_path is not None:
                episode_idx_str = str(episode_idx_from_path)

        title = f"Episode {episode_idx_str} ({os.path.dirname(meta_path)})"
        plot_episode(frame_indices, positions, title)

    print("符合条件的 episode 已全部显示完毕。")


if __name__ == "__main__":
    main()
