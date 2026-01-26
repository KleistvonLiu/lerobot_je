import json
import os
from typing import Dict, List


class TorqueFilter:
    def __init__(self, fs: float, tau_sec: float):
        dt = 1.0 / fs
        self.alpha = dt / (tau_sec + dt)
        self.y = None

    def update(self, x: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y


def filter_one_meta_file(meta_path: str, fs_est: float, tau_sec: float,
                         output_name: str = "meta_effort_filtered.jsonl") -> None:
    """
    对单个 episode 目录中的 meta.jsonl 做 effort 低通滤波，
    并在同目录下保存为 meta_effort_filtered.jsonl。
    """
    # 读取所有行
    with open(meta_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f if line.strip()]

    # 解析 JSON
    records = [json.loads(line) for line in raw_lines]

    # 对每个 topic 建一组 TorqueFilter（每一维一个 filter）
    topic_filters: Dict[str, List[TorqueFilter]] = {}

    for data in records:
        joints = data.get("joints", [])
        if not joints:
            continue

        for j in joints:
            effort = j.get("effort", None)
            if effort is None:
                continue

            topic = j.get("topic", "")

            # 第一次遇到这个 topic，初始化对应维度的滤波器
            if topic not in topic_filters:
                topic_filters[topic] = [
                    TorqueFilter(fs=fs_est, tau_sec=tau_sec)
                    for _ in range(len(effort))
                ]

            filters = topic_filters[topic]

            # 如果维度改变（一般不会），重新初始化
            if len(filters) != len(effort):
                topic_filters[topic] = [
                    TorqueFilter(fs=fs_est, tau_sec=tau_sec)
                    for _ in range(len(effort))
                ]
                filters = topic_filters[topic]

            # 对每一维做一阶低通滤波
            filtered_effort = []
            for i, x in enumerate(effort):
                y = filters[i].update(float(x))
                filtered_effort.append(y)

            # 替换为滤波后的结果
            j["effort"] = filtered_effort

    # 写出到新的 jsonl 文件
    out_path = os.path.join(os.path.dirname(meta_path), output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for data in records:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Saved filtered file: {out_path}")


def filter_all_episodes(root_dir: str, fs_est: float, tau_sec: float) -> None:
    """
    遍历 root_dir 下所有 episode_xxxxxx 目录，
    对其中的 meta.jsonl 做 effort 低通滤波，
    生成 meta_effort_filtered.jsonl。
    """
    for name in sorted(os.listdir(root_dir)):
        ep_dir = os.path.join(root_dir, name)
        if not os.path.isdir(ep_dir):
            continue

        meta_path = os.path.join(ep_dir, "meta.jsonl")
        if not os.path.isfile(meta_path):
            continue

        print(f"Processing: {meta_path}")
        filter_one_meta_file(meta_path, fs_est=fs_est, tau_sec=tau_sec)


if __name__ == "__main__":
    # 示例：根据你的目录修改这三项
    root_dir = "/media/kleist/NewNTFS1/test_1128_edited/"  # 根目录
    fs_est = 30.0    # 采样频率估计值（Hz）
    tau_sec = 0.15    # 时间常数（秒）

    filter_all_episodes(root_dir, fs_est=fs_est, tau_sec=tau_sec)
