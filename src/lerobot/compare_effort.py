import json

import matplotlib.pyplot as plt
import numpy as np


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


def main():
    # ===== 需要你根据实际路径修改 =====
    jsonl_path = "/media/kleist/Lenovo Y910/log/episode_000000/meta.jsonl"
    target_episode_idx = 39  # 只处理 episode_idx == 39 的数据

    timestamps = []      # 保存每一帧的 timestamp
    efforts_list = []    # 保存每一帧的 7 维 effort
    efforts_filtered_list = []    # 保存每一帧的 7 维 effort
    joint_names = None   # 关节名字

    # ===== 1. 读取 jsonl 文件，筛选指定 episode 的 effort 数据 =====
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            timestamps.append(data["timestamp"])

            joints = data.get("joints", [])
            if not joints:
                continue

            j0 = joints[0]
            effort = j0.get("effort", None)
            effort_filtered = j0.get("effort_filtered", None)
            if effort is None or effort_filtered is None:
                continue

            if joint_names is None:
                joint_names = j0.get(
                    "name",
                    [f"joint{i+1}" for i in range(len(effort))]
                )

            efforts_list.append(effort)
            efforts_filtered_list.append(effort_filtered)

    if not efforts_list or not efforts_filtered_list:
        print("没有读取到有效的 effort 数据，请检查 episode_idx 和文件内容。")
        return

    efforts = np.array(efforts_list)   # 形状: (T, 7)
    efforts_filtered = np.array(efforts_filtered_list)   # 形状: (T, 7)
    T, D = efforts.shape
    print(f"读取到 {T} 帧 effort, 维度 = {D}")

    # ===== 4. 画图：一个 figure，7 个子图，对比原始 vs 滤波后 =====
    x = np.arange(T)

    if joint_names is None or len(joint_names) != D:
        joint_names = [f"joint{i+1}" for i in range(D)]

    fig, axes = plt.subplots(D, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"Episode {target_episode_idx} Effort 一阶低通滤波对比\n",
        fontsize=14
    )

    for i in range(D):
        ax = axes[i]
        ax.plot(x, efforts[:, i], label="original effort", alpha=0.6)
        ax.plot(x, efforts_filtered[:, i], label="filtered effort", linewidth=2)
        ax.set_ylabel(joint_names[i])
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("frame index")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


if __name__ == "__main__":
    main()
