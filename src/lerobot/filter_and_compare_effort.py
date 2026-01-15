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
    joint_names = None   # 关节名字

    target_effort = "effort_filtered" # "effort"

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
            effort = j0.get(target_effort, None)
            if effort is None:
                continue

            if joint_names is None:
                joint_names = j0.get(
                    "name",
                    [f"joint{i+1}" for i in range(len(effort))]
                )

            efforts_list.append(effort)

    if not efforts_list:
        print("没有读取到有效的 effort 数据，请检查 episode_idx 和文件内容。")
        return

    efforts = np.array(efforts_list)   # 形状: (T, 7)
    T, D = efforts.shape
    print(f"读取到 {T} 帧 effort, 维度 = {D}")

    # ===== 2. 估算采样频率 fs（也可以直接设成 100.0） =====
    timestamps = np.array(timestamps)
    if len(timestamps) > 1:
        dt = np.diff(timestamps).mean()
        fs_est = 1.0 / dt
    else:
        fs_est = 100.0  # 只有一帧的话就用默认值

    print(f"估算采样频率 fs ≈ {fs_est:.2f} Hz")

    # 你给的示例是 fs=100, tau_sec=0.02，这里可以按你需要覆盖:
    fs = fs_est          # 如果你想固定 100Hz，改成：fs = 100.0
    tau_sec = 0.15       # 20 ms 时间常数

    # ===== 3. 对每一维 effort 使用 TorqueFilter 做一阶低通滤波 =====
    efforts_filtered = np.zeros_like(efforts)

    # 为每个关节单独建一个滤波器（内部有状态 y）
    filters = [TorqueFilter(fs=fs, tau_sec=tau_sec) for _ in range(D)]

    for t in range(T):
        for i in range(D):
            x = float(efforts[t, i])
            y = filters[i].update(x)
            efforts_filtered[t, i] = y

    # ===== 4. 画图：一个 figure，7 个子图，对比原始 vs 滤波后 =====
    x = np.arange(T)

    if joint_names is None or len(joint_names) != D:
        joint_names = [f"joint{i+1}" for i in range(D)]

    fig, axes = plt.subplots(D, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"Episode {target_episode_idx} Effort 一阶低通滤波对比\n"
        f"(fs ≈ {fs:.1f} Hz, tau = {tau_sec*1000:.0f} ms)",
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
