# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Example:

```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```
"""
import datetime
import logging
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    aloha_agilex_follower,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot1: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    replay_mode: int = 0


def interpolate_action_array_rows(action_array: np.ndarray, num_interpolated_rows: int) -> np.ndarray:
    """
    对 action_array 中的每一行进行插值，生成更多的行。

    Args:
        action_array (np.ndarray): 输入的原始数据数组，形状为 (num_frames, num_features)
        num_interpolated_rows (int): 每两行之间插入的行数（目标插值后的行数）

    Returns:
        np.ndarray: 插值后的数据，形状为 (num_frames * (num_interpolated_rows - 1), num_features)
    """
    # 目标插值后的行数（每两行之间插入 num_interpolated_rows - 1 个数据点）
    interpolated_data = []

    for i in range(action_array.shape[0] - 1):  # 遍历所有行，最后一行不插值
        # 获取当前行和下一行
        row_start = action_array[i]
        row_end = action_array[i + 1]

        # 生成原始数据的索引（即当前行和下一行的索引）
        original_indices = np.linspace(0, 1, 2)

        # 生成插值的目标索引
        target_indices = np.linspace(0, 1, num_interpolated_rows)

        # 对每个特征列进行插值
        interpolated_rows = np.array([np.interp(target_indices, original_indices, [row_start[j], row_end[j]])
                                      for j in range(action_array.shape[1])]).T

        interpolated_data.append(interpolated_rows)

    # 将所有插值后的数据合并为一个数组
    return np.vstack(interpolated_data)

# === 工具函数：规范形状到 (480, 640, 1) ===
def _normalize_depth(
    arr: np.ndarray,
    target_hw=(480, 640),
    reduce3: str = "first",   # 可选: "first" | "mean" | "median"
    dtype=None,               # 若需要强制类型(如 np.uint16)，可传入；默认保持原 dtype
) -> np.ndarray:
    """
    将任何以下形状的深度数组规范成 (H, W, 1) 并返回 C-contiguous：
      - (H, W)
      - (W, H)
      - (H, W, 1)
      - (W, H, 1)
      - (3, H, W)     # CHW   ← 新增支持
      - (1, H, W)     # CHW
      - (H, W, 3)     # HWC   ← 顺便也兜一下
    当为 3 通道时，按 reduce3 聚合到单通道：
      - "first": 取第 0 通道
      - "mean" : 三通道取平均
      - "median": 三通道取中位数
    """
    Ht, Wt = target_hw
    a = np.asarray(arr)

    def _finish(x: np.ndarray) -> np.ndarray:
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return np.ascontiguousarray(x)

    # 2D: HW or WH
    if a.ndim == 2:
        h, w = a.shape
        if (h, w) == (Ht, Wt):
            return _finish(a[..., None])
        if (h, w) == (Wt, Ht):
            return _finish(a.T[..., None])
        raise ValueError(f"意外形状: {a.shape}，期望 {(Ht,Wt)} 或 {(Wt,Ht)}")

    # 3D: HWC / CHW / HW1 / WH1
    if a.ndim == 3:
        h, w = a.shape[:2]

        # HWC 情况
        if a.shape == (Ht, Wt, 1):
            return _finish(a)
        if a.shape == (Wt, Ht, 1):
            return _finish(np.transpose(a, (1, 0, 2)))

        if a.shape == (Ht, Wt, 3):
            # HWC 三通道 → 单通道
            if reduce3 == "first":
                one = a[..., 0]
            elif reduce3 == "mean":
                one = a.mean(axis=2)
            elif reduce3 == "median":
                one = np.median(a, axis=2)
            else:
                raise ValueError(f"未知 reduce3: {reduce3}")
            return _finish(one[..., None])

        if a.shape == (Wt, Ht, 3):
            a = np.transpose(a, (1, 0, 2))
            if reduce3 == "first":
                one = a[..., 0]
            elif reduce3 == "mean":
                one = a.mean(axis=2)
            elif reduce3 == "median":
                one = np.median(a, axis=2)
            else:
                raise ValueError(f"未知 reduce3: {reduce3}")
            return _finish(one[..., None])

        # CHW 情况： (C, H, W)
        if a.shape[0] in (1, 3) and a.shape[1:] == (Ht, Wt):
            C = a.shape[0]
            if C == 1:
                return _finish(a[0][..., None])  # (H,W) → (H,W,1)
            # C == 3
            if reduce3 == "first":
                one = a[0]
            elif reduce3 == "mean":
                one = a.mean(axis=0)
            elif reduce3 == "median":
                one = np.median(a, axis=0)
            else:
                raise ValueError(f"未知 reduce3: {reduce3}")
            return _finish(one[..., None])

        # CHW 但 H/W 颠倒： (C, W, H)
        if a.shape[0] in (1, 3) and a.shape[1:] == (Wt, Ht):
            a = a[:, ::].transpose(0, 2, 1)  # → (C, H, W)
            C = a.shape[0]
            if C == 1:
                return _finish(a[0][..., None])
            if reduce3 == "first":
                one = a[0]
            elif reduce3 == "mean":
                one = a.mean(axis=0)
            elif reduce3 == "median":
                one = np.median(a, axis=0)
            else:
                raise ValueError(f"未知 reduce3: {reduce3}")
            return _finish(one[..., None])

        # (H,W,? 不是 1/3) 或 (C,H,W 且 C 不是 1/3) 都视为不支持
        raise ValueError(f"不支持的 3D 深度形状: {a.shape}")

    raise ValueError(f"不支持的维度: {a.shape}")


# === 工具函数：把单帧写成文本块并“追加”到 txt ===
def _append_depth_block(txt_path: Path, key: str, depth_hw1: np.ndarray) -> None:
    H, W, _ = depth_hw1.shape
    # 方式A：你用的是 `import datetime`
    ts = datetime.datetime.now(datetime.timezone.utc) \
        .isoformat(timespec="milliseconds") \
        .replace("+00:00", "Z")
    img2d = depth_hw1.reshape(H, W)

    # 选择文本格式：整数用 %d，浮点用 %.6f
    if np.issubdtype(img2d.dtype, np.floating):
        fmt = "%.6f"
        dtype_name = img2d.dtype.name
    else:
        fmt = "%d"
        dtype_name = img2d.dtype.name

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"# BEGIN key={key} timestamp={ts} shape={H}x{W} dtype={dtype_name}\n")
        np.savetxt(f, img2d, fmt=fmt, delimiter=" ")
        f.write(f"# END key={key}\n")

@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # 输出文件：可按需修改；若 cfg 有自定义字段也可替换为 cfg.output_txt
    out_txt = Path(cfg.dataset.root) / f"{cfg.dataset.repo_id}_ep{cfg.dataset.episode}_camera_depth.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        # 逐相机读取；容错处理：不存在的键会跳过
        for cam in ("camera0", "camera1", "camera2"):
            key_path = f"observation.images.{cam}_depth"
            try:
                depth = dataset[idx][key_path]
            except KeyError:
                logging.warning("Sample %d missing key: %s", idx, key_path)
                continue

            # tensor -> numpy
            if hasattr(depth, "detach") and hasattr(depth, "cpu") and hasattr(depth, "numpy"):
                depth = depth.detach().cpu().numpy()
            elif hasattr(depth, "numpy"):
                depth = depth.numpy()

            # try:
            depth_hw1 = _normalize_depth(depth, target_hw=(480, 640))   # 若你的真实分辨率不同请改这里
            # except Exception as e:
            #     logging.warning("Normalize depth failed at idx=%d, %s: %r", idx, cam, e)
            #     continue

            # 追加写入文本块；把样本索引也写进 key 里更易追踪
            block_key = f"idx={idx}.{key_path}"
            _append_depth_block(out_txt, block_key, depth_hw1)

if __name__ == "__main__":
    replay()
