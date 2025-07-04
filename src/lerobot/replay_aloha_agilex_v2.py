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
    robot2: RobotConfig
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

@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot1 = make_robot_from_config(cfg.robot1)
    robot2 = make_robot_from_config(cfg.robot2)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    replay_type = 'action' if cfg.replay_mode == 0 else 'observation.state'
    actions = dataset.hf_dataset.select_columns(replay_type)

    # 转换所有数据为 numpy 数组
    all_numpy_data = [tensor.numpy() for tensor in actions[replay_type]]

    robot1.connect()
    robot2.connect()

    # 逐帧播放
    last_action_array = all_numpy_data[0]
    step = 5
    step_time = step/dataset.fps

    for idx in range(len(all_numpy_data)-1):
        if idx == 0:
            logging.info("Start replaying episode.")
        if idx % step != 0:
            continue
        new_actions = np.linspace(last_action_array, all_numpy_data[idx+1], 1)  # 插值
        last_action_array = all_numpy_data[idx+1]
        for act in new_actions:
            start_episode_t = time.perf_counter()

            # 使用线程并行发送动作给 robot1 和 robot2
            def send_action_robot1():
                robot1.send_action_np(act[0:7])

            def send_action_robot2():
                robot2.send_action_np(act[7:14])

            # 创建两个线程
            thread1 = threading.Thread(target=send_action_robot1)
            thread2 = threading.Thread(target=send_action_robot2)

            # 启动线程
            thread1.start()
            thread2.start()

            # 等待两个线程完成
            thread1.join()
            thread2.join()

            dt_s = time.perf_counter() - start_episode_t
            # logging.info(f"<UNK>: {dt_s}")
            time.sleep(max(step_time - dt_s,0))
            # busy_wait(1 / dataset.fps - dt_s)

    robot1.disconnect()
    robot2.disconnect()


if __name__ == "__main__":
    replay()
