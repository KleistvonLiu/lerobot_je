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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch
from datasets import tqdm
import matplotlib
matplotlib.use('Agg')  # 非GUI绘图后端
import matplotlib.pyplot as plt

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    aloha_agilex_follower,
)
from lerobot.configs import parser
from lerobot.utils.control_utils import predict_action, predict_action_v2
from lerobot.utils.utils import (
    init_logging, get_safe_torch_device,
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
    policy: PreTrainedConfig
    replay_mode: int = 0

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        logging.info(f"Loading policy from {policy_path}")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

def fix_image_axes(obs):
    for k, v in obs.items():
        if k.startswith("observation.images"):
            logging.info(v.shape)
            obs[k] = v.permute(0, 2, 3, 1).contiguous()
    return obs

@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    robot1 = make_robot_from_config(cfg.robot1)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    # logging.info(dataset.meta.tasks[0])
    merged_action_features = {**robot1.action_features}
    merged_observation_features = {**robot1.observation_features}
    action_features = hw_to_dataset_features(merged_action_features, "action")
    obs_features = hw_to_dataset_features(merged_observation_features, "observation")
    print(dataset.meta)
    print(dataset.hf_dataset.column_names)
    print(LeRobotDataset.__init__)
    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    device = get_safe_torch_device(policy.config.device)
    use_amp = policy.config.use_amp
    task = "pick up the brown pump and put it into the blue box"

    # ────────────── METRIC ACCUMULATORS ─────────────
    abs_error_sum  = 0   # L1 累加器
    sq_error_sum   = 0   # L2 累加器
    n_frames       = 0

    pred_actions = []  # list[Tensor] -> shape (action_dim,)
    gt_actions = []  # list[Tensor] -> shape (action_dim,)
    gt_observations = []  # list[Tensor] -> shape (action_dim,)

    for i in range(len(dataset)):  # 不要遍历 dataset.hf_dataset，而是遍历 dataset 本身
        sample = dataset[i]  # __getitem__ 会自动解码当帧图像
        idx = int(sample["index"])  # or any key you prefer

        # 只保留 observation 相关内容
        observation = {k: v for k, v in sample.items() if k.startswith("observation.")}
        # print(observation.keys())
        # for k, v in observation.items():
        #     print(f"{k}: {v.shape}")
        # exit(1)
        # fix_image_axes(observation)
        t0 = time.perf_counter()
        pred_action = predict_action_v2(observation, policy, device, use_amp, task=dataset.meta.tasks[0])
        logging.info(f"inference time cost: {time.perf_counter() - t0:.3f}")
        # 2) 取 Ground-Truth 动作
        gt_action = sample["action"]
        gt_observation = sample["observation.state"]
        logging.info(f"ground truth action: {gt_action}")
        logging.info(f"ground truth observation: {gt_observation}")
        logging.info(f"predicted: {pred_action}")

        # 2) 添加入缓存
        pred_actions.append(pred_action.detach().cpu())
        gt_actions.append(gt_action.detach().cpu())
        gt_observations.append(gt_observation.detach().cpu())

        abs_error_sum += (gt_action - pred_action).abs()
        sq_error_sum += (gt_action - pred_action).pow(2)
        n_frames += 1
    abs_error_sum /= n_frames
    sq_error_sum /= n_frames
    # ──────────────── SUMMARY ───────────────────────
    logging.info(f"Processed frames: {n_frames}")

    # 先算全局指标
    print("\n=== Global Action Error ===")
    print(f"MAE  (mean absolute): {abs_error_sum}")
    print(f"RMSE (root  mean sq): {sq_error_sum}")

    # ─────────── 3. 生成对比曲线 ────────────
    pred_all = torch.stack(pred_actions)  # (T, dof)
    gt_all = torch.stack(gt_actions)  # (T, dof)
    gt_ob = torch.stack(gt_observations)  # (T, dof)
    T, dof = pred_all.shape

    fig, axes = plt.subplots(dof, 1, figsize=(10, 2 * dof), sharex=True)

    for d in range(dof):
        ax = axes[d] if dof > 1 else axes
        ax.plot(gt_all[:, d], label="GT Action")
        ax.plot(gt_ob[:, d], label="GT Observation")
        ax.plot(pred_all[:, d], label="Pred")
        ax.set_ylabel(f"dim {d}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Predicted vs. Ground-Truth Actions")
    fig.tight_layout()
    plt.savefig("replay_plot.png")

    logging.info("Replay finished.")

if __name__ == "__main__":
    replay()
