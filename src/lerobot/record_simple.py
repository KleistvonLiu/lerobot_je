import os
import cv2
import json
import time
from pathlib import Path
from queue import Queue
import threading
from lerobot.datasets.image_writer import AsyncImageWriter

# 假设有如下接口可用
# robot1.get_observation() -> dict, 其中包含 {camera_name: image(np.ndarray), ...}
# robot1.get_leader_action() -> dict

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def record_simple(
    robot1,
    dataset_root,
    num_episodes=10,
    episode_time_s=6,
    fps=30,
    cameras=None,
    num_image_writer_threads=4,
    num_image_writer_processes=0,
):
    dataset_root = Path(dataset_root)
    image_writer = AsyncImageWriter(
        num_processes=num_image_writer_processes,
        num_threads=num_image_writer_threads
    )

    for ep in range(num_episodes):
        ep_dir = dataset_root / f"episode_{ep:05d}"
        ensure_dir(ep_dir)
        camera_dirs = {}
        for cam in cameras:
            cam_dir = ep_dir / cam
            ensure_dir(cam_dir)
            camera_dirs[cam] = cam_dir
        meta_path = ep_dir / "metadata.jsonl"

        start_t = time.perf_counter()
        frame_idx = 0
        meta_buffer = []  # 缓存所有帧的元数据
        while time.perf_counter() - start_t < episode_time_s:
            obs = robot1.get_observation()  # 假设返回 {camera_name: image, ...}
            action = robot1.get_leader_action()  # 假设返回 dict

            # 保存图片
            for cam in cameras:
                img = obs[cam]
                img_path = camera_dirs[cam] / f"frame_{frame_idx:05d}.png"
                image_writer.save_image(img, img_path)

            # 缓存元数据
            meta = {
                "frame_idx": frame_idx,
                "timestamp": time.time(),
                "state": {k: v for k, v in obs.items() if k not in cameras},
                "action": action,
            }
            meta_buffer.append(meta)

            frame_idx += 1
            time.sleep(max(0, 1.0 / fps))

        # episode结束后统一写入元数据
        with open(meta_path, "a") as metaf:
            for meta in meta_buffer:
                metaf.write(json.dumps(meta, ensure_ascii=False) + "\n")

        print(f"Episode {ep} done, {frame_idx} frames.")

    image_writer.wait_until_done()
    image_writer.stop()
    print("All done.")

# 用法示例（需替换robot1和cameras为你的实际对象和相机名列表）
# record_simple(robot1, "dataset_root", num_episodes=10, episode_time_s=6, fps=30, cameras=["cam1", "cam2"])