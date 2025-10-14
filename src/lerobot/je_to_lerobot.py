import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import json
from pathlib import Path
from PIL import Image

def convert_expand_to_lerobot(
    episode_dir,
    lerobot_root,
    repo_id="dragonbobo-no3/lerobot_dataset",
    fps=30,
    use_videos=True,
    task="test_task",
):
    t0 = time.time()
    episode_dir = Path(episode_dir)
    # 读取 metadata.jsonl
    features_path = episode_dir / "metadata.jsonl"
    with open(features_path, "r") as f:
        features_list = [json.loads(line) for line in f]
    batch_size = len(features_list)
    print(f"[INFO] metadata.jsonl 读取完成，帧数: {batch_size}")
    # 自动检测相机
    camera_names = [d.name for d in episode_dir.iterdir() if d.is_dir()]
    print(f"[INFO] 检测到相机: {camera_names}")
    # 读取所有图片
    images_dict = {}
    for cam in camera_names:
        cam_dir = episode_dir / cam
        imgs = []
        for i in range(batch_size):
            img_path = cam_dir / f"frame_{i:06d}.png"
            img = np.array(Image.open(img_path))
            imgs.append(img)
        images_dict[cam] = imgs
    # 构造 features，只保留 action、observation 前缀和相机字段
    features = {}
    for k, v in features_list[0].items():
        if not (k.startswith("action") or k.startswith("observation") or k in camera_names):
            continue
        arr = np.array(v)
        dtype = str(arr.dtype)
        if dtype.startswith("float"):
            dtype = "float32"
        shape = list(arr.shape)
        # 修正：标量 int64/float32 字段 shape 必须为 [1]
        if dtype in ("int64", "float32") and (shape == [] or shape is None):
            shape = [1]
        features[k] = {"dtype": dtype, "shape": shape}
    for cam in camera_names:
        img_shape = images_dict[cam][0].shape
        features[cam] = {
            "dtype": "video" if use_videos else "image",
            "shape": list(img_shape),
            "names": ["height", "width", "channels"]
        }
    print(f"[INFO] 自动构造 features: {features}")
    # 初始化 LerobotDataset
    t2 = time.time()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=lerobot_root,
        use_videos=use_videos,
    )
    t3 = time.time()
    print(f"[INFO] LerobotDataset 初始化完成，耗时: {t3-t2:.3f}s")
    # 从文件夹名提取 episode_index
    episode_index = int(episode_dir.name.split('_')[-1])
    # 构造 episode_data，显式赋值主要字段
    episode_data = {
        "size": batch_size,
        "action": [f["action"] for f in features_list],
        "frame_index": list(range(batch_size)),
        "timestamp": [float(i) / fps for i in range(batch_size)],
        "task": [task] * batch_size,
        "episode_index": episode_index,  # int
        "task_index": [0] * batch_size,
        "index": list(range(batch_size)),
    }
    # 自动添加 observation 前缀的所有字段（已存在的不重复）
    for k in features_list[0].keys():
        if k.startswith("observation.") and k not in episode_data:
            episode_data[k] = [f[k] for f in features_list]
    # 自动添加所有相机字段
    for cam in camera_names:
        cam_dir = episode_dir / cam
        img_paths = [str((cam_dir / f"frame_{i:06d}.png")) for i in range(batch_size)]
        episode_data[cam] = img_paths
    print("[INFO] episode_data 字段:")
    for k, v in episode_data.items():
        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
            print(f"  {k}: list, shape[0]={v[0].shape}, len={len(v)}")
        else:
            print(f"  {k}: {type(v)}, len={len(v) if hasattr(v,'__len__') else 'N/A'}")
    # 建立软链接到 LeRobotDataset 期望的 images 目录结构
    for cam in camera_names:
        # 目标目录: {lerobot_root}/images/{cam}/episode_{episode_index:06d}/
        target_dir = Path(lerobot_root) / "images" / cam / f"episode_{episode_index:06d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        src_dir = episode_dir / cam
        for i in range(batch_size):
            src_img = src_dir / f"frame_{i:06d}.png"
            dst_img = target_dir / f"frame_{i:06d}.png"
            if not dst_img.exists():
                try:
                    os.symlink(src_img, dst_img)
                except FileExistsError:
                    pass
    # 存储 episode
    t4 = time.time()
    dataset.save_episode(episode_data=episode_data, encode_videos=True)
    t5 = time.time()
    print(f"[INFO] Saved lerobot episode to {lerobot_root}，耗时: {t5-t4:.3f}s")

if __name__ == "__main__":
    # 示例用法
    lerobot_root = "/home/agx/jedata/wrapper"
    episode_dir = "/home/agx/jedata/sim_recorded_dataset/episode_00000"
    convert_expand_to_lerobot(episode_dir, lerobot_root)