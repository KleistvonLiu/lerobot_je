import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import json
from pathlib import Path
from PIL import Image

def convert_expand_to_lerobot_batch(
    episodes_root,
    lerobot_root,
    task="test_task",
    repo_id="dragonbobo-no3/lerobot_dataset",
    fps=30,
    use_videos=True,
    batch_encode_num=5,  # 新增参数，控制每次并发编码多少个episode
):
    t0 = time.time()
    episodes_root = Path(episodes_root)
    # 自动查找所有 episode 目录
    episode_dirs = sorted([d for d in episodes_root.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"[INFO] 共检测到 {len(episode_dirs)} 个episode: {[d.name for d in episode_dirs]}")
    dataset = None
    first_ep_idx = None
    last_ep_idx = None
    for idx, episode_dir in enumerate(episode_dirs):
        # 读取 meta.jsonl
        features_path = episode_dir / "meta.jsonl"
        with open(features_path, "r") as f:
            features_list = [json.loads(line) for line in f]
        batch_size = len(features_list)
        print(f"[INFO] [{episode_dir.name}] meta.jsonl 读取完成，帧数: {batch_size}")
        # 自动检测 images 下的相机目录为相机名（兼容 manager_node 产出命名）
        images_root = episode_dir / "images"
        camera_names = [d.name for d in images_root.iterdir() if d.is_dir()]
        print(f"[INFO] 检测到相机: {camera_names}")
        # 处理可能的多一层目录结构（例如 images/camera_x/camera_x/frame_000000.png）
        cam_dirs = {}
        for cam in camera_names:
            cam_root = images_root / cam
            # 直接存在帧文件则使用该目录
            first_frame = cam_root / f"frame_{0:06d}.png"
            if first_frame.exists():
                cam_dirs[cam] = cam_root
                continue
            # 否则在子目录中查找第一个包含 frame_*.png 的目录
            found = None
            for p in cam_root.iterdir():
                if p.is_dir() and any(p.glob('frame_*.png')):
                    found = p
                    break
            if found is None:
                # 最后尝试递归查找任何 frame_*.png 的父目录
                matches = list(cam_root.rglob('frame_*.png'))
                if matches:
                    found = matches[0].parent
            cam_dirs[cam] = found if found is not None else cam_root
        # 只读取每个相机的第一帧图片用于推断 shape，无需全部加载
        img_shapes = {}
        for cam in camera_names:
            cam_dir = cam_dirs[cam]
            img_path = cam_dir / f"frame_{0:06d}.png"
            # 使用 PIL 获取尺寸和通道数，但不要把图片读成 numpy.ndarray
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
            if mode == 'L':
                channels = 1
            elif mode == 'RGBA':
                channels = 4
            else:
                channels = 3
            img_shapes[cam] = (h, w, channels)
        # 构造 features，自动处理 joints -> observation.state & action
        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        if dataset is None:
            features = {}
            # 如果 frames 中包含 joints，则构造 observation.state 与 action
            if len(features_list) > 0 and "joints" in features_list[0] and features_list[0]["joints"]:
                # 计算总关节数（所有 joint entries 的 position 长度之和）
                first_joints = features_list[0]["joints"]
                total_joints = sum(len(j.get("position", [])) for j in first_joints)
                # observation.state 包含 positions, velocities, efforts 串联
                features["observation.state"] = {"dtype": "float32", "shape": [total_joints * 3]}
                # action 使用所有 joint 的 position 串联
                features["action"] = {"dtype": "float32", "shape": [total_joints]}
            # 如果存在 observation dict，兼容展开其他字段
            if len(features_list) > 0 and "observation" in features_list[0]:
                obs_flat = flatten_dict(features_list[0]["observation"], parent_key="observation")
                for k, v in obs_flat.items():
                    arr = np.array(v)
                    dtype = str(arr.dtype)
                    if dtype.startswith("float"):
                        dtype = "float32"
                    shape = list(arr.shape)
                    if dtype in ("int64", "float32") and (shape == [] or shape is None):
                        shape = [1]
                    features[k] = {"dtype": dtype, "shape": shape}
            # 相机字段
            for cam in camera_names:
                img_shape = img_shapes[cam]
                features[cam] = {
                    "dtype": "video" if use_videos else "image",
                    "shape": list(img_shape),
                    "names": ["height", "width", "channels"],
                }
            print(f"[INFO] 自动构造 features: {features}")
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=lerobot_root,
                use_videos=use_videos,
            )
        # 从文件夹名提取 episode_index
        episode_index = int(episode_dir.name.split('_')[-1])
        if first_ep_idx is None:
            first_ep_idx = episode_index
        last_ep_idx = episode_index
        # 构造 frames 列表，每帧一个 dict，包含所有必需字段
        frames = []
        for i in range(batch_size):
            meta = {
                "episode_index": episode_index,
                "frame_index": i,
                "index": i,  # 通常 index == frame_index
                "timestamp": float(i) / fps,
                "task_index": 0,  # 默认 0，可根据需要调整
                "task": task,
            }
            # 相机图片字段（如 camera_01_color_image_raw）
            for cam in camera_names:
                cam_dir = cam_dirs.get(cam, images_root / cam)
                img_path = cam_dir / f"frame_{i:06d}.png"
                meta[cam] = str(img_path)
            # joints/tactiles（如有）——将 joints 的 position/velocity/effort 拼接到 observation.state；action 为 position
            joints_list = features_list[i].get("joints", []) if i < len(features_list) else []
            if joints_list:
                positions = []
                velocities = []
                efforts = []
                for j in joints_list:
                    pos = list(j.get("position", []))
                    vel = list(j.get("velocity", []))
                    eff = list(j.get("effort", []))
                    positions.extend(pos)
                    velocities.extend(vel)
                    efforts.extend(eff)
                # observation.state = [positions..., velocities..., efforts...]
                obs_state = np.array(positions + velocities + efforts, dtype=np.float32)
                act = np.array(positions, dtype=np.float32)
                meta["observation.state"] = obs_state
                meta["action"] = act
            # tactiles（如有）
            if "tactiles" in features_list[i]:
                meta["tactiles"] = features_list[i]["tactiles"]
            # 兼容原有 observation/action 字段
            if "action" in features_list[i] and "action" not in meta:
                meta["action"] = features_list[i]["action"]
            if "observation" in features_list[i]:
                obs_flat = flatten_dict(features_list[i]["observation"], parent_key="observation")
                meta.update(obs_flat)
            frames.append(meta)
        # 为 save_episode 构造 episode_data（使用 frames 列表包装并包含 size/task/episode_index）
        # frames 已构造好，接下来把 frames 转换为 save_episode 接受的 episode_data（按 feature 聚合）
        # 保持相机字段为图片路径字符串，交由 dataset / compute_stats 在需要时读取。
        # 不要在这里把路径读成 numpy.ndarray（会导致后续 PIL.open 报错）。
        # 使用 dataset.meta.features 来决定需要哪些 keys
        feat_keys = list(dataset.meta.features.keys())
        episode_data = {
            "size": batch_size,
            "task": [task] * batch_size,
            "episode_index": episode_index,
        }
        # 为每个 feature 生成按帧的列表
        for key in feat_keys:
            # episode_index 是 scalar（dataset 要求），不要覆盖
            if key == "episode_index":
                continue
            vals = []
            for f in frames:
                if key in f:
                    vals.append(f[key])
                else:
                    # 补默认值
                    if key == "timestamp":
                        vals.append(f.get("timestamp", float(0)))
                    elif key == "frame_index":
                        vals.append(f.get("frame_index", None))
                    elif key == "index":
                        vals.append(f.get("index", None))
                    elif key == "task_index":
                        vals.append(0)
                    else:
                        vals.append(None)
            episode_data[key] = vals

        # Ensure task_index field exists as a per-frame list
        if "task_index" not in episode_data:
            episode_data["task_index"] = [0] * batch_size

        # 存储 episode，不编码视频
        t4 = time.time()
        dataset.save_episode(episode_data=episode_data, encode_videos=False)
        t5 = time.time()
        print(f"[INFO] Saved episode {episode_index} to {lerobot_root}，耗时: {t5-t4:.3f}s")
    # 分批顺序编码视频
    print(f"[INFO] 开始分批编码视频，每批 {batch_encode_num} 个episode")
    for start in range(first_ep_idx, last_ep_idx + 1, batch_encode_num):
        end = min(start + batch_encode_num, last_ep_idx + 1)
        print(f"[INFO] 批量编码: episodes {start} ~ {end - 1}")
        batch_t0 = time.time()
        # 只为本批次 episode 建软链接
        for ep_idx in range(start, end):
            ep_dir = [d for d in episode_dirs if int(d.name.split('_')[-1]) == ep_idx][0]
            images_root = ep_dir / "images"
            # 获取所有相机目录（不要局限前缀），并处理可能的多层嵌套
            camera_names = [d.name for d in images_root.iterdir() if d.is_dir()]
            cam_dirs = {}
            for cam in camera_names:
                cam_root = images_root / cam
                first_frame = cam_root / f"frame_{0:06d}.png"
                if first_frame.exists():
                    cam_dirs[cam] = cam_root
                    continue
                found = None
                for p in cam_root.iterdir():
                    if p.is_dir() and any(p.glob('frame_*.png')):
                        found = p
                        break
                if found is None:
                    matches = list(cam_root.rglob('frame_*.png'))
                    if matches:
                        found = matches[0].parent
                cam_dirs[cam] = found if found is not None else cam_root

            for cam in camera_names:
                target_src_dir = cam_dirs.get(cam, images_root / cam)
                # 将软链接创建到 <lerobot_root>/images/<cam>/episode_xxxxxx（视频编码器期望的目录结构）
                target_dir = Path(lerobot_root) / "images" / cam / f"episode_{ep_idx:06d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                for img_file in target_src_dir.glob("frame_*.png"):
                    dst_img = target_dir / img_file.name
                    if not dst_img.exists():
                        try:
                            os.symlink(img_file, dst_img)
                        except FileExistsError:
                            pass
        # 编码
        dataset.batch_encode_videos(start, end)
        batch_t1 = time.time()
        print(f"[INFO] 本批次编码耗时: {batch_t1-batch_t0:.2f}s")
    print(f"[INFO] 批量编码完成，总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    # 示例用法
    lerobot_root = "/home/agx/jedata/wrapper"
    episodes_root = "/home/agx/jedata/manager_node_temp"  # 传入包含多个episode_xxxxxx的根目录
    task = "test_task"
    convert_expand_to_lerobot_batch(episodes_root, lerobot_root, task)