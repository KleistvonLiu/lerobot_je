import os
import h5py
import numpy as np
import cv2
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_images_from_folder(images_root, cam_name, eid, num_frames):
    imgs = []
    for idx in range(num_frames):
        img_path = os.path.join(images_root, f"{cam_name}_ep{eid}_{idx:05d}.png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        imgs.append(img)
    return imgs

def main():
    export_root = "/home/agx/jedata/hdf5_expanded_image/"
    images_root = os.path.join(export_root, "images")
    lerobot_root = "/home/agx/jedata/wrapper_from_image/"
    # os.makedirs(lerobot_root, exist_ok=True)
    repo_id = "dragonbobo-no3/lerobot_dataset"
    fps = 30
    use_videos = True
    task = "test_task"

    # 遍历所有 meta.hdf5 文件
    for fname in os.listdir(export_root):
        print(f"[INFO] 处理文件: {fname}")
        if not fname.endswith("_meta.hdf5"):
            continue
        eid = int(fname.split("_")[2])
        hdf5_path = os.path.join(export_root, fname)
        t0 = time.time()
        with h5py.File(hdf5_path, 'r') as f:
            states = f['states'][...]
            actions = f['actions'][...]
        t1 = time.time()
        print(f"[TIME] 读取 states/actions: {t1-t0:.3f}s")
        batch_size = states.shape[0]
        # 读取图片
        cam_names = ["top_rgb", "right_wrist_rgb", "right_pole_rgb", "base_rgb"]
        images_dict = {}
        t2 = time.time()
        for cam in cam_names:
            imgs = load_images_from_folder(images_root, cam, eid, batch_size)
            images_dict[cam] = np.stack(imgs, axis=0)  # (N, H, W, 3)
        t3 = time.time()
        print(f"[TIME] 读取所有图片: {t3-t2:.3f}s")
        # 构造 features
        t4 = time.time()
        joint_num = actions.shape[1]
        joint_prefix = "right.joint"
        joint_suffix = ".pos"
        action_names = [f"{joint_prefix}{i}{joint_suffix}" for i in range(joint_num)]
        state_names = [f"{joint_prefix}{i}{joint_suffix}" for i in range(joint_num)]
        features = {
            "action": {"dtype": "float32", "shape": [joint_num], "names": action_names},
            "observation.state": {"dtype": "float32", "shape": [joint_num], "names": state_names},
        }
        for cam in cam_names:
            img_shape = images_dict[cam][0].shape
            features[f"observation.images.{cam}"] = {
                "dtype": "video" if use_videos else "image",
                "shape": list(img_shape),
                "names": ["height", "width", "channels"]
            }
        t5 = time.time()
        print(f"[TIME] 构造 features: {t5-t4:.3f}s")
        # 初始化 LeRobotDataset
        t6 = time.time()
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=lerobot_root,
            use_videos=use_videos,
        )
        t7 = time.time()
        print(f"[TIME] 初始化 LeRobotDataset: {t7-t6:.3f}s")
        # 构造 episode_data
        t8 = time.time()
        episode_data = {
            "size": batch_size,
            "action": [actions[i] for i in range(batch_size)],
            "observation.state": [states[i] for i in range(batch_size)],
            "frame_index": list(range(batch_size)),
            "timestamp": [float(i) / fps for i in range(batch_size)],
            "task": [task] * batch_size,
            "task_index": [0] * batch_size,
            "index": list(range(batch_size)),
        }
        for cam in cam_names:
            # (N, H, W, 3) -> (N, C, H, W)
            episode_data[f"observation.images.{cam}"] = [np.transpose(images_dict[cam][i], (2, 0, 1)) for i in range(batch_size)]
        t9 = time.time()
        print(f"[TIME] 构造 episode_data: {t9-t8:.3f}s")
        print(f"[INFO] episode_data 字段:")
        for k, v in episode_data.items():
            if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
                print(f"  {k}: list, shape[0]={v[0].shape}, len={len(v)}")
            else:
                print(f"  {k}: {type(v)}, len={len(v) if hasattr(v,'__len__') else 'N/A'}")
        t10 = time.time()
        dataset.save_episode(episode_data=episode_data, encode_videos=True)
        t11 = time.time()
        print(f"[TIME] save_episode: {t11-t10:.3f}s")
        print(f"[INFO] Saved lerobot episode {eid} to {lerobot_root}")

if __name__ == "__main__":
    main()
