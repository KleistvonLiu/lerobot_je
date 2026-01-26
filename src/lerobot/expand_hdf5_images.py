import os
import h5py
import numpy as np
import cv2
import sys
import time
import json
import shutil
from pathlib import Path
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import lerobot.datasets.lerobot_dataset as lerobot_dataset

def decode_jpg_array(jpg_list, height=480, width=640):
    imgs = []
    for jpg_bytes in jpg_list:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def expand_episode_images(hdf5_path, out_path, height=480, width=640):
    with h5py.File(hdf5_path, 'r') as f:
        imgs_grp = f['observations/images']
        expanded = {}
        for cam in ['top_rgb', 'right_wrist_rgb', 'right_pole_rgb', 'base_rgb']:
            jpg_list = [imgs_grp[cam][i].tobytes() for i in range(len(imgs_grp[cam]))]
            expanded[cam] = decode_jpg_array(jpg_list, height, width)
    with h5py.File(out_path, 'w') as f_out:
        for cam, arr in expanded.items():
            f_out.create_dataset(cam, data=arr, compression='gzip')
    print(f"Expanded images saved to {out_path}")


def main():
    repo_id = "lerobot/test"
    root = "/home/agx/jedata/test_0927"
    export_root = "/home/agx/jedata/frames_png/"
    os.makedirs(export_root, exist_ok=True)
    t_load0 = time.time()
    dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root)
    t_load1 = time.time()
    print(f"[TIME] 加载dataset: {t_load1-t_load0:.3f}s")
    # 加载info.json
    info_path = os.path.join(root, "meta/info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    feature_keys = list(info["features"].keys())
    # 找出所有相机key
    camera_keys = [k for k in feature_keys if info["features"][k]["dtype"] == "image"]
    other_keys = [k for k in feature_keys if info["features"][k]["dtype"] != "image"]

    episode_indices = [item["episode_index"].item() for item in dataset.hf_dataset]
    unique_episode_ids = sorted(set(episode_indices))

    for eid in unique_episode_ids:
        t0 = time.time()
        episode = [i for i, ep_idx in enumerate(episode_indices) if ep_idx == eid]
        # 创建episode文件夹
        ep_dir = Path(export_root) / f"episode_{eid:06d}"
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_dir.mkdir(parents=True)
        # 创建相机子文件夹
        cam_dirs = {cam: ep_dir / cam for cam in camera_keys}
        for d in cam_dirs.values():
            d.mkdir(parents=True)
        # jsonl文件
        jsonl_path = ep_dir / "features.jsonl"
        t_prepare = time.time()
        img_times = []
        json_times = []
        with open(jsonl_path, "w") as jsonl_f:
            t_loop_start = time.time()
            for t, idx in enumerate(episode):
                step = dataset[idx]
                # 保存图片
                t_img_start = time.time()
                for cam in camera_keys:
                    img = step.get(cam)
                    if img is None:
                        shape = info["features"][cam]["shape"]
                        img = np.zeros(shape, dtype=np.uint8)
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)
                    img_pil = Image.fromarray(img)
                    img_path = cam_dirs[cam] / f"frame_{t:06d}.png"
                    img_pil.save(img_path)
                t_img_end = time.time()
                img_times.append(t_img_end - t_img_start)
                # 保存其他feature到jsonl
                t_json_start = time.time()
                feat = {}
                for k in other_keys:
                    val = step.get(k)
                    if val is None:
                        shape = info["features"][k]["shape"]
                        dtype = info["features"][k]["dtype"]
                        if dtype == "float32":
                            val = np.zeros(shape, dtype=np.float32)
                        elif dtype == "int64":
                            val = np.zeros(shape, dtype=np.int64)
                        else:
                            val = np.zeros(shape)
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    feat[k] = val
                jsonl_f.write(json.dumps(feat, ensure_ascii=False) + "\n")
                t_json_end = time.time()
                json_times.append(t_json_end - t_json_start)
            t_loop_end = time.time()
        t1 = time.time()
        img_avg = np.mean(img_times) * 1000 if img_times else 0
        img_std = np.std(img_times) * 1000 if img_times else 0
        json_avg = np.mean(json_times) * 1000 if json_times else 0
        json_std = np.std(json_times) * 1000 if json_times else 0
        print(f"Episode {eid} done, frames saved to {ep_dir}")
        print(f"[TIME] 预处理: {t_prepare-t0:.3f}s, 主循环: {t_loop_end-t_loop_start:.3f}s, 总: {t1-t0:.3f}s")
        print(f"[TIME] 单帧图片: {img_avg:.2f}±{img_std:.2f}ms, 单帧json: {json_avg:.2f}±{json_std:.2f}ms, 帧数: {len(img_times)}")

if __name__ == '__main__':
    main()
