import os
import h5py
import numpy as np
import time
from pathlib import Path
from lerobot.datasets.image_writer import AsyncImageWriter
import json
from PIL import Image

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class ReplayRobotFromExpanded:
    """
    一次性把所有图片和metadata.jsonl读入内存，极快模拟采集。
    """
    def __init__(self, episode_dir, camera_keys=None, jsonl_name="features.jsonl", meta=None):
        self.episode_dir = Path(episode_dir)
        self.camera_keys = camera_keys or self._find_camera_keys()
        self.jsonl_path = self.episode_dir / jsonl_name
        # 预读取所有 jsonl 行
        with open(self.jsonl_path, "r") as f:
            self.meta = [json.loads(line) for line in f]
        self.length = len(self.meta)
        # 一次性读入所有图片到buffer
        self.buffer = {cam: [] for cam in self.camera_keys}
        for cam in self.camera_keys:
            cam_dir = self.episode_dir / cam
            for frame_idx in range(self.length):
                img_path = cam_dir / f"frame_{frame_idx:06d}.png"
                img = np.array(Image.open(img_path))
                self.buffer[cam].append(img)
        self.idx = 0
        # 写死一个示例meta
        self._meta_schema = meta if meta is not None else {
            "observation.images.camera2_depth": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera1_depth": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera3_depth": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera0_depth": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera3": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera0": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera2": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.camera1": {"dtype": "image", "shape": [480, 640, 3]},
            "action": {"dtype": "float32", "shape": [7]},
            "timestamp": {"dtype": "float32", "shape": []},
            "frame_index": {"dtype": "int64", "shape": []},
            "episode_index": {"dtype": "int64", "shape": []},
            "index": {"dtype": "int64", "shape": []},
            "task_index": {"dtype": "int64", "shape": []},
            "observation": {
                "state": {"dtype": "float32", "shape": [7]},
                "tactile.right.tactile1": {"dtype": "int64", "shape": [4, 8]}
            }
        }

    def _find_camera_keys(self):
        # 自动检测所有相机子文件夹
        return [d.name for d in self.episode_dir.iterdir() if d.is_dir()]

    def get_observation(self):
        obs = {}
        frame_idx = self.idx
        for cam in self.camera_keys:
            obs[cam] = self.buffer[cam][frame_idx]
        # 其他 feature
        obs.update(self.meta[frame_idx])
        return obs

    def get_leader_action(self):
        # 假设 action 在 jsonl 里
        return self.meta[self.idx].get("action")

    def next(self):
        self.idx += 1
        if self.idx >= self.length:
            raise StopIteration

    def get_meta_schema(self):
        if self._meta_schema is not None:
            return self._meta_schema
        """
        返回 observation 的 schema，float统一为float32，int统一为int64。
        observation_*字段（除图像）全部包在observation下，图像字段在最外层。
        """
        def infer_shape(val):
            if isinstance(val, np.ndarray):
                return list(val.shape)
            elif isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return [0]
                sub_shape = infer_shape(val[0])
                if all(isinstance(x, (list, tuple, np.ndarray)) for x in val):
                    return [len(val)] + sub_shape
                else:
                    return [len(val)]
            else:
                return []

        def dtype_name(val):
            # 递归判断所有叶子节点类型
            def all_leaf_type(x, typ):
                if isinstance(x, (list, tuple)):
                    return len(x) > 0 and all(all_leaf_type(xx, typ) for xx in x)
                return isinstance(x, typ)
            if isinstance(val, (float, np.floating)):
                return "float32"
            elif isinstance(val, (int, np.integer)):
                return "int64"
            elif isinstance(val, np.ndarray):
                if np.issubdtype(val.dtype, np.floating):
                    return "float32"
                elif np.issubdtype(val.dtype, np.integer):
                    return "int64"
                else:
                    return str(val.dtype)
            elif isinstance(val, (list, tuple)):
                if all_leaf_type(val, float):
                    return "float32"
                elif all_leaf_type(val, int):
                    return "int64"
                else:
                    return "list"
            else:
                return type(val).__name__

        obs = self.get_observation()
        schema = {}
        obs_sub = {}
        for k, v in obs.items():
            # 图像字段（observation.images.* 或 camera_keys）在最外层
            is_image = False
            if k in self.camera_keys:
                is_image = True
            elif k.startswith("observation.images."):
                is_image = True
            if is_image:
                if isinstance(v, np.ndarray):
                    img_shape = list(v.shape)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
                    img_shape = list(v[0].shape)
                else:
                    img_shape = []
                schema[k] = {"dtype": "image", "shape": img_shape}
                continue
            # 其它 observation_* 字段全部包进 observation
            if k.startswith("observation."):
                key = k[len("observation."):]
                target = obs_sub
            else:
                key = k
                target = schema
            if isinstance(v, (np.ndarray, list, tuple)):
                target[key] = {
                    "dtype": dtype_name(v),
                    "shape": infer_shape(v)
                }
            elif isinstance(v, float):
                target[key] = {
                    "dtype": "float32",
                    "shape": []
                }
            elif isinstance(v, int):
                target[key] = {
                    "dtype": "int64",
                    "shape": []
                }
            elif isinstance(v, bool):
                target[key] = {
                    "dtype": "bool",
                    "shape": []
                }
            elif isinstance(v, str):
                target[key] = {
                    "dtype": "str",
                    "shape": []
                }
            else:
                target[key] = {
                    "dtype": str(type(v)),
                    "shape": []
                }
        if obs_sub:
            schema["observation"] = obs_sub
        return schema

def simulate_record_from_hdf5(
    robot,  # 直接传入已构造好的robot实例
    out_root,
    episode_idx=0,
    fps=30,
    num_image_writer_threads=12,
    num_image_writer_processes=0,
):
    meta_schema = robot.get_meta_schema()
    camera_keys = [k for k, v in meta_schema.items() if isinstance(v, dict) and v.get("dtype") == "image"]
    out_root = Path(out_root)
    ep_dir = out_root / f"episode_{episode_idx:06d}"
    ensure_dir(ep_dir)
    camera_dirs = {}
    for cam in camera_keys:
        cam_dir = ep_dir / "images" / cam
        ensure_dir(cam_dir)
        camera_dirs[cam] = cam_dir
    meta_path = ep_dir / "metadata.jsonl"
    meta_json_out_path = ep_dir / "meta.json"
    # 保存meta schema到输出目录
    with open(meta_json_out_path, "w") as meta_f:
        json.dump(meta_schema, meta_f, ensure_ascii=False, indent=2)
    image_writer = AsyncImageWriter(
        num_processes=num_image_writer_processes,
        num_threads=num_image_writer_threads
    )
    frame_idx = 0
    meta_buffer = []
    frame_times = []
    t1 = time.perf_counter()
    while True:
        t_start = time.perf_counter()
        try:
            obs = robot.get_observation()
            action = robot.get_leader_action()
        except Exception:
            break
        obs_for_json = dict(obs)
        # 保存图片到 images/ 子文件夹
        for cam in camera_keys:
            img = obs[cam]
            img_path = camera_dirs[cam] / f"frame_{frame_idx:06d}.png"
            image_writer.save_image(img, img_path)
            obs_for_json.pop(cam, None)
        obs_sub = {}
        for k in list(obs_for_json.keys()):
            is_image = False
            if k in camera_keys:
                is_image = True
            elif k.startswith("observation.images."):
                is_image = True
            if not is_image and (k.startswith("observation_") or k.startswith("observation.")):
                key = k.split("_", 1)[-1] if k.startswith("observation_") else k.split(".", 1)[-1]
                obs_sub[key] = obs_for_json.pop(k)
        if obs_sub:
            obs_for_json["observation"] = obs_sub
        obs_for_json["frame_idx"] = frame_idx
        obs_for_json["timestamp"] = time.time()
        meta_buffer.append(obs_for_json)
        frame_idx += 1
        try:
            robot.next()
        except StopIteration:
            break
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        print(f"[Frame {frame_idx-1}] 用时: {(t_end-t_start)*1000:.2f} ms")
        time.sleep(max(0, 1.0 / fps))
    # episode结束后统一写入元数据
    with open(meta_path, "a") as metaf:
        for data in meta_buffer:
            metaf.write(json.dumps(data, ensure_ascii=False) + "\n")
    image_writer.wait_until_done()
    image_writer.stop()
    t_wait_end = time.perf_counter()
    wait_time = t_wait_end - t1
    print(f"[INFO] 主进程等待异步写入耗时: {wait_time*1000:.2f} ms")
    # 帧率统计
    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        std = (sum((x - avg) ** 2 for x in frame_times) / len(frame_times)) ** 0.5
        print(f"平均每帧采集耗时: {avg*1000:.2f} ms, 标准差: {std*1000:.2f} ms, 帧数: {len(frame_times)}")
        print(f"理论帧间隔: {1000/fps:.2f} ms")
    print(f"模拟采集完成，episode {episode_idx}，共{frame_idx}帧，数据保存在 {ep_dir}")

if __name__=="__main__":
    # Example usage
    episode_dir = "/home/agx/jedata/frames_png/episode_000000"  # 传入单个 episode 目录
    out_root = "/home/agx/jedata/sim_recorded_dataset_out"
    episode_idx = 0
    fps = 30
    # 先创建robot实例
    robot = ReplayRobotFromExpanded(episode_dir)
    simulate_record_from_hdf5(
        robot=robot,
        out_root=out_root,
        episode_idx=episode_idx,
        fps=fps
    )

# 用法示例：
# simulate_record_from_hdf5(
#     hdf5_path="/path/to/lerobot_episode_21_expanded.hdf5",
#     out_root="/path/to/sim_recorded_dataset",
#     episode_idx=21,
#     fps=30,
#     cameras=["top_rgb", "right_wrist_rgb", "right_pole_rgb", "base_rgb"]
# )
