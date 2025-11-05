import os
import shutil
import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import re

def get_episode_dirs(dataset_root):
    """Return sorted list of episode directories (Path objects)."""
    return sorted([p for p in Path(dataset_root).iterdir() if p.is_dir() and p.name.startswith('episode_')])

def renumber_episodes(dataset_root, start_idx):
    """Renumber episodes after deleting one, so episode_xxxxxx is continuous, and update meta.jsonl episode_idx."""
    episode_dirs = get_episode_dirs(dataset_root)
    for ep in episode_dirs:
        ep_idx = int(ep.name.split('_')[1])
        if ep_idx > start_idx:
            new_idx = ep_idx - 1
            new_name = f"episode_{new_idx:06d}"
            # 修改 meta.jsonl 里的 episode_idx
            meta_path = ep / "meta.jsonl"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    lines = [json.loads(line) for line in f]
                for item in lines:
                    item['episode_idx'] = new_idx
                with open(meta_path, 'w') as f:
                    for item in lines:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            ep.rename(ep.parent / new_name)

def delete_episode(dataset_root, episode_idx):
    ep_dir = Path(dataset_root) / f"episode_{episode_idx:06d}"
    if not ep_dir.exists():
        print(f"Episode {ep_dir} does not exist.")
        return
    shutil.rmtree(ep_dir)
    renumber_episodes(dataset_root, episode_idx)
    print(f"Deleted {ep_dir} and renumbered subsequent episodes.")

def delete_frames(episode_dir, start_idx, num_frames):
    ep_dir = Path(episode_dir)
    meta_path = ep_dir / "meta.jsonl"
    if not meta_path.exists():
        print(f"No metadata.jsonl in {ep_dir}")
        return
    # 1. 读meta
    with open(meta_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    end_idx = start_idx + num_frames
    frame_indices = set(range(start_idx, end_idx))
    # 2. 删除指定帧
    keep = [i for i in range(len(lines)) if i not in frame_indices]
    new_lines = [lines[i] for i in keep]
    # 3. 顺移frame_idx和frame_index，并递归替换图片索引
    def recursive_update_img_idx(obj, old_idx, new_idx):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = recursive_update_img_idx(v, old_idx, new_idx)
        elif isinstance(obj, list):
            return [recursive_update_img_idx(v, old_idx, new_idx) for v in obj]
        elif isinstance(obj, str):
            pattern = rf"frame_{old_idx:06d}\.png"
            new_name = f"frame_{new_idx:06d}.png"
            return re.sub(pattern, new_name, obj)
        return obj
    for new_idx, item in enumerate(new_lines):
        item['frame_idx'] = new_idx
        if 'frame_index' in item:
            item['frame_index'] = new_idx
        # 递归替换所有 frame_xxxxxx.png
        old_idx = keep[new_idx]
        item = recursive_update_img_idx(item, old_idx, new_idx)
        new_lines[new_idx] = item
    # 4. 删除图片
    images_dir = ep_dir / "images"
    if images_dir.exists():
        for cam_dir in images_dir.iterdir():
            if cam_dir.is_dir():
                for idx in frame_indices:
                    img_path = cam_dir / f"frame_{idx:06d}.png"
                    if img_path.exists():
                        img_path.unlink()
                # 重命名剩余图片
                imgs = sorted(cam_dir.glob('frame_*.png'))
                for new_idx, img in enumerate(imgs):
                    img.rename(cam_dir / f"frame_{new_idx:06d}.png")
    # 5. 写回meta
    with open(meta_path, 'w') as f:
        for item in new_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Deleted frames from {start_idx} to {end_idx-1} in {ep_dir} and updated metadata/images.")

def get_attr(obj, path):
    parts = path.replace(']', '').replace('[', '.').split('.')
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = obj[p]
    return obj

def set_attr(obj, path, value):
    parts = path.replace(']', '').replace('[', '.').split('.')
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = obj[p]
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        obj[last] = value

def flatten_numeric_attrs_and_shapes(d, prefix=""):
    attrs = []
    shapes = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full_k = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                sub_attrs, sub_shapes = flatten_numeric_attrs_and_shapes(v, full_k)
                attrs.extend(sub_attrs)
                shapes.update(sub_shapes)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    sub_attrs, sub_shapes = flatten_numeric_attrs_and_shapes(item, f"{full_k}[{i}]")
                    attrs.extend(sub_attrs)
                    shapes.update(sub_shapes)
            elif isinstance(v, (tuple, np.ndarray)):
                arr = np.array(v)
                attrs.append(full_k)
                if arr.ndim == 0:
                    shapes[full_k] = []
                else:
                    shapes[full_k] = list(arr.shape)
            elif isinstance(v, (float, int)):
                attrs.append(full_k)
                shapes[full_k] = []
    return attrs, shapes

def interpolate_meta_attr(episode_dir, frame_idx, attr_path):
    ep_dir = Path(episode_dir)
    meta_path = ep_dir / "meta.jsonl"
    with open(meta_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    if frame_idx <= 0 or frame_idx >= len(lines) - 1:
        print("插值帧必须在中间，不能是首帧或末帧")
        return
    prev = get_attr(lines[frame_idx-1], attr_path)
    next_ = get_attr(lines[frame_idx+1], attr_path)
    prev_np = np.array(prev)
    next_np = np.array(next_)
    interp = ((prev_np + next_np) / 2).tolist() if prev_np.shape else float((prev_np + next_np) / 2)
    set_attr(lines[frame_idx], attr_path, interp)
    with open(meta_path, 'w') as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Interpolated {attr_path} at frame {frame_idx} in {ep_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Edit sim_recorded_dataset episodes/frames/meta.")
    subparsers = parser.add_subparsers(dest='command')

    # 删除episode
    p_del_ep = subparsers.add_parser('delete-episode')
    p_del_ep.add_argument('dataset_root')
    p_del_ep.add_argument('episode_idx', type=int)

    # 删除帧
    p_del_frame = subparsers.add_parser('delete-frames')
    p_del_frame.add_argument('episode_dir')
    p_del_frame.add_argument('start_idx', type=int)
    p_del_frame.add_argument('num_frames', type=int)

    # 插值meta属性
    p_interp = subparsers.add_parser('interpolate-meta')
    p_interp.add_argument('episode_dir')
    p_interp.add_argument('frame_idx', type=int)
    p_interp.add_argument('attr_path', type=str, help="如 observation.state[2] 或 action[0]")

    args = parser.parse_args()
    if args.command == 'delete-episode':
        delete_episode(args.dataset_root, args.episode_idx)
    elif args.command == 'delete-frames':
        delete_frames(args.episode_dir, args.start_idx, args.num_frames)
    elif args.command == 'interpolate-meta':
        interpolate_meta_attr(args.episode_dir, args.frame_idx, args.attr_path)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
