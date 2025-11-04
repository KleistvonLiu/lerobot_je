import streamlit as st
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import re

def get_episode_dirs(dataset_root):
    return sorted([p for p in Path(dataset_root).iterdir() if p.is_dir() and p.name.startswith('episode_')])

def get_meta_info(episode_dir):
    meta_path = Path(episode_dir) / "meta.jsonl"
    if not meta_path.exists():
        return 0, {}
    with open(meta_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    if not lines:
        return 0, {}
    # 统计帧数和属性
    frame_count = len(lines)
    first_meta = lines[0]
    return frame_count, first_meta

def flatten_numeric_keys(d, prefix=""):
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            full_k = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(flatten_numeric_keys(v, full_k))
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = np.array(v)
                if arr.ndim == 0:
                    keys.append(full_k)
                else:
                    for i in range(arr.shape[0]):
                        keys.append(f"{full_k}[{i}]")
            elif isinstance(v, (float, int)):
                keys.append(full_k)
    return keys

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

def get_attr(obj, path):
    # 兼容 joints[0].position[0] 这类路径
    parts = path.replace(']', '').replace('[', '.').split('.')
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            if isinstance(obj, list) and len(obj) > int(p):
                obj = obj[int(p)]
            else:
                return None
        else:
            if isinstance(obj, dict):
                obj = obj.get(p, None)
            else:
                return None
    return obj

def main():
    st.title("Sim Recorded Dataset 可视化编辑工具")
    st.markdown("---")
    dataset_root = st.text_input("数据集根目录", value="/path/to/sim_recorded_dataset")
    if not Path(dataset_root).exists():
        st.warning("请填写正确的数据集根目录")
        st.stop()
    episode_dirs = get_episode_dirs(dataset_root)
    st.write(f"共 {len(episode_dirs)} 个 episode")
    if not episode_dirs:
        st.stop()
    ep_names = [ep.name for ep in episode_dirs]
    ep_idx = st.selectbox("选择 episode", range(len(ep_names)), format_func=lambda i: ep_names[i])
    episode_dir = episode_dirs[ep_idx]
    frame_count, first_meta = get_meta_info(episode_dir)
    st.write(f"该 episode 有 {frame_count} 帧")
    # 预览帧 metadata
    if frame_count > 0:
        preview_idx = st.slider("预览帧 idx", min_value=0, max_value=frame_count-1, value=0)
        meta_path = Path(episode_dir) / "meta.jsonl"
        with open(meta_path, 'r') as f:
            lines = [json.loads(line) for line in f]
        st.write(f"第 {preview_idx} 帧 metadata:")
        st.json(lines[preview_idx])
        # 专门显示 tactile 字段（如果存在）
        if 'tactile' in lines[preview_idx]:
            st.write("tactile 数据:")
            st.json(lines[preview_idx]['tactile'])
        # 预览相机图片
        images_dir = Path(episode_dir) / "images"
        if images_dir.exists():
            with st.expander("预览该帧所有相机图片"):
                cam_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
                for cam_dir in cam_dirs:
                    img_path = cam_dir / f"frame_{preview_idx:06d}.png"
                    if img_path.exists():
                        st.image(str(img_path), caption=cam_dir.name, width='stretch')
                    else:
                        st.write(f"{cam_dir.name}: 无图片")

    st.markdown("---")
    # 删除 episode
    with st.expander("删除整个 episode"):
        if st.button(f"删除 {episode_dir.name}"):
            from edit_sim_recorded_dataset import delete_episode
            dataset_root_str = str(Path(episode_dir).parent)
            episode_idx = int(episode_dir.name.split('_')[1])
            delete_episode(dataset_root_str, episode_idx)
            st.success(f"已删除 {episode_dir} 并顺移后续编号")
            st.rerun()
    # 删除帧
    with st.expander("删除帧"):
        start_idx = st.number_input("起始帧 (0-based)", min_value=0, max_value=max(0, frame_count-1), value=0)
        num_frames = st.number_input("要删除的帧数", min_value=1, max_value=max(1, frame_count-int(start_idx)), value=1)
        if st.button("删除帧"):
            from edit_sim_recorded_dataset import delete_frames
            delete_frames(str(episode_dir), int(start_idx), int(num_frames))
            st.success(f"已删除 {episode_dir.name} 的帧 {start_idx} ~ {int(start_idx)+int(num_frames)-1}")
            st.rerun()
    # 插值属性
    with st.expander("插值属性"):
        attr_options, attr_shapes = flatten_numeric_attrs_and_shapes(first_meta) if first_meta else ([], {})
        # 拆分属性、列表索引、数组维度
        def split_attr_path(attr):
            parts = attr.split('[')
            base = parts[0]
            list_idx = None
            if len(parts) > 1:
                # Extract integer before any non-digit (e.g., '0].stamp_n')
                match = re.match(r'(\d+)', parts[1])
                if match:
                    list_idx = int(match.group(1))
            return base, list_idx
        base_attrs = sorted(set([split_attr_path(a)[0] for a in attr_options]))
        selected_base = st.selectbox("选择属性", base_attrs) if base_attrs else ""
        list_indices = sorted(set([split_attr_path(a)[1] for a in attr_options if split_attr_path(a)[0]==selected_base and split_attr_path(a)[1] is not None])) if selected_base else []
        selected_list_idx = st.selectbox("选择列表索引", list_indices) if list_indices else None
        # 组合属性路径
        attr_candidates = [a for a in attr_options if split_attr_path(a)[0]==selected_base and (split_attr_path(a)[1]==selected_list_idx if selected_list_idx is not None else True)]
        selected_attr = attr_candidates[0] if attr_candidates else ""
        dim_options = []
        if selected_attr:
            shape = attr_shapes.get(selected_attr, [])
            if shape and len(shape) > 0:
                dim_options = list(range(shape[0]))
        selected_dim = st.selectbox("选择维度", dim_options) if dim_options else None
        frame_idx = preview_idx if frame_count > 2 else 1
        if selected_attr:
            attr_path = f"{selected_attr}[{selected_dim}]" if selected_dim is not None else selected_attr
        else:
            attr_path = ""
        st.write(f"插值路径: {attr_path}")
        st.write(f"插值帧 idx: {frame_idx}")
        if st.button("插值该属性"):
            from edit_sim_recorded_dataset import interpolate_meta_attr
            interpolate_meta_attr(str(episode_dir), int(frame_idx), attr_path)
            st.success(f"已插值 {episode_dir.name} 第 {frame_idx} 帧的 {attr_path}")
            st.rerun()
    # 异常检测与修复（仅 joint/tactile）
    with st.expander("异常检测与修复（滑动均值法）"):
        allowed_top = [k for k in ['joint', 'joints', 'tactile'] if k in first_meta]
        if not allowed_top:
            st.warning("当前数据不包含 joint 或 tactile 字段")
        else:
            top_selected = st.selectbox("选择属性类型", allowed_top)
            # 新递归选择：字典递归下拉，遇到数组就停止
            # 优化递归选择：list先选索引再递归，只有遇到数值数组才显示维度
            def is_numeric_array(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    flat = np.array(val, dtype=object).flatten()
                    return all(isinstance(x, (int, float, np.integer, np.floating)) for x in flat)
                return False
            def recursive_select(obj, prefix=""):
                allowed_joint_keys = ["position", "velocity", "effort"]
                if isinstance(obj, dict):
                    # joint/joints下只允许 position/velocity/effort
                    if prefix.startswith("joint") or prefix.startswith("joints"):
                        keys = [k for k in obj.keys() if k in allowed_joint_keys]
                        if not keys:
                            st.warning(f"joint属性只支持: {allowed_joint_keys}")
                            return obj, prefix
                        selected = st.selectbox(f"选择 joint 属性 {prefix}", keys, key=prefix, index=0 if keys else None)
                        # 递归到 position/velocity/effort后停止
                        value = obj[selected]
                        if is_numeric_array(value):
                            return value, prefix + "." + selected if prefix else selected
                        else:
                            return recursive_select(value, prefix + "." + selected if prefix else selected)
                    else:
                        keys = list(obj.keys())
                        selected = st.selectbox(f"选择字典属性 {prefix}", keys, key=prefix, index=0 if keys else None)
                        return recursive_select(obj[selected], prefix + "." + selected if prefix else selected)
                elif isinstance(obj, list):
                    indices = list(range(len(obj)))
                    selected = st.selectbox(f"选择列表索引 {prefix}", indices, key=prefix, index=0 if indices else None)
                    return recursive_select(obj[selected], prefix + f"[{selected}]")
                elif is_numeric_array(obj):
                    # 数值数组，停止递归，主流程处理维度
                    return obj, prefix
                else:
                    return obj, prefix
            preview_obj = lines[0][top_selected]
            value, attr_path = recursive_select(preview_obj, top_selected)
            if is_numeric_array(value):
                arr = np.array(value)
                dim = None
                if arr.ndim > 0:
                    dim = st.selectbox("选择维度", list(range(arr.shape[0])), key=attr_path, index=0 if arr.shape[0] > 0 else None)
                    attr_path_full = f"{attr_path}[{dim}]"
                else:
                    attr_path_full = attr_path
            else:
                arr = value
                dim = None
                attr_path_full = attr_path
            window = st.number_input("滑动窗口大小", min_value=1, max_value=50, value=5)
            threshold = st.number_input("异常判定阈值（与滑动均值差的绝对值）", min_value=0.0, value=5.0)
            seq = []
            warn_once = False
            for meta in lines:
                frame_id = meta.get('frame_idx', lines.index(meta))
                if top_selected not in meta or meta[top_selected] is None:
                    if not warn_once:
                        st.warning(f"帧 {frame_id} 路径无有效数据: {attr_path}")
                        warn_once = True
                    seq.append(np.nan)
                    continue
                obj = meta[top_selected]
                # 只用下拉栏选择的完整路径索引，不做自动补齐
                obj = get_attr(obj, attr_path[len(top_selected):].strip('.')) if attr_path != top_selected else obj
                # 检查 obj 是否为数值数组
                if obj is None or not is_numeric_array(obj):
                    if not warn_once:
                        st.warning(f"帧 {frame_id} 路径无有效数据: {attr_path}")
                        warn_once = True
                    seq.append(np.nan)
                    continue
                arr = np.array(obj)
                val = None
                if dim is not None and arr.ndim > 0:
                    try:
                        v = arr[dim]
                        while isinstance(v, (list, np.ndarray)) and not isinstance(v, (int, float, np.integer, np.floating)):
                            v = v[0]
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            val = float(v)
                    except Exception as e:
                        if not warn_once:
                            st.warning(f"帧 {frame_id} 维度数据异常: {e}")
                            warn_once = True
                        val = None
                elif dim is not None and arr.ndim == 0:
                    if not warn_once:
                        st.warning(f"帧 {frame_id} 维度数据异常: 数据为标量，无法索引维度")
                        warn_once = True
                    val = None
                else:
                    try:
                        v = arr
                        while isinstance(v, (list, np.ndarray)) and not isinstance(v, (int, float, np.integer, np.floating)):
                            v = v[0]
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            val = float(v)
                    except Exception as e:
                        if not warn_once:
                            st.warning(f"帧 {frame_id} 数据异常: {e}")
                            warn_once = True
                        val = None
                if val is not None:
                    seq.append(val)
                else:
                    seq.append(np.nan)
            seq = np.array(seq)
            if len(seq) > 0:
                smooth = pd.Series(seq).rolling(window, center=True, min_periods=1).mean().to_numpy()
                diff = np.abs(seq - smooth)
                outlier_idx = np.where(diff > threshold)[0]
                st.line_chart({"原始": seq, "滑动均值": smooth})
                st.write(f"检测到 {len(outlier_idx)} 个异常点：", outlier_idx.tolist())
                if len(outlier_idx) > 0:
                    st.dataframe({"异常帧idx": outlier_idx, "原始值": seq[outlier_idx], "滑动均值": smooth[outlier_idx], "差值": diff[outlier_idx]})
                    if st.button("对所有异常点插值修复"):
                        from edit_sim_recorded_dataset import interpolate_meta_attr
                        for idx in outlier_idx:
                            if idx == 0 or idx == len(seq)-1:
                                continue
                            interpolate_meta_attr(str(episode_dir), int(idx), attr_path_full)
                        st.success(f"已对 {len(outlier_idx)} 个异常点插值修复")
                        st.rerun()

# streamlit run tools/edit_sim_recorded_dataset_gui.py
if __name__ == '__main__':
    main()

