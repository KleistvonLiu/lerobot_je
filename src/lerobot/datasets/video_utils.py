#!/usr/bin/env python

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
import glob
import importlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Union
import subprocess

import os
os.environ["SVT_LOG"] = "1"   # 小心：必须在 import av 之前
import av
import pyarrow as pa
import numpy as np
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")

def encode_video_frames_depth_image(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
    depth_image: bool = False,   # 新增：深度图无损模式（MKV+FFV1+gray16le）
) -> None:
    """Encode a sequence of images into a video.

    - Normal mode (depth_image=False): h264/hevc/libsvtav1 with chosen pix_fmt.
    - Depth mode   (depth_image=True): MKV container + FFV1 codec + gray16le (lossless 16-bit).
    """

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=True if overwrite else True)

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir} matching {template}.")

    # Determine size from the first frame
    first = Image.open(input_list[0])
    width, height = first.size

    # Set logging level for PyAV/FFmpeg
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    # -------------------------------
    # Depth (lossless) path: MKV+FFV1
    # -------------------------------
    if depth_image:
        # Ensure MKV container
        if video_path.suffix.lower() != ".mkv":
            logging.warning(
                f"Depth mode requires MKV container; changing suffix to .mkv: {video_path}"
            )
            video_path = video_path.with_suffix(".mkv")

        # Open output & add FFV1 stream (lossless), 16-bit gray
        with av.open(str(video_path), mode="w") as output:
            # options: level=3 是常用选择；不设置也可
            output_stream = output.add_stream("ffv1", rate=fps, options={"level": "3"})
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "gray16le"

            for p in input_list:
                # 读取单通道 16-bit（PNG I;16）
                img = Image.open(p)
                arr = np.array(img, dtype=np.uint16)

                # 形状校验：必须是 (H,W) 单通道
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                if arr.ndim != 2:
                    raise ValueError(
                        f"Depth image must be single-channel; got shape {arr.shape} for {p}"
                    )

                # 保证 C 连续 & dtype 正确
                arr = np.ascontiguousarray(arr, dtype=np.uint16)

                # 创建帧（gray16le）
                frame = av.VideoFrame.from_ndarray(arr, format="gray16le")
                packet = output_stream.encode(frame)
                if packet:
                    output.mux(packet)

            # flush
            packet = output_stream.encode(None)
            if packet:
                output.mux(packet)

        # Reset logging
        if log_level is not None:
            av.logging.restore_default_callback()

        if not video_path.exists():
            raise OSError(f"Video encoding failed: {video_path} not found.")
        return

    # ----------------------------------------
    # Normal path: h264/hevc/libsvtav1 (lossy)
    # ----------------------------------------
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported: h264, hevc, libsvtav1.")

    # Simple compatibility: some encoders dislike yuv444p by default
    if (vcodec in ("libsvtav1", "hevc")) and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    video_options: dict[str, str] = {}
    if g is not None:
        video_options["g"] = str(g)
    if crf is not None:
        video_options["crf"] = str(crf)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        for p in input_list:
            # 正常彩色可视化：转 RGB
            input_image = Image.open(p).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path} not found.")


def encode_video_frames_fast(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1", #"av1_nvenc"
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = None,  # ignored in CLI version
    overwrite: bool = False,
) -> None:
    imgs_dir = Path(imgs_dir)
    video_path = Path(video_path)
    input_pattern = str(imgs_dir / "frame_%06d.png")

    if (vcodec in ["libsvtav1", "hevc"]) and pix_fmt == "yuv444p":
        import logging
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, switching to 'yuv420p'")
        pix_fmt = "yuv420p"

    if overwrite:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        overwrite_flag = "-y"
    else:
        overwrite_flag = "-n"

    cmd = [
        "ffmpeg",
        overwrite_flag,
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
    ]

    if crf is not None:
        cmd += ["-crf", str(crf)]

    if g is not None:
        cmd += ["-g", str(g)]

    if fast_decode:
        if vcodec == "libsvtav1":
            cmd += ["-svtav1-params", f"fast-decode={fast_decode}"]
        elif vcodec == "hevc":
            cmd += ["-tune", "fastdecode"]

    cmd.append(str(video_path))
    subprocess.run(cmd, check=True)

def encode_video_frames_depth_image_v2(
    imgs_dir: Union[Path, str],
    video_path: Union[Path, str],
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
    is_depth_image: bool = False,   # 深度图3通道→MP4（无损）开关
) -> None:
    """
    读取 imgs_dir 下按 frame_000000.png 命名的 3通道图像并编码为视频。
    - is_depth_image=False：按 vcodec/pix_fmt（默认 libsvtav1 + yuv420p）写 MP4（有损/可视化）
    - is_depth_image=True ：强制 libx264rgb + rgb24 + crf=0 无损写 MP4（适合 3通道打包深度）
    """
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # 统一输出为 mp4
    if video_path.suffix.lower() != ".mp4":
        logging.warning(f"Output will be MP4; changing suffix to .mp4: {video_path}")
        video_path = video_path.with_suffix(".mp4")

    # 收集帧
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir} matching {template}.")

    # 读首帧确定尺寸
    first_img = Image.open(input_list[0]).convert("RGB")  # 只考虑3通道
    width, height = first_img.size

    # yuv420p 需要偶数分辨率（仅普通模式需要）
    def _maybe_pad_rgb(im: Image.Image) -> Image.Image:
        if pix_fmt != "yuv420p":
            return im
        w, h = im.size
        if (w % 2 == 0) and (h % 2 == 0):
            return im
        new_w = w + (w % 2)
        new_h = h + (h % 2)
        pad = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        pad.paste(im, (0, 0))
        return pad

    # 设置 PyAV 日志
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    # ---------------- 深度三通道 → MP4（无损） ----------------
    if is_depth_image:
        # 强制使用 libx264rgb + rgb24 + crf=0（无损，且不做色度下采样/颜色空间转换）
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream("libx264rgb", rate=fps, options={"crf": "0", "preset": "veryslow"})
            stream.pix_fmt = "rgb24"
            stream.width = width
            stream.height = height

            # 首帧
            arr = np.array(first_img, dtype=np.uint8)  # (H,W,3) uint8
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            pkt = stream.encode(frame)
            if pkt:
                output.mux(pkt)

            # 余下帧
            for p in input_list[1:]:
                img = Image.open(p).convert("RGB")
                arr = np.array(img, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                pkt = stream.encode(frame)
                if pkt:
                    output.mux(pkt)

            # flush
            pkt = stream.encode(None)
            if pkt:
                output.mux(pkt)

        if log_level is not None:
            av.logging.restore_default_callback()
        if not video_path.exists():
            raise OSError(f"Video encoding failed (depth, mp4): {video_path} not found.")
        return

    # ---------------- 普通三通道 → MP4（可视化） ----------------
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported: h264, hevc, libsvtav1.")

    # 兼容性提示
    if (vcodec in ("libsvtav1", "hevc")) and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # 若 yuv420p 且首帧需要补齐
    first_img = _maybe_pad_rgb(first_img)
    width, height = first_img.size

    video_options: dict[str, str] = {}
    if g is not None:
        video_options["g"] = str(g)
    if crf is not None:
        video_options["crf"] = str(crf)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(vcodec, fps, options=video_options)
        stream.pix_fmt = pix_fmt
        stream.width = width
        stream.height = height

        # 首帧
        frame = av.VideoFrame.from_image(first_img)
        pkt = stream.encode(frame)
        if pkt:
            output.mux(pkt)

        # 余下帧
        for p in input_list[1:]:
            img = Image.open(p).convert("RGB")
            img = _maybe_pad_rgb(img)
            frame = av.VideoFrame.from_image(img)
            pkt = stream.encode(frame)
            if pkt:
                output.mux(pkt)

        # flush
        pkt = stream.encode(None)
        if pkt:
            output.mux(pkt)

    if log_level is not None:
        av.logging.restore_default_callback()
    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path} not found.")

def _maybe_pad_even_rgb(im: Image.Image, need_even: bool) -> Image.Image:
    if not need_even:
        return im
    w, h = im.size
    if (w % 2 == 0) and (h % 2 == 0):
        return im
    new_w = w + (w % 2)
    new_h = h + (h % 2)
    pad = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    pad.paste(im, (0, 0))
    return pad


def _ndarray_rgb24_sane(arr: np.ndarray, target_wh_even: bool = False) -> np.ndarray:
    """确保 HxWx3 uint8、C 连续；必要时裁偶数边；并返回独享拷贝以避免 PyAV/FFmpeg 触碰共享缓冲。"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    assert arr.ndim == 3 and arr.shape[2] == 3, "frame must be (H,W,3)"
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if target_wh_even:
        H, W = arr.shape[:2]
        arr = arr[: H // 2 * 2, : W // 2 * 2, :]
    # 关键：给编码器独享缓冲，避免只读/共享视图引发段错
    return arr.copy()


def _encode_and_mux(stream: av.video.stream.VideoStream, frame: av.VideoFrame, container: av.container.OutputContainer):
    for pkt in stream.encode(frame):
        container.mux(pkt)


def _flush_and_mux(stream: av.video.stream.VideoStream, container: av.container.OutputContainer):
    for pkt in stream.encode(None):
        container.mux(pkt)


def encode_video_frames_depth_image_v3(
    imgs_dir: Union[Path, str],
    video_path: Union[Path, str],
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
    is_depth_image: bool = False,   # 深度图3通道→MP4（无损）开关
) -> None:
    """
    读取 imgs_dir 下按 frame_000000.png 命名的 3通道图像并编码为视频。
    - is_depth_image=False：按 vcodec/pix_fmt（默认 libsvtav1 + yuv420p）写 MP4（可视化）
    - is_depth_image=True ：强制 libx264rgb + rgb24 + qp=0 无损写 MP4（适合 3通道打包深度）
    """
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # 统一输出为 mp4
    if video_path.suffix.lower() != ".mp4":
        logging.warning(f"Output will be MP4; changing suffix to .mp4: {video_path}")
        video_path = video_path.with_suffix(".mp4")

    # 收集帧
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir} matching {template}.")

    # 设置 PyAV 日志
    if log_level is not None:
        av.logging.set_level(log_level)

    # 读取首帧（仅用于尺寸/像素格式探测）
    with Image.open(input_list[0]) as _im0:
        first_img = _im0.convert("RGB")
    width, height = first_img.size

    # ---------------- 深度三通道 → MP4（无损 x264rgb） ----------------
    if is_depth_image:
        # mp4 + rgb24 在多数环境可用；若个别系统仍不稳，可改成 .mkv 容器更保险
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream("libx264rgb", rate=fps)
            stream.pix_fmt = "rgb24"
            stream.width = width
            stream.height = height
            # 关键稳定项：无损 + 单线程 + 禁用切片线程
            stream.options = {
                "qp": "0",              # 真正无损
                "threads": "1",
                "sliced_threads": "0",
                # "tune": "zerolatency",  # 可选
            }

            # 编码所有帧
            # 首帧
            arr0 = _ndarray_rgb24_sane(np.array(first_img, dtype=np.uint8))
            frame0 = av.VideoFrame.from_ndarray(arr0, format="rgb24")
            _encode_and_mux(stream, frame0, output)

            # 余下帧
            for p in input_list[1:]:
                with Image.open(p) as _im:
                    img = _im.convert("RGB")
                arr = _ndarray_rgb24_sane(np.array(img, dtype=np.uint8))
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                _encode_and_mux(stream, frame, output)

            _flush_and_mux(stream, output)

        if log_level is not None:
            av.logging.set_level(av.logging.ERROR)
        if not video_path.exists():
            raise OSError(f"Video encoding failed (depth, mp4): {video_path} not found.")
        return

    # ---------------- 普通三通道 → MP4（可视化） ----------------
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported: h264, hevc, libsvtav1.")

    # 兼容性：部分编码器不接受 yuv444p
    if (vcodec in ("libsvtav1", "hevc")) and pix_fmt == "yuv444p":
        logging.warning("Incompatible pixel format 'yuv444p' for codec %s, auto-selecting 'yuv420p'", vcodec)
        pix_fmt = "yuv420p"

    need_even = (pix_fmt == "yuv420p")
    first_img = _maybe_pad_even_rgb(first_img, need_even)
    width, height = first_img.size

    video_options: dict[str, str] = {}
    if g is not None:
        video_options["g"] = str(g)
    if crf is not None:
        video_options["crf"] = str(crf)
    if fast_decode:
        if vcodec == "libsvtav1":
            video_options["svtav1-params"] = f"fast-decode={fast_decode}"
        else:
            video_options["tune"] = "fastdecode"

    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(vcodec, rate=fps, options=video_options)
        stream.pix_fmt = pix_fmt
        stream.width = width
        stream.height = height

        # 首帧
        frame = av.VideoFrame.from_image(first_img)
        _encode_and_mux(stream, frame, output)

        # 余下帧
        for p in input_list[1:]:
            with Image.open(p) as _im:
                img = _im.convert("RGB")
            img = _maybe_pad_even_rgb(img, need_even)
            frame = av.VideoFrame.from_image(img)
            _encode_and_mux(stream, frame, output)

        _flush_and_mux(stream, output)

    if log_level is not None:
        av.logging.set_level(av.logging.ERROR)
    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path} not found.")


def encode_video_frames_depth_image_v4(
        imgs_dir: Path | str,
        video_path: Path | str,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
        log_level: int | None = av.logging.ERROR,
        overwrite: bool = False,
        *,
        codec_threads: int = 1,  # 新增：控制非 SVT 编码器内部线程
        svt_lp: int = 2  # 新增：控制 SVT-AV1 的逻辑核数
) -> None:
    # ——把日志和资源的修改放在 try/finally，确保恢复——
    _set_log = False
    try:
        if vcodec not in ["h264", "hevc", "libsvtav1"]:
            raise ValueError(f"Unsupported video codec: {vcodec}. Supported: h264, hevc, libsvtav1.")

        video_path = Path(video_path)
        imgs_dir = Path(imgs_dir)
        video_path.parent.mkdir(parents=True, exist_ok=True)  # 目录存在就行

        if (vcodec in {"libsvtav1", "hevc"}) and pix_fmt == "yuv444p":
            logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-select 'yuv420p'")
            pix_fmt = "yuv420p"

        template = "frame_" + ("[0-9]" * 6) + ".png"
        input_list = sorted(
            glob.glob(str(imgs_dir / template)),
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        if not input_list:
            raise FileNotFoundError(f"No images found in {imgs_dir}.")

        # 用 with 确保句柄关闭
        with Image.open(input_list[0]) as dummy:
            width, height = dummy.size

        # 组装 codec 选项，限制内部线程
        video_options = {}
        if g is not None:
            video_options["g"] = str(g)
        if crf is not None:
            video_options["crf"] = str(crf)

        if vcodec == "libsvtav1":
            svt_params = []
            if fast_decode:
                svt_params.append(f"fast-decode={fast_decode}")
            if svt_lp and svt_lp > 0:
                svt_params.append(f"lp={svt_lp}")  # 控制 SVT 的线程数
            if svt_params:
                video_options["svtav1-params"] = ":".join(svt_params)
        else:
            # FFmpeg 通用线程参数，适用于 x264/x265 等
            video_options["threads"] = str(max(1, codec_threads))

        if log_level is not None:
            logging.getLogger("libav").setLevel(log_level)
            _set_log = True

        # 写文件（'w' 本身会覆盖）
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream(vcodec, fps, options=video_options)
            stream.pix_fmt = pix_fmt
            stream.width = width
            stream.height = height

            for p in input_list:
                # 逐个打开并立即关闭，避免堆积文件句柄
                with Image.open(p) as im:
                    im_rgb = im.convert("RGB")
                frame = av.VideoFrame.from_image(im_rgb)

                # PyAV 有时一帧可能产生 0/1/多 个 packet，用可迭代写法更稳
                for packet in stream.encode(frame):
                    output.mux(packet)

            # flush
            for packet in stream.encode():
                output.mux(packet)

        if not video_path.exists():
            raise OSError(f"Video encoding did not work. File not found: {video_path}.")

    finally:
        if _set_log:
            av.logging.restore_default_callback()

@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt or "gbrp" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format: {}".format(pix_fmt))


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format: {}".format(image.mode))
