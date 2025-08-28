#!/usr/bin/env python
import csv
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
import logging
import os
import os.path as osp
import platform
import select
import subprocess
import sys
import time
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Union, Sequence, Any, Dict

import numpy as np
import torch


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def inside_slurm():
    """Check whether the python process was launched through slurm"""
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using cuda.")
        return torch.device("mps")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    match try_device:
        case "cuda":
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        case "mps":
            assert torch.backends.mps.is_available()
            device = torch.device("mps")
        case "cpu":
            device = torch.device("cpu")
            if log:
                logging.warning("Using CPU, this will be slow.")
        case _:
            device = torch.device(try_device)
            if log:
                logging.warning(f"Using custom {try_device} device.")

    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def init_logging(log_file: Path | None = None, display_pid: bool = False):
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"

        # NOTE: Display PID is useful for multi-process logging.
        if display_pid:
            pid_str = f"[PID: {os.getpid()}]"
            message = f"{record.levelname} {pid_str} {dt} {fnameline[-15:]:>15} {record.msg}"
        else:
            message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO) #DEBUG
    logging.info("Logging initialized.")

    if log_file is not None:
        # Additionally write logs to file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def _relative_path_between(path1: Path, path2: Path) -> Path:
    """Returns path1 relative to path2."""
    path1 = path1.absolute()
    path2 = path2.absolute()
    try:
        return path1.relative_to(path2)
    except ValueError:  # most likely because path1 is not a subpath of path2
        common_parts = Path(osp.commonpath([path1, path2])).parts
        return Path(
            "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
        )


def print_cuda_memory_usage():
    """Use this function to locate and debug memory leak."""
    import gc

    gc.collect()
    # Also clear the cache if you want to fully release the memory
    torch.cuda.empty_cache()
    print("Current GPU Memory Allocated: {:.2f} MB".format(torch.cuda.memory_allocated(0) / 1024**2))
    print("Maximum GPU Memory Allocated: {:.2f} MB".format(torch.cuda.max_memory_allocated(0) / 1024**2))
    print("Current GPU Memory Reserved: {:.2f} MB".format(torch.cuda.memory_reserved(0) / 1024**2))
    print("Maximum GPU Memory Reserved: {:.2f} MB".format(torch.cuda.max_memory_reserved(0) / 1024**2))


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def say(text: str, blocking: bool = False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines):
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")


class TimerManager:
    """
    Lightweight utility to measure elapsed time.

    Examples
    --------
    ```python
    # Example 1: Using context manager
    timer = TimerManager("Policy", log=False)
    for _ in range(3):
        with timer:
            time.sleep(0.01)
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```

    ```python
    # Example 2: Using start/stop methods
    timer = TimerManager("Policy", log=False)
    timer.start()
    time.sleep(0.01)
    timer.stop()
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```
    """

    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            if self.logger is not None:
                self.logger.info(f"{self.label}: {elapsed:.6f} s")
            else:
                logging.info(f"{self.label}: {elapsed:.6f} s")
        return elapsed

    def reset(self):
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_history(self) -> list[float]:
        return [1.0 / t for t in self._history]

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:
        """
        Return the p-th percentile of recorded times.
        """
        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:
        """
        FPS corresponding to the p-th percentile time.
        """
        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val

def write_action_csv(
    action: Union[torch.Tensor, np.ndarray, Sequence],
    file_path: str | Path = "actions.csv",
) -> None:
    """
    将动作张量写入 CSV（追加模式）。

    Parameters
    ----------
    action : Tensor | ndarray | Sequence
        形状 (B, T, D) 或 (T, D) 的动作张量 / 数组 / 可迭代对象。
        - 若是 torch.Tensor，会自动 .detach().cpu().
    file_path : str | Path, default "actions.csv"
        目标 CSV 路径，文件不存在时会创建并写表头。
    """
    # -------- 1. 统一成 numpy，形状 (B, T, D) --------
    if isinstance(action, torch.Tensor):
        data = action.detach().cpu().numpy()
    else:  # ndarray 或 list
        data = np.asarray(action)

    if data.ndim == 2:         # (T, D) → (1, T, D)
        data = data[None, ...]

    assert data.ndim == 3, "action must be (B, T, D) or (T, D)"

    B, T, D = data.shape
    flat = data.reshape(B * T, D)   # (B*T, D)

    # -------- 2. 写入 CSV --------
    file_path = Path(file_path)
    write_header = not file_path.exists()

    with file_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"action_{i}" for i in range(D)]
            writer.writerow(header)

        writer.writerows(flat.tolist())

def append_depth_arrays_to_txt(
    observation: Dict[Any, Any],
    txt_path: Union[str, Path],
    target_shape=(480, 640, 1),   # H, W, C 其中 C 必须为 1
    dtype=np.uint16,
) -> int:
    """
    遍历 observation（可嵌套 dict），凡是 key 含 'depth' 且 value 是 np.ndarray，
    规范为 target_shape (默认 (480,640,1))，并以文本块形式 **追加** 写入 txt_path。

    文本格式（示例）：
      # BEGIN key=nested.left_depth timestamp=2025-08-26T08:12:34.567Z shape=480x640 dtype=uint16
      0 0 1 1 2 ...  (共 480 行，每行 640 个整数)
      ...
      # END key=nested.left_depth

    返回：本次写入的数组块数量
    """
    Ht, Wt, Ct = target_shape
    if Ct != 1:
        raise ValueError("target_shape 的最后一维必须为 1（单通道）")
    txt_path = Path(txt_path)

    def _norm_depth(arr: np.ndarray) -> np.ndarray:
        """把 (H,W)/(W,H)/(H,W,1)/(W,H,1) 统一成 (Ht,Wt,1)，并保证 C-contiguous 与 dtype。"""
        a = np.asarray(arr)
        if a.ndim == 2:
            h, w = a.shape
            if (h, w) == (Ht, Wt):
                a = a[..., None]
            elif (h, w) == (Wt, Ht):
                a = a.T[..., None]
            else:
                raise ValueError(f"意外形状: {a.shape}，期望 {(Ht,Wt)} 或 {(Wt,Ht)}")
        elif a.ndim == 3 and a.shape[2] == 1:
            h, w, _ = a.shape
            if (h, w) == (Ht, Wt):
                pass
            elif (h, w) == (Wt, Ht):
                a = np.transpose(a, (1, 0, 2))
            else:
                raise ValueError(f"意外形状: {a.shape}，期望 {(Ht,Wt,1)} 或 {(Wt,Ht,1)}")
        else:
            raise ValueError(f"不支持的深度维度: {a.shape}")
        return np.ascontiguousarray(a, dtype=dtype)

    def _walk(d: Dict[Any, Any], prefix: str = "") -> list[tuple[str, np.ndarray]]:
        out = []
        for k, v in d.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.extend(_walk(v, name))
            else:
                if "depth" in str(k).lower() and isinstance(v, np.ndarray):
                    try:
                        out.append((name, _norm_depth(v)))
                    except Exception as e:
                        # 不合规的形状/类型直接跳过（也可改成 raise 强制失败）
                        print(f"[skip] {name}: {e}")
        return out

    items = _walk(observation)
    if not items:
        return 0

    ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    count = 0
    with open(txt_path, "a", encoding="utf-8") as f:
        for key, arr in items:
            # (H,W,1) → (H,W) 逐行写入
            img2d = arr.reshape(Ht, Wt)

            f.write(f"# BEGIN key={key} timestamp={ts} shape={Ht}x{Wt} dtype={arr.dtype.name}\n")
            # 一行 640 个整数，以空格分隔；np.savetxt 会自动换行
            np.savetxt(f, img2d, fmt="%d", delimiter=" ")
            f.write(f"# END key={key}\n")
            count += 1
    return count