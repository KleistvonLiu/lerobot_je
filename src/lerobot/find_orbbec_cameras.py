#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

import numpy as np
import cv2

# pip install pyorbbecsdk
from pyorbbecsdk import *

try:
    # 某些版本会导出 OBError
    from pyorbbecsdk import OBError
except Exception:
    class OBError(Exception):
        pass


def _safe_call(fn_or_none, default=None):
    """fn_or_none 是可调用对象或 None；调用失败时返回 default。"""
    if fn_or_none is None:
        return default
    try:
        return fn_or_none()
    except Exception:
        return default


def _frame_to_bgr(color_frame: VideoFrame) -> np.ndarray:
    """
    将 Orbbec 彩色帧转为 OpenCV 的 BGR ndarray。
    兼容 RGB/BGR/YUYV/MJPG；确保输出是 C-Contiguous。
    """
    if color_frame is None:
        raise ValueError("color_frame is None")

    # 优先直接从 frame 取格式；不行就从 profile 取
    fmt = None
    for getter in ("get_format", "format"):
        if hasattr(color_frame, getter):
            try:
                fmt = getattr(color_frame, getter)()
            except Exception:
                pass
            if fmt is not None:
                break
    if fmt is None:
        try:
            prof = color_frame.get_stream_profile()
            for getter in ("get_format", "format"):
                if hasattr(prof, getter):
                    fmt = getattr(prof, getter)()
                    if fmt is not None:
                        break
        except Exception:
            pass

    # 宽高
    W = _safe_call(getattr(color_frame, "get_width", None))
    H = _safe_call(getattr(color_frame, "get_height", None))
    if not W or not H:
        # 兜底再从 profile 取
        try:
            prof = color_frame.get_stream_profile()
            W = W or _safe_call(getattr(prof, "get_width", None))
            H = H or _safe_call(getattr(prof, "get_height", None))
        except Exception:
            pass
    if not W or not H:
        raise RuntimeError("无法获取彩色帧的宽高")

    # 原始字节
    buf = color_frame.get_data()  # memoryview / bytes
    # 对于 MJPG 可以直接解码；其他格式走 reshape
    if fmt == OBFormat.MJPG:
        # OpenCV 解码需要一份 bytes
        arr = np.frombuffer(buf, dtype=np.uint8)
        # imdecode 已经返回 BGR
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("MJPG 解码失败")
        return np.ascontiguousarray(bgr)

    # 非 MJPG：先转为 np.uint8 的“可拷贝”数组，确保 C-Contiguous
    arr = np.frombuffer(buf, dtype=np.uint8).copy()

    if fmt == OBFormat.RGB:
        # RGB -> BGR
        rgb = arr.reshape(H, W, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr)

    if fmt == OBFormat.BGR:
        bgr = arr.reshape(H, W, 3)
        return np.ascontiguousarray(bgr)

    if fmt == OBFormat.YUYV:
        # OpenCV 期望 shape=(H, W, 2)
        yuyv = arr.reshape(H, W, 2)
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        return np.ascontiguousarray(bgr)

    # 其他少见格式：尝试按 RGB 解释再转（可能失败）
    try:
        rgb = arr.reshape(H, W, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr)
    except Exception as e:
        raise RuntimeError(f"不支持的彩色帧格式：{fmt}") from e


def _pick_color_profile(dev: Device,
                        req_width: int,
                        req_height: int,
                        req_format: OBFormat,
                        req_fps: int) -> VideoStreamProfile:
    """
    在“指定设备”的彩色传感器上，选择合适的 profile。
    先试用户请求；失败则按常见格式降级；最终回落到默认 profile。
    """
    color_sensor = dev.get_sensor(OBSensorType.COLOR_SENSOR)
    plist = color_sensor.get_stream_profile_list()

    # 先试用户请求
    try:
        return plist.get_video_stream_profile(req_width, req_height, req_format, req_fps)
    except Exception:
        pass

    # 逐个降级尝试（不同相机可用性不同）
    fallbacks = [OBFormat.RGB, OBFormat.MJPG, OBFormat.YUYV, OBFormat.BGR]
    # 把 req_format 放在首位，避免重复
    tried = set([req_format])
    for fmt in fallbacks:
        if fmt in tried:
            continue
        tried.add(fmt)
        try:
            return plist.get_video_stream_profile(req_width, req_height, fmt, req_fps)
        except Exception:
            continue

    # 最终回落默认
    return plist.get_default_video_stream_profile()


def snapshot_all_orbbec_cameras(
    out_dir: str = "captures",
    req_width: int = 640,
    req_height: int = 0,                 # 某些版本允许 0 表示自适应（不保证），因此仍以“默认 profile”兜底
    req_fps: int = 30,
    req_format: OBFormat = OBFormat.RGB,
    wait_timeout_ms: int = 1500,
    warmup_frames: int = 3
) -> None:
    """
    依次绑定每一台 Orbbec 相机，抓拍彩色帧，保存到 out_dir。
    不回落到默认 start()，避免误连到第 0 台。
    """
    os.makedirs(out_dir, exist_ok=True)

    ctx = Context()
    dev_list = ctx.query_devices()

    # 取设备数
    if hasattr(dev_list, "__len__"):
        n = len(dev_list)
    elif hasattr(dev_list, "get_count"):
        n = dev_list.get_count()
    else:
        # 尝试逐个索引直到失败（极少用到）
        k = 0
        while True:
            try:
                _ = dev_list.get_device_by_index(k)
                k += 1
            except Exception:
                break
        n = k

    if n == 0:
        print("未发现任何 Orbbec 设备。")
        return

    print(f"发现 {n} 台 Orbbec 设备：")
    devices = []
    for i in range(n):
        dev = dev_list.get_device_by_index(i)
        info = dev.get_device_info()
        name = _safe_call(info.get_name, "Unknown")
        sn   = _safe_call(info.get_serial_number, None)
        fw   = _safe_call(getattr(info, "get_firmware_version", None), None)
        vid  = _safe_call(getattr(info, "get_vid", None), None)
        pid  = _safe_call(getattr(info, "get_pid", None), None)
        print(f"  [{i}] name={name}  serial={sn}  vid={vid} pid={pid} fw={fw}")
        devices.append((i, dev, name, sn, vid, pid, fw))

    # 逐台相机抓拍
    for (i, dev, name, sn, vid, pid, fw) in devices:
        tag = f"idx{i}_{sn or 'nosn'}_{int(time.time())}"
        out_path = os.path.join(out_dir, f"orbbec_{tag}.png")

        pipeline = Pipeline(dev)
        config = Config()

        try:
            # 1) 在“该设备”上选取合适的彩色 profile
            vprof = _pick_color_profile(dev, req_width, req_height, req_format, req_fps)

            # 2) 启用彩色流
            config.enable_stream(vprof)

            # 3) 启动：强制用设备对象绑定。只有没有这个 API 时，才用 SN/UID 方式
            started = False
            if hasattr(pipeline, "start_with_device"):
                pipeline.start_with_device(dev, config)  # 强绑定到这台 dev
                started = True
            else:
                # 某些老版本没有 start_with_device，则尽量用 SN 或 UID 绑定
                bound = False
                if sn and hasattr(config, "enable_device_by_sn"):
                    try:
                        config.enable_device_by_sn(sn)
                        bound = True
                    except Exception:
                        bound = False

                # 若 SN 不可用/无效，尝试 UID（很多机型都有 UID）
                uid = _safe_call(getattr(info, "get_uid", None), None)
                if (not bound) and uid and hasattr(config, "enable_device_by_uid"):
                    try:
                        config.enable_device_by_uid(uid)
                        bound = True
                    except Exception:
                        bound = False

                if not bound:
                    raise RuntimeError(
                        f"SDK 无法通过 start_with_device/SN/UID 绑定到设备[{i}] ({name}/sn={sn})"
                    )

                pipeline.start(config)
                started = True

            # 4) 启动后做一次“实际设备校验”（防止仍然连到了别的相机）
            try:
                if hasattr(pipeline, "get_device"):
                    cur = pipeline.get_device()
                    cur_info = cur.get_device_info()
                    cur_sn = _safe_call(cur_info.get_serial_number, None)
                    # 如果拿得到 SN 且和期望不符，则报错
                    if sn and cur_sn and cur_sn != sn:
                        raise RuntimeError(f"绑定错设备：期望 SN={sn}，实际 SN={cur_sn}")
            except Exception as chk_e:
                # 仅打印告警，不中断（某些版本可能没有 get_device）
                print(f"[warn] 无法校验当前 pipeline 设备或 SN，原因：{chk_e}")

            # # 5) 若 4 失败（比如 enable_device_by_sn 不支持/无效），再用设备对象强绑定
            # if not started and hasattr(pipeline, "start_with_device"):
            #     pipeline.start_with_device(dev, config)
            #     started = True
            #
            # # 6) 明确绑定仍失败：直接报错（不要再默认 start() 以免误连其它设备）
            # if not started:
            #     raise RuntimeError(f"无法绑定并启动设备[{i}] ({name}/sn={sn})")

            # 7) 丢弃热身帧
            for _ in range(max(0, warmup_frames)):
                _ = pipeline.wait_for_frames(wait_timeout_ms)

            # 8) 取一帧并保存
            frames = pipeline.wait_for_frames(wait_timeout_ms)
            if frames is None:
                raise RuntimeError("wait_for_frames 返回 None")
            color_frame = frames.get_color_frame()
            if color_frame is None:
                raise RuntimeError("未获得彩色帧（color_frame is None）")

            bgr = _frame_to_bgr(color_frame)
            ok = cv2.imwrite(out_path, bgr)
            if not ok:
                raise RuntimeError(f"cv2.imwrite 失败：{out_path}")

            # 打印实际使用到的 profile 信息
            try:
                prof = color_frame.get_stream_profile()
                W   = _safe_call(getattr(prof, "get_width", None), None)  or _safe_call(getattr(prof, "width", None), None)
                H   = _safe_call(getattr(prof, "get_height", None), None) or _safe_call(getattr(prof, "height", None), None)
                FPS = _safe_call(getattr(prof, "get_fps", None), None)    or _safe_call(getattr(prof, "fps", None), None)
                FMT = _safe_call(getattr(prof, "get_format", None), None) or _safe_call(getattr(prof, "format", None), None)
            except Exception:
                W = H = FPS = FMT = None

            print(
                f"[OK] 保存 {out_path}\n"
                f"     设备索引={i}, 名称={name}, 序列号={sn}, VID={vid}, PID={pid}, FW={fw}\n"
                f"     实际分辨率={W}x{H}, FPS={FPS}, FORMAT={FMT}"
            )

        except Exception as e:
            import traceback
            print(f"[FAIL] 设备索引={i}（{name} / sn={sn}）：{e}")
            traceback.print_exc()

        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            # 给 UVC/USB 一点间隔，避免频繁启停导致占用
            time.sleep(0.5)


if __name__ == "__main__":
    snapshot_all_orbbec_cameras(
        out_dir="captures",
        req_width=640,
        req_height=0,           # 若该相机不接受 0，高度会回落到默认 profile
        req_fps=30,
        req_format=OBFormat.RGB,
        wait_timeout_ms=1500,
        warmup_frames=3
    )
