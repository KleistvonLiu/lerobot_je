import logging
import struct
import time
from typing import Optional, Tuple
from threading import Thread, Event, Lock

import numpy as np
import serial

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..tactile_sensor import TactileSensor
from .serial_tactile_config import SerialTactileConfig


class SerialTactileSensor(TactileSensor):
    """
    串口读取 32 通道触觉帧，后台线程持续抓取，read() 返回最新缓存。
    帧格式（70B）= [0xFF,0x84][cnt_hi,cnt_lo][ADC0..ADC31 (BE uint16)] [CRC_hi,CRC_lo]
    """

    def __init__(self, config: SerialTactileConfig):
        super().__init__(config)
        self.ser: Optional[serial.Serial] = None
        self.port = config.port
        self.baudrate = config.baudrate
        self.timeout = config.timeout
        self.max_wait_s = 2.0  # 等待首帧/搜帧的最大时长
        self.width = config.width
        self.height = config.height
        self.frame_size = config.frame_size
        self.header: bytes = bytes(config.header)
        self._h0: bytes = self.header[:1]  # e.g. b'\xFF'
        self._h1: Optional[bytes] = self.header[1:2] or None  # e.g. b'\x84' 或 None
        # 后台线程相关
        self._reader_thr: Optional[Thread] = None
        self._stop_evt: Optional[Event] = None
        self._first_frame_evt: Optional[Event] = None
        self._lock = Lock()
        # 最新缓存
        self._latest_adc: Optional[np.ndarray] = None  # shape=(self.height, self.width), float32
        self._latest_cnt: Optional[int] = None
        self._latest_ts_ns: Optional[int] = None
        # print(self.width, self.height, self.frame_size, self.header)
        # print(self._h0, self._h1)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.port})"

    @property
    def is_connected(self) -> bool:
        return (self.ser is not None) and self.ser.is_open

    # -------- 低层：读取与解析 --------

    def _read_exact_until(self, n: int, deadline: float) -> bytes:
        """阻塞读取 n 字节或到 deadline 超时。"""
        buf = bytearray()
        while len(buf) < n and time.monotonic() < deadline:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < n:
            raise TimeoutError("read timeout while filling buffer")
        return bytes(buf)

    @staticmethod
    def _verify_checksum(frame: bytes) -> bool:
        # 文档：对 2..67 求和，低16位；68..69 为校验（高字节在前）
        calc = sum(frame[2:68]) & 0xFFFF
        recv = (frame[68] << 8) | frame[69]
        return calc == recv

    @staticmethod
    def _parse_adc(frame: bytes) -> Tuple[int, np.ndarray]:
        """返回 (计数器, 32路 ADC 数组 uint16)。"""
        cnt = (frame[2] << 8) | frame[3]  # 大端
        adc_u16 = np.frombuffer(frame, dtype='>u2', count=32, offset=4).astype(np.uint16, copy=True)
        return cnt, adc_u16

    def _read_one_frame(self, deadline: float) -> bytes:
        """从流中同步到帧头并读取一整帧；校验失败会继续尝试，直到超时。"""
        saw_ff = False
        while time.monotonic() < deadline and not self._stop_evt.is_set():
            b = self.ser.read(1)
            if not b:
                continue
            if not saw_ff:
                saw_ff = (b == self._h0)
                continue
            # 已看到 0xFF
            if self._h1 is None or b == self._h1:
                # 读取剩余 68 字节
                rest = self._read_exact_until(self.frame_size - 2, deadline)
                frame = self.header + rest
                if self._verify_checksum(frame):
                    return frame
                # 校验失败，重新找 0xFF
                saw_ff = False
            else:
                # 连续 0xFF 允许作为新起点
                saw_ff = (b == self._h0)
        raise TimeoutError("timeout waiting for a valid frame")

    # -------- 后台读线程 --------

    def _reader_loop(self):
        """持续读取有效帧并更新缓存。"""
        # 尝试持续读取；若偶发超时，继续；严重串口错误则退出线程。
        while not self._stop_evt.is_set():
            try:
                frame = self._read_one_frame(deadline=time.monotonic() + 1.0)
            except TimeoutError:
                # 允许空转，继续尝试直到 stop
                continue
            except (serial.SerialException, OSError) as e:
                logging.error(f"{self}: serial error in reader thread: {e}")
                break

            cnt, adc_u16 = self._parse_adc(frame)
            adc = adc_u16.astype(np.float32).reshape((self.height, self.width))
            ts_ns = time.monotonic_ns()

            with self._lock:
                # 分配一次后复用内存，减少分配抖动（可选）
                if self._latest_adc is None or self._latest_adc.shape != (self.height, self.width):
                    self._latest_adc = adc.copy()
                else:
                    np.copyto(self._latest_adc, adc)
                self._latest_cnt = cnt
                self._latest_ts_ns = ts_ns

            # 标记首帧已到
            if self._first_frame_evt and not self._first_frame_evt.is_set():
                self._first_frame_evt.set()

    # -------- 公共接口 --------

    def connect(self, warmup: bool = True):
        """打开串口并启动后台读线程。"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        self.ser = serial.Serial(
            port=self.port, baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout, write_timeout=0.2,
            rtscts=False, dsrdtr=False, xonxoff=False,
        )
        try:
            # 注意：某些板卡 DTR 变化会复位；如不希望复位，可改为 False
            self.ser.dtr = True
            self.ser.rts = False
        except Exception:
            pass

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.05)

        # 清空缓存
        with self._lock:
            self._latest_adc = None
            self._latest_cnt = None
            self._latest_ts_ns = None

        # 读线程控制
        self._stop_evt = Event()
        self._first_frame_evt = Event()

        self._reader_thr = Thread(target=self._reader_loop, name=f"{self}-reader", daemon=True)
        self._reader_thr.start()

        # 可选：预热，尝试等首帧，避免上层第一次 read() 立即拿不到
        if warmup:
            self._first_frame_evt.wait(timeout=self.max_wait_s)

        logging.info(f"Tactile sensor {self.port} connected and reader started.")

    def read(self) -> np.ndarray:
        """
        返回最新缓存的触觉数据（shape=(self.height,self.width), float32）。
        - 若首帧尚未到达，则最多等待 max_wait_s；仍无则抛 TimeoutError。
        - 之后每次调用都为即时返回（可能多次返回同一帧，这是预期行为）。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._latest_adc is None:
            # 首帧等待：避免上层拿到 None
            if not (self._first_frame_evt and self._first_frame_evt.wait(timeout=self.max_wait_s)):
                raise TimeoutError("no frame available yet")

        with self._lock:
            if self._latest_adc is None:
                raise TimeoutError("no frame available yet")
            # 返回副本，避免外部修改内部缓存
            return self._latest_adc.copy()

    def disconnect(self):
        """优雅停止后台线程并关闭串口。"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        # 通知线程停止
        if self._stop_evt:
            self._stop_evt.set()
        # 尝试唤醒阻塞读
        try:
            if self.ser:
                self.ser.cancel_read()
        except Exception:
            pass

        # 等线程退出
        if self._reader_thr and self._reader_thr.is_alive():
            self._reader_thr.join(timeout=1.0)

        # 释放控制线并关闭串口
        try:
            if hasattr(self.ser, "dtr"):
                self.ser.dtr = False
            if hasattr(self.ser, "rts"):
                self.ser.rts = False
        except Exception:
            pass

        self.ser.close()
        logging.info(f"{self} disconnected.")
