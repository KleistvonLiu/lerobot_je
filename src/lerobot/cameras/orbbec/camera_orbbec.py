import logging
import threading
import time
from threading import Event, Lock, Thread
from typing import Any, Optional, Union

import cv2
import numpy as np
import pyorbbecsdk as OB

from .configuration_orbbec import OrbbecCameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.utils.utils import capture_timestamp_utc
from lerobot.utils.cameras.OButils import i420_to_bgr, nv12_to_bgr, nv21_to_bgr


def frame_to_bgr_image(frame: OB.VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # print(f"image format: {color_format}")
    if color_format == OB.OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OB.OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OB.OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OB.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OB.OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OB.OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        logging.info("Unsupported color format: {}".format(color_format))
        return None
    return image


def frame_to_rgb_image(frame: OB.VideoFrame) -> Optional[np.ndarray]:
    """将 OB.VideoFrame 转为 RGB ndarray，返回 None 表示不支持的格式或解码失败。"""
    if frame is None:
        return None

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    buf = frame.get_data()  # memoryview / bytes
    raw = np.frombuffer(buf, dtype=np.uint8)

    try:
        if fmt == OB.OBFormat.RGB:
            # 已经是 RGB，直接 reshape
            rgb = raw.reshape(height, width, 3)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.BGR:
            bgr = raw.reshape(height, width, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.YUYV:
            # OpenCV 期望 (H, W, 2)
            yuyv = raw.reshape(height, width, 2)
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.UYVY:
            uyvy = raw.reshape(height, width, 2)
            rgb = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.MJPG:
            # imdecode 得到 BGR，需要再转 RGB
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.I420:
            # I420 (YUV420 planar) 在 OpenCV 中 reshape 为 (H*3/2, W)
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.NV12:
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            return np.ascontiguousarray(rgb)

        elif fmt == OB.OBFormat.NV21:
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
            return np.ascontiguousarray(rgb)

        else:
            logging.info(f"Unsupported color format: {fmt}")
            return None

    except Exception as e:
        logging.warning(f"Failed to convert frame to RGB: {e}")
        return None


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


SERIAL_NUMBER_INDEX = 1

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


class OrbbecCamera:
    def __init__(
            self,
            config: OrbbecCameraConfig,
    ):
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.mock = config.mock
        self.index_or_path = config.index_or_path
        self.channels = 3
        self.Hi_resolution_mode = config.Hi_resolution_mode

        self.depth_height = None
        if self.use_depth:
            match self.width:
                case 640:
                    self.depth_height = 400
                case 1280:
                    self.depth_height = 800

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}
        self.temporal_filter = TemporalFilter(config.TemporalFilter_alpha)
        self.device = config.device_list.get_device_by_serial_number(self.index_or_path)

    def fuse_color_and_depth(self, color_image, depth_rgb_packed) -> np.ndarray:
        if color_image.shape[1] != depth_rgb_packed.shape[1]:
            raise ValueError("Width mismatch between color and depth images.")

        stacked = np.vstack((color_image, depth_rgb_packed))
        return stacked

    def load_depth_config(self):
        try:
            profile_list = self.camera.get_stream_profile_list(OB.OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            depth_profile = profile_list.get_video_stream_profile(
                self.width, self.depth_height, OB.OBFormat.Y16, self.fps
            )
            assert depth_profile is not None
            logging.info("\033[32mDEPTH Profile Loaded:\033[0m", depth_profile)
            self.OBconfig.enable_stream(depth_profile)
        except Exception as e:
            logging.info(e)
            return

    def load_color_config(self):
        try:
            profile_list = self.camera.get_stream_profile_list(OB.OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            color_profile = profile_list.get_video_stream_profile(
                self.width, self.height, OB.OBFormat.RGB, self.fps
            )
            assert color_profile is not None
            logging.info("\033[32mCOLOR Profile Loaded:\033[0m ", color_profile)
            self.OBconfig.enable_stream(color_profile)
        except Exception as e:
            logging.info(e)
            return

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("OrbbecCamera is readyConnected")
        if self.mock:
            logging.info("Waring!!MockMode is under repairing")
            return

        logging.info("\033[32mHello! Orbbec!\033[0m")

        self.OBconfig = OB.Config()
        self.camera = OB.Pipeline(self.device)

        if self.use_depth:
            self.load_depth_config()

        self.load_color_config()

        self.camera.start(self.OBconfig)

        self.is_connected = True
        logging.info("\033[32mCAMERA CONNECTED\033[0m ")
        time.sleep(5)

    def HandleDepth(self, depth_frame):
        if depth_frame is None:
            logging.info("No depth frame received")
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        # Apply temporal filter
        filtered_depth_data = self.temporal_filter.process(depth_data)
        # Convert to float32 and apply scale
        # filtered_depth_data = filtered_depth_data.astype(np.float32) * scale
        filtered_depth_data = filtered_depth_data.astype(np.uint16)

        # depth_mm = (filtered_depth_data * 1000).astype(np.uint32)
        if self.Hi_resolution_mode:
            R = ((filtered_depth_data >> 8) & 0xFF).astype(np.uint8)
            G = (filtered_depth_data & 0xFF).astype(np.uint8)
            B = np.zeros_like(R, dtype=np.uint8)

            filtered_depth_data = cv2.merge([B, G, R])

        else:
            filtered_depth_data = cv2.normalize(
                filtered_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            filtered_depth_data = cv2.applyColorMap(filtered_depth_data, cv2.COLORMAP_JET)
        return filtered_depth_data

    def read(self):
        start_time = time.perf_counter()
        frames = self.camera.wait_for_frames(100)
        if frames is None:
            logging.info("No frames received")
        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                logging.info("No depth frame received")

            self.depth_map = self.HandleDepth(depth_frame)

        color_frame = frames.get_color_frame()

        if color_frame is None:
            logging.info("No color frame received")

        self.color_image = frame_to_rgb_image(color_frame)

        if self.color_image is None:
            logging.info("failed to convert frame to image")

        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

    def _read_loop(self):
        logging.info(f"start read loop for {self.index_or_path}")
        while not self.stop_event.is_set():
            try:
                self.read()
            except Exception as e:
                logging.info(e)
                break

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        num_tries = 0
        while self.color_image is None:
            # TODO(rcadene, aliberts): intelrealsense has diverged compared to opencv over here
            num_tries += 1
            time.sleep(1 / self.fps)
            # if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
            # raise Exception(
            # logging.info(   "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called.")######可能一直报错

        if self.use_depth:
            return self.fuse_color_and_depth(self.color_image, self.depth_map)

        else:
            return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            self._stop_read_thread()

        self.camera.stop()
        self.camera = None

        self.is_connected = False
        logging.info(f"{self} disconnected.")

    def test_read(self):
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.test_loop, args=())
            self.thread.daemon = True
            self.thread.start()


if __name__ == "__main__":
    # Create a configuration for the OrbbecCamera
    config = OrbbecCameraConfig(
        fps=30,
        width=640,
        height=480,
        color_mode="bgr",
        use_depth=True,
        mock=False,
        index_or_path=0,
    )

    # Initialize the camera
    camera = OrbbecCamera(config)
    # Connect to the camera
    camera.connect()
    time.sleep(10)

    # Start asynchronous reading
    while True:
        camera.async_read()
