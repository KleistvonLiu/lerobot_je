import cv2
import pyorbbecsdk

from cameras.orbbec import OrbbecCamera
from cameras.orbbec import OrbbecCameraConfig

from cameras.opencv import OpenCVCamera
from cameras.opencv import OpenCVCameraConfig

from lerobot.cameras.utils import make_cameras_from_configs

#########
# ctx = pyorbbecsdk.Context()
# dev_list = ctx.query_devices()
# camera_config = OrbbecCameraConfig(index_or_path="CP02653000ZL", width=640, height=480, fps=30, device_list=dev_list)
# cameras = OrbbecCamera(camera_config)
########
camera_config = OpenCVCameraConfig(index_or_path="/dev/video26", width=640, height=480, fps=30)
cameras = OpenCVCamera(camera_config)
#############

cameras.connect()

max_timesteps = 1
i = 0
while i < max_timesteps:
    img = cameras.async_read()
    cv2.imwrite(f"image{i}.png", img)
    print(f"write image {i}: {img.shape}")
    i += 1
