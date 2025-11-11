
from pathlib import Path
import logging
import numpy as np
# import pytest
import cv2
import os

from lerobot.cameras.configs import ColorMode

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.cameras.utils import make_cameras_from_configs

# 配置日志级别，确保INFO级别的日志能显示
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

cameras = make_cameras_from_configs(
    {
        "orbbec": OrbbecCameraConfig(
            index_or_path="CP02653000YJ",
            width=640,
            height=480,
            fps=30,
            use_depth=True,
            warmup_s=2,
        )
    }
)

def save_color_frame(color_image:np.ndarray, index = 0):
    save_image_dir = os.path.join(os.getcwd(), "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    filename = save_image_dir + "/color_{}.png".format(index)
    print(f"Save image to: {filename}")
    image = color_image
    if image is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, image)

# def save_depth_frame(depth_map:np.ndarray, index):
    
#     save_image_dir = os.path.join(os.getcwd(), "depth_images")
#     if not os.path.exists(save_image_dir):
#         os.mkdir(save_image_dir)
#     raw_filename = save_image_dir + "/depth_{}x{}_{}_{}.raw".format(width, height, index, timestamp)
#     data.tofile(raw_filename)

cameras["orbbec"].connect()

(color_image, depth_map) = cameras["orbbec"].read()
print("color_image shape:", color_image.shape)
# save_color_frame(color_image)
print("depth_map shape:", depth_map.shape)

cameras["orbbec"].disconnect()

