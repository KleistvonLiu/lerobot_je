import logging
import sys
import os
import time

import numpy as np

# 替换为你的项目实际路径
project_path = "/home/kleist/Documents/Code/lerobot/lerobot"
sys.path.append(project_path)

from lerobot.utils.utils import init_logging
from lerobot.robots.aloha_agilex_follower import (
    AlohaAgileXFollower,
    AlohaAgileXFollowerConfig,
)
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

DEFAULT_PNG_FILE_PATH = "/dev/video0"

if __name__ == "__main__":
    init_logging()

    config = AlohaAgileXFollowerConfig(
        port="can_right",
        id="can_right",
    )
    print("Starting Aloha Agilex Follower Test")
    logging.info(config.id)
    # 3. 创建并连接机器人
    robot = AlohaAgileXFollower(config)

    fps = 30
    step = 1/fps

    try:
        robot.connect(calibrate=True)
        robot.enable()
        while True:
            # 获取初始观测
            # obs = robot.get_status()
            action = np.array([0,0,0,0,0,0,0])
            # # sent_action = robot.send_action(action)
            sent_action = robot.send_action_np(action)
            print(sent_action)
            time.sleep(step)
    finally:
        # 确保断开连接
        robot.disconnect()
        print("robot disconnected")
