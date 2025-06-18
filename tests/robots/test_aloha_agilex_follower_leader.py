import sys
import os
import numpy as np

# 替换为你的项目实际路径
project_path = "/home/kleist/Documents/Code/lerobot/lerobot"
sys.path.append(project_path)

from lerobot.common.robots.aloha_agilex_follower import (
    AlohaAgileXFollower,
    AlohaAgileXFollowerConfig,
)
from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
DEFAULT_PNG_FILE_PATH = "/dev/video0"

if __name__ == "__main__":
    # 1. 创建相机配置
    print("here1")
    camera1_config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, width=640, height=480, fps=30)
    print("here2")
    cameras = {"camera1":camera1_config}
    # camera = OpenCVCamera(config)
    # camera.connect(warmup=False)

    # 2. 创建机器人配置
    config = AlohaAgileXFollowerConfig(
        port="can_left",
        cameras=cameras,
    )

    # 3. 创建并连接机器人
    robot = AlohaAgileXFollower(config)

    try:
        robot.connect(calibrate=True)
        robot.enable()

        # 获取初始观测
        obs = robot.get_observation()
        print("follower arm joints (0.001 deg):")
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):  # 如果值不是图像数据
                print(f"  {key}: {value}")

        leader_arm_action = robot.get_leader_action()
        print("leader arm joints(0.001 deg):")
        for key, value in leader_arm_action.items():
            if not isinstance(value, np.ndarray):  # 如果值不是图像数据
                print(f"  {key}: {value}")

        # # 发送动作
        action = {
            "joint0.pos": obs["joint0.pos"],
            "joint1.pos": obs["joint1.pos"],
            "joint2.pos": obs["joint2.pos"],
            "joint3.pos": obs["joint3.pos"],
            "joint4.pos": obs["joint4.pos"],
            "joint5.pos": obs["joint5.pos"],
            "joint6.pos": obs["joint6.pos"] # 夹爪位置
        }
        sent_action = robot.send_action(action)
        print(f"已发送动作: {sent_action}")

    finally:
        # 确保断开连接
        robot.disconnect()
        print("robot disconnected")