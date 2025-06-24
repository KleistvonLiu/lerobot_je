import logging
import sys
import os
import numpy as np

# 替换为你的项目实际路径
project_path = "/home/kleist/Documents/Code/lerobot/lerobot"
sys.path.append(project_path)

from lerobot.common.utils.utils import init_logging
from lerobot.common.robots.aloha_agilex_follower import (
    AlohaAgileXFollower,
    AlohaAgileXFollowerConfig,
)
from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

DEFAULT_PNG_FILE_PATH = "/dev/video0"

if __name__ == "__main__":
    init_logging()
    # 1. 创建相机配置
    camera1_config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, width=640, height=480, fps=30)
    cameras = {"camera1": camera1_config}
    # camera = OpenCVCamera(config)
    # camera.connect(warmup=False)

    # 2. 创建机器人配置
    config = AlohaAgileXFollowerConfig(
        port="can_left",
        cameras=cameras,
        id="can_left",
    )
    print("Starting Aloha Agilex Follower Test")
    logging.info(config.id)
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
        print(obs.keys())
        # # 发送动作
        # action = {
        #     robot.id + ".joint0.pos": obs[robot.id + ".joint0.pos"],
        #     robot.id + ".joint1.pos": obs[robot.id + ".joint1.pos"],
        #     robot.id + ".joint2.pos": obs[robot.id + ".joint2.pos"],
        #     robot.id + ".joint3.pos": obs[robot.id + ".joint3.pos"],
        #     robot.id + ".joint4.pos": obs[robot.id + ".joint4.pos"],
        #     robot.id + ".joint5.pos": obs[robot.id + ".joint5.pos"],
        #     robot.id + ".joint6.pos": obs[robot.id + ".joint6.pos"]  # 夹爪位置
        # }
        action = np.array([obs[robot.id + ".joint0.pos"], obs[robot.id + ".joint1.pos"], obs[robot.id + ".joint2.pos"],
                           obs[robot.id + ".joint3.pos"], obs[robot.id + ".joint4.pos"], obs[robot.id + ".joint5.pos"],
                           obs[robot.id + ".joint6.pos"]])
        action = np.array([0,0,0,0,0,0,0])
        # sent_action = robot.send_action(action)
        sent_action = robot.send_action_np(action)
        print(f"已发送动作: {sent_action}")

    finally:
        # 确保断开连接
        robot.disconnect()
        print("robot disconnected")
