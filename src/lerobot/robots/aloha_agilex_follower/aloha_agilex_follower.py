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

import logging
import math
import time
from functools import cached_property
from gc import enable
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.tactile.utils import make_tactiles_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_aloha_agilex_follower import AlohaAgileXFollowerConfig

from piper_sdk import C_PiperInterface

logger = logging.getLogger(__name__)


class AlohaAgileXFollower(Robot):
    config_class = AlohaAgileXFollowerConfig
    name = "aloha_agilex_follower"

    def __init__(self, config: AlohaAgileXFollowerConfig):
        super().__init__(config)
        self.config = config
        self.piper = C_PiperInterface(can_name=self.config.port)
        # self.piper.ConnectPort()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.tactile_sensors = make_tactiles_from_configs(config.tactiles)
        self.is_enabled_ = False
        self.is_robot_connected_ = False
        self.is_piper_port_connected_ = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        # expose position, velocity and effort keys for each joint so recorder can pick them up
        out: dict[str, type] = {}
        for i in range(7):
            out[self.id + f".joint{i}.pos"] = float
            out[self.id + f".joint{i}.vel"] = float
            out[self.id + f".joint{i}.effort"] = float
        return out

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        base = {
            cam: (self.config.cameras[cam].height,
                  self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        depth = {
                f"{cam}_depth": (self.config.cameras[cam].height,
                                 self.config.cameras[cam].width, 3)
                for cam in self.cameras if self.cameras[cam].use_depth
            }
        return {**base, **depth}

    @property
    def _tactile_ft(self) -> dict[str, tuple[int, int]]:
        tactile_sensors = {
            self.id+f"." +tactile: (self.config.tactiles[tactile].height, self.config.tactiles[tactile].width)
            for tactile in self.tactile_sensors
        }
        return tactile_sensors

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._tactile_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        ## TODO: add piper is connected
        return self.is_robot_connected_ and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        # self.piper.ConnectPort()
        self.is_robot_connected_ = True
        for cam in self.cameras.values():
            cam.connect()

        for tactile in self.tactile_sensors.values():
            tactile.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        ## TODO: add calibration process
        return True

    def calibrate(self) -> None:
        logger.info(f"\nNo calibration {self} can be run")

    def configure(self) -> None:
        ## TODO: we need to configure the torque
        logger.info(f"\nNo configuration {self} can be run")
        return

    def setup_motors(self) -> None:
        logger.info(f"\nWe don't have to setup motor for {self}")
        return

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.is_piper_port_connected_:
            self.piper.ConnectPort()
            self.is_piper_port_connected_ = True

        # Read arm joint states (position, velocity, effort when available)
        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        joint_state = self.piper.GetArmJointMsgs().joint_state
        highspd = self.piper.GetArmHighSpdInfoMsgs()
        gripper = self.piper.GetArmGripperMsgs().gripper_state

        # joints 0..5
        for i in range(6):
            # position from joint_state
            pos_val = None
            try:
                jattr = getattr(joint_state, f"joint_{i + 1}")
            except Exception:
                jattr = None
            try:
                if jattr is None:
                    pos_val = None
                elif isinstance(jattr, (int, float)):
                    pos_val = float(jattr)
                else:
                    # try common attribute names
                    if hasattr(jattr, "position"):
                        pos_val = float(jattr.position)
                    elif hasattr(jattr, "angle"):
                        pos_val = float(jmsg.angle) if (jmsg := getattr(jattr, 'angle', None)) is not None else None
                    elif hasattr(jattr, "pos"):
                        pos_val = float(jattr.pos)
            except Exception:
                pos_val = None

            # velocity from highspd.motor_X.motor_speed if available
            vel_val = None
            try:
                motor = getattr(highspd, f"motor_{i + 1}", None)
                if motor is not None:
                    vel_val = float(getattr(motor, "motor_speed", motor))
            except Exception:
                vel_val = None

            # per-joint effort not provided by this interface; default to None
            eff_val = None

            obs_dict[self.id + f".joint{i}.pos"] = pos_val
            obs_dict[self.id + f".joint{i}.vel"] = vel_val
            obs_dict[self.id + f".joint{i}.effort"] = eff_val

        # gripper as joint6
        try:
            obs_dict[self.id + ".joint6.pos"] = float(gripper.grippers_angle)
        except Exception:
            obs_dict[self.id + ".joint6.pos"] = None
        try:
            obs_dict[self.id + ".joint6.effort"] = float(gripper.grippers_effort)
        except Exception:
            obs_dict[self.id + ".joint6.effort"] = None
        # velocity for gripper not available; set to 0
        obs_dict[self.id + ".joint6.vel"] = 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.3f}ms,{list(obs_dict.values())}")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            # start = time.perf_counter()
            camera_frame = cam.async_read()
            if isinstance(camera_frame, tuple):
                color_image, depth_map = camera_frame
                obs_dict[cam_key] = color_image
                obs_dict[cam_key + "_depth"] = depth_map
            else:
                obs_dict[cam_key] = camera_frame
            # dt_ms = (time.perf_counter() - start) * 1e3
            # logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Capture data from tactile sensors
        for tactile_key, tactile in self.tactile_sensors.items():
            tactile_frame = tactile.read()
            obs_dict[tactile_key] = tactile_frame

        return obs_dict

    def get_status(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.is_piper_port_connected_:
            self.piper.ConnectPort()
            self.is_piper_port_connected_ = True


        # Read arm position
        start = time.perf_counter()
        obs_dict = {
            self.id + f".joint{i}.pos":
                (getattr(self.piper.GetArmJointMsgs().joint_state, f"joint_{i + 1}"))
            for i in range(6)  # 从 0 到 5
        }
        obs_dict[self.id + ".joint6.pos"] = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        obs_dict['endpose_x'] = self.piper.GetArmEndPoseMsgs().end_pose.X_axis
        obs_dict['endpose_y'] = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis
        obs_dict['endpose_z'] = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis
        obs_dict['endpose_rx'] = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis
        obs_dict['endpose_ry'] = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis
        obs_dict['endpose_rz'] = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis
        return obs_dict

    def get_leader_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = {
            self.id + f".joint{i}.pos":
                (getattr(self.piper.GetArmJointCtrl().joint_ctrl, f"joint_{i + 1}"))
            for i in range(6)  # 从 0 到 5
        }
        obs_dict[self.id + ".joint6.pos"] = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_enabled:
            self.enable()

        goal_pos = {key.removesuffix(".pos").removeprefix(f"{self.id}."): val for key, val in action.items() if
                    (key.endswith(".pos") and key.startswith(self.id))}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        # if self.config.max_relative_target is not None:
        #     present_pos = {
        #         f"joint{i}":
        #             (getattr(self.piper.GetArmJointMsgs().joint_state, f"joint_{i + 1}") / 1000) * math.pi / 180
        #         for i in range(6)  # 从 0 到 5
        #     }
        #     present_pos["joint6"] = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        #
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
        #     goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        factor = 1000 * 180 / math.pi
        joint_0 = int(goal_pos["joint0"].item())
        joint_1 = int(goal_pos["joint1"].item())
        joint_2 = int(goal_pos["joint2"].item())
        joint_3 = int(goal_pos["joint3"].item())
        joint_4 = int(goal_pos["joint4"].item())
        joint_5 = int(goal_pos["joint5"].item())
        joint_6 = int(goal_pos["joint6"].item())
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(joint_0, joint_1, joint_2,
                             joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        # self.piper.MotionCtrl_2(0x01, 0x01, 100)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def send_action_np(self, action: np.ndarray):
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        # start_episode_t = time.perf_counter()
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_enabled:
            self.enable()
        # time_point1 = time.perf_counter()
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        # time_point2 = time.perf_counter()
        self.piper.JointCtrl(int(action[0]), int(action[1]), int(action[2]),
                             int(action[3]), int(action[4]), int(action[5]))
        # time_point3 = time.perf_counter()
        self.piper.GripperCtrl(abs(int(action[6])), 1000, 0x01, 0)
        print("here")
        # time_point4 = time.perf_counter()
        # logging.info(
        #     f"time cost {1e3 * (time_point1 - start_episode_t):.3f}/{1e3 * (time_point2 - start_episode_t):.3f}/"
        #     f"{1e3 * (time_point3 - start_episode_t):.3f}/{1e3 * (time_point4 - start_episode_t):.3f}")
        return

    def send_endpose(self, action: np.ndarray):
        """Command arm to move to a target joint configuration.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_enabled:
            self.enable()
        # time_point1 = time.perf_counter()
        self.piper.MotionCtrl_2(0x01, 0x00, 100)
        # time_point2 = time.perf_counter()
        self.piper.EndPoseCtrl(int(action[0]*1000), int(action[1]*1000), int(action[2]*1000),
                             int(action[3]*1000), int(action[4]*1000), int(action[5]*1000))
        return

    @property
    def is_enabled(self) -> bool:
        return self.is_enabled_

    def enable(self) -> None:
        logger.info(f"try to enable {self.id}")
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.is_enabled_ = True
        return

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.is_piper_port_connected_:
            self.piper.DisconnectPort()
            self.is_piper_port_connected_ = False

        # self.piper.DisconnectPort()
        for cam in self.cameras.values():
            cam.disconnect()

        for tactile in self.tactile_sensors.values():
            tactile.disconnect()

        logger.info(f"{self} disconnected.")

    def disconnect_port(self):
        if self.is_piper_port_connected_:
            self.piper.DisconnectPort()
            self.is_piper_port_connected_ = False

    def just_for_test(self):
        print("here just for test")
        self.piper.ConnectPort()
        max_steps = 10000000000
        for idx in range(max_steps):
            start_time = time.perf_counter()
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
            self.piper.GripperCtrl(abs(0), 1000, 0x01, 0)
            end_time = time.perf_counter()
            if idx % 1 == 0:
                print(f"time cost:{(end_time - start_time) * 1e3}")
            # piper.MotionCtrl_2(0x01, 0x01, 100)
            time.sleep(0.01)
