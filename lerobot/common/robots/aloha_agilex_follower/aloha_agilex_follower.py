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

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_aloha_agilex_follower import AlohaAgileXFollowerConfig

from piper_sdk import *
from piper_sdk import C_PiperInterface

logger = logging.getLogger(__name__)


class AlohaAgileXFollower(Robot):
    config_class = AlohaAgileXFollowerConfig
    name = "aloha_agilex_follower"

    def __init__(self, config: AlohaAgileXFollowerConfig):
        super().__init__(config)
        self.config = config
        self.piper = C_PiperInterface(can_name=self.config.port)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.is_enabled_ = False
        self.is_robot_connected_ = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {self.id + f".joint{i}.pos": float for i in range(7)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

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
        self.piper.ConnectPort()
        self.is_robot_connected_ = True
        for cam in self.cameras.values():
            cam.connect()

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

        # Read arm position
        start = time.perf_counter()
        obs_dict = {
            self.id + f".joint{i}.pos":
                (getattr(self.piper.GetArmJointMsgs().joint_state, f"joint_{i + 1}"))
            for i in range(6)  # 从 0 到 5
        }
        obs_dict[self.id + ".joint6.pos"] = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

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

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

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

    @property
    def is_enabled(self) -> bool:
        return self.is_enabled_

    def enable(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.is_enabled_ = True
        return

    def disconnect(self):
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
