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

import abc

import numpy as np

from .tactile_config import TactileConfig


class TactileSensor(abc.ABC):
    """
    Base class for tactile implementations.
    """

    def __init__(self, config: TactileConfig):
        """Initialize the camera with the given configuration.

        Args:
            config: Camera configuration containing FPS and resolution.
        """
        self.port = config.port

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.

        Returns:
            bool: True if the camera is connected and ready to capture frames,
                  False otherwise.
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the camera.

        Args:
            warmup: If True (default), captures a warmup frame before returning. Useful
                   for cameras that require time to adjust capture settings.
                   If False, skips the warmup frame.
        """
        pass

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """Capture and return a single frame from the camera.

        Returns:
            np.ndarray: Captured frame as a numpy array.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        pass
