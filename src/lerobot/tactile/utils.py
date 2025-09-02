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

from pathlib import Path
from typing import TypeAlias

from .tactile_sensor import TactileSensor
from .tactile_config import TactileConfig

IndexOrPath: TypeAlias = int | Path


def make_tactiles_from_configs(tactile_config: dict[str, TactileConfig]) -> dict[str, TactileSensor]:
    tactiles = {}
    for key, cfg in tactile_config.items():
        if cfg.type == "serial":
            from .serial.serial_tactile_sensor import SerialTactileSensor
            tactiles[key] = SerialTactileSensor(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")
    return tactiles