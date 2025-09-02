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

from dataclasses import dataclass
from typing import Tuple

from ..tactile_config import TactileConfig

@TactileConfig.register_subclass("serial")
@dataclass
class SerialTactileConfig(TactileConfig):
    type: str = "serial"
    baudrate: int = 9600
    timeout: float = 1
    frame_size: int = None
    header: Tuple[int, ...] = (0xFF, 0x84)
