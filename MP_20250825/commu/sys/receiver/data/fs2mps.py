from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TargetLocalInfo:
    """비주기"""

    avs_id: int = -1
    target_local_id: int = -1
    local_position: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    target_local_class: int = -1
    class_probability: float = -1.0


@dataclass
class TargetFusionResult:
    """비주기"""

    target_id: int = -1
    position: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    target_class: int = -1
    target_state: int = -1
    local_info_list: List[TargetLocalInfo] = field(default_factory=lambda: [])
