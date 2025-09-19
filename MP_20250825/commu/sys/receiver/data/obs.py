from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AvsObservation:
    id: int = -1
    is_auto: int = -1
    mode: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    wp_num: int = 4


@dataclass
class ObjectObservation:
    id: int = -1
    trg_cls: int = -1
    state: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    is_target: int = -1


@dataclass
class FusionObservaion:
    timestamp: int = 0
    world_name: str = ""  # deprecated
    mission_type: str = ""
    num_avs: int = 0
    num_obj: int = 0
    map_origin: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    roi_area: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    friendly_area: np.ndarray = field(default_factory=lambda: np.zeros((1,)))  # deprecated
    avs_obs: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    obj_global_obs: np.ndarray = field(default_factory=lambda: np.zeros((1,)))  # deprecated
    obj_fusion_obs: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    footprints: np.ndarray = field(default_factory=lambda: np.zeros((1,)))  # deprecated
    grid_data: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    arrival_per_group: Dict[int, bool] = field(default_factory=lambda: {})  # deprecated
    group_to_ids: Dict[int, List[int]] = field(default_factory=lambda: {})
