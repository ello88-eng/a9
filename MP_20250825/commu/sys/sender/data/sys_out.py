from abc import ABC
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MissionPlanResultInfo:
    avs_list: List[int] = field(default_factory=lambda: [])
    smp_group_id: int = -1
    mission_info: int = -1
    avs_count: int = -1


@dataclass
class BaseSelectionResult(ABC):
    avs_id: int = -1
    system_group_id: int = -1
    smp_mode: int = -1


@dataclass
class SurveillanceSelectionResult(BaseSelectionResult):
    speed: int = -1
    waypoint_count: int = -1
    waypoints: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))

    def __post_init__(self):
        # 외부에서 값을 주더라도 0으로 고정
        object.__setattr__(self, "smp_group_id", 0)


@dataclass
class ReconnaissanceSelectionResult(BaseSelectionResult):
    smp_group_id: int = -1
    coordinate: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    turning_radius: int = -1
    turning_direction: int = -1
    target_position: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    target_id: int = -1


@dataclass
class TrackSelectionResult:
    target_id: int = -1
    total_avs_list: List[int] = field(default_factory=lambda: [])
    track_avs_list: List[int] = field(default_factory=lambda: [])
    priority_list: List[int] = field(default_factory=lambda: [])


@dataclass
class AttackSelectionResult(ReconnaissanceSelectionResult):
    attack_priority: int = -1
    bd_assignment_flag: int = -1


@dataclass
class SmpStatus:
    c_bit: int = -1


@dataclass
class AvsInfoForRl:
    pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
            ]
        )
    )
    attitude: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
            ]
        )
    )
    duration: float = field(
        default_factory=lambda: np.random.normal(loc=-1, scale=0.0),
    )


@dataclass
class TargetInfoForRl:
    pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
                np.random.normal(loc=-1, scale=0.0),
            ]
        )
    )
    value: float = field(
        default_factory=lambda: np.random.normal(loc=-1, scale=0.0),
    )
