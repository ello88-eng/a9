from dataclasses import dataclass, field

import numpy as np


@dataclass
class Waypoint:
    waypoint_id: int
    position: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))


@dataclass
class AvsLastWaypoint:
    avs_id: int
    system_group_id: int
    mission_profile_id: int
    last_waypoint: Waypoint
