from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SimulationInitCommand:
    num_sims: int = -1
    base_ports: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    scenario_paths: List[str] = field(default_factory=list)
    speeds: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
