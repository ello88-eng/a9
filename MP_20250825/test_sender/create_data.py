import math
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from mission_planning.configs.config import RES


def create_positions(num: int) -> np.ndarray:
    positions_2d = np.random.uniform(low=[37.954782, 126.955085], high=[38.045218, 127.044915], size=(num, 2))
    return np.array([[x, y, 500 + 900] for x, y in positions_2d])


def choice_indices(num_avs: int, num_trg: int) -> np.ndarray[float]:
    if num_trg > 0:
        return np.random.choice(num_avs, num_trg * 5, replace=False).reshape(num_trg, -1) + 1


def create_trg(num_trg: int, mission: str) -> np.ndarray:
    if mission == "surv":
        return None

    if mission == "reconn":
        return np.array([0 for _ in range(num_trg)])

    if mission == "attack":
        return np.array([1 for _ in range(num_trg)])


def is_valid(points, new_point):
    for point in points:
        distance = math.sqrt((point[0] - new_point[0]) ** 2 + (point[1] - new_point[1]) ** 2)
        if distance < 1000:
            return False
    return True


def create_loc_infos(index: np.ndarray[float]) -> np.ndarray[float]:
    loc_infos = []
    loc_info = []
    pos = np.array([0.0, 0.0, 900.0]) / np.tile([RES.lat, RES.lon, RES.alt], (1,))
    for i in index:
        loc_info.append([i, 0, *[int(p) for p in pos]])
    loc_infos.extend(loc_info)
    return np.array(loc_infos).flatten()
