from typing import List, Tuple

import numpy as np

from manager.manager import Manager


def get_corner_coordinate_array(rx: List[float], ry: List[float], alt: float):
    corner_waypoint_array = [(rx[0], ry[0], alt)]
    for i in range(1, len(rx) - 1):
        dx1, dy1 = rx[i] - rx[i - 1], ry[i] - ry[i - 1]
        ang_1 = np.degrees(np.arctan2(dy1, dx1))
        dx2, dy2 = rx[i + 1] - rx[i], ry[i + 1] - ry[i]
        ang_2 = np.degrees(np.arctan2(dy2, dx2))
        if np.abs(ang_2 - ang_1) > 20:
            corner_waypoint_array.append((rx[i], ry[i], alt))
    corner_waypoint_array.append((rx[-1], ry[-1], alt))

    return np.array(corner_waypoint_array)


def get_coordinate_array(rx: List[float], ry: List[float], alt: float):
    corner_waypoint_array = [(rx[0], ry[0], alt)]
    for i in range(1, len(rx) - 1):
        corner_waypoint_array.append((rx[i], ry[i], alt))

    return np.array(corner_waypoint_array)


def create_mesh(
    manager: Manager,
    top_left: np.ndarray[float],
    top_right: np.ndarray[float],
    bottom_right: np.ndarray[float],
    bottom_left: np.ndarray[float],
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    xs = np.linspace(top_left[1], top_right[1], manager.num_columns)
    ys = np.linspace(bottom_right[0], top_right[0], manager.num_rows)
    mesh_xs, mesh_ys = np.meshgrid(xs, ys)

    return mesh_xs, mesh_ys
