import argparse
import random
import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage.transform import resize

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from policies.surv.algorithms.ADD2.darp import DARP
from policies.surv.algorithms.ADD2.kruskal import Kruskal
from policies.surv.algorithms.ADD2.turns import turns
from policies.surv.algorithms.algo_utils import create_mesh, get_coordinate_array
from utils.coordinates import convert_boundary_lla_to_enu, convert_waypoints_enu_to_lla
from utils.recognition import get_avail_avs


def get_area_map(path, area=0, obs=-1):
    """
    Creates an array from a given png-image(path).
    :param path: path to the png-image
    :param area: non-obstacles tiles value; standard is 0
    :param obs: obstacle tiles value; standard is -1
    :return: an array of area(0) and obstacle(-1) tiles
    """
    le_map = np.array(Image.open(path))
    ma = np.array(le_map).mean(axis=2) != 0
    le_map = np.int8(np.zeros(ma.shape))
    le_map[ma] = area
    le_map[~ma] = obs
    return le_map


def get_area_indices(area, value, inv=False, obstacle=-1):
    """
    Returns area tiles indices that have value
    If inv(erted), returns indices that don't have value
    :param area: array with value and obstacle tiles
    :param value: searched tiles with value
    :param inv: if True: search will be inverted and index of non-value tiles will get returned
    :param obstacle: defines obstacle tiles
    :return:
    """
    try:
        value = int(value)
        if inv:
            return np.concatenate([np.where((area != value))]).T
        return np.concatenate([np.where((area == value))]).T
    except:
        mask = area == value[0]
        if inv:
            mask = area != value[0]
        for v in value[1:]:
            if inv:
                mask &= area != v
            else:
                mask |= area == v
        mask &= area != obstacle
        return np.concatenate([np.where(mask)]).T


class MultiRobotPathPlanner_ADD2:

    def __init__(
        self,
        nx,
        ny,
        notEqualPortions,
        initial_positions,
        portions,
        obs_pos,
        visualization,
        MaxIter=80000,
        CCvariation=0.01,
        randomLevel=0.0001,
        dcells=2,
        importance=False,
    ):
        # Initialize DARP
        self.darp_instance = DARP(
            nx,
            ny,
            notEqualPortions,
            initial_positions,
            portions,
            obs_pos,
            visualization,
            MaxIter=MaxIter,
            CCvariation=CCvariation,
            randomLevel=randomLevel,
            dcells=dcells,
            importance=importance,
        )

        # Divide areas based on robots initial positions
        self.DARP_success, self.iterations = self.darp_instance.divideRegions()

        # Check if solution was found
        if not self.DARP_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            # Iterate for 4 different ways to join edges in MST
            # Make convex hull -------------------------------
            resize_shape = (100, 100)
            resized_grid = resize(
                self.darp_instance.A,
                resize_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(int)
            hull_grids = []
            shull_grids = []  # shrinked hull grids
            spacing = 5.0
            circ_paths = []
            self.best_case = dict()
            self.best_case["paths"] = []
            for value in range(self.darp_instance.droneNo):
                mask = resized_grid == value
                coords = np.column_stack(np.where(mask))
                if coords.shape[0] < 3:
                    hull_grid = np.full(resized_grid.shape, -1)
                    hull_grids.append(hull_grid)
                    shull_grid = np.full(resized_grid.shape, -1)
                    shull_grids.append(shull_grid)

                hull = ConvexHull(coords)
                hull_coords = coords[hull.vertices]
                hull_grid = np.full(resized_grid.shape, -1)
                rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], resized_grid.shape)
                hull_grid[rr, cc] = value
                hull_grids.append(hull_grid)

                centroid = np.mean(coords, axis=0)
                vectors = coords - centroid
                distances = np.linalg.norm(vectors, axis=1, keepdims=True)
                unit_vectors = vectors / distances

                scoords = coords - unit_vectors * spacing
                shull = ConvexHull(scoords)
                shull_coords = scoords[shull.vertices]
                shull_grid = np.full(resized_grid.shape, -1)
                rr1, cc1 = polygon(shull_coords[:, 0], shull_coords[:, 1], resized_grid.shape)
                shull_grid[rr1, cc1] = value
                shull_grids.append(shull_grid)
                # Make convex hull -------------------------------

                # Generate Path ----------------------------------
                hull_edges = []
                for simplex in shull.simplices:
                    hull_edges.append((scoords[simplex[0]], scoords[simplex[1]]))
                mean = np.mean(scoords, axis=0)
                centroid_points = scoords - mean
                cov_matrix = np.cov(centroid_points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                major_axis = eigenvectors[:, np.argmax(eigenvalues)]

                perpendicular_axis = np.array([-major_axis[1], major_axis[0]])
                line_count = 20
                line_spacing = 5
                lines = []

                for i in range(-line_count // 2, line_count // 2 + 1):
                    offset = perpendicular_axis * (i * line_spacing)
                    point_on_line = mean + offset
                    # normal = np.array([-major_axis[1], major_axis[0]])
                    lines.append((point_on_line, major_axis))

                intersections = []
                for line_point, direction in lines:
                    direction = direction / np.linalg.norm(direction)
                    line_intersections = []
                    for edge_start, edge_end in hull_edges:
                        intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                        if intersection is not None:
                            line_intersections.append(intersection)
                    line_intersections = sorted(line_intersections, key=lambda x: x[1])
                    intersections.append(line_intersections)

                path = []
                reverse = False
                for line_intersections in intersections:
                    if reverse:
                        line_intersections = line_intersections[::-1]
                    path.extend(line_intersections)
                    reverse = not reverse
                path = np.array(path)
                split_index = int(np.floor(len(path) / 2))
                if split_index % 2 != 0:
                    front_path = path[: split_index + 1]
                    back_path = path[split_index + 1 :]
                else:
                    front_path = path[:split_index]
                    back_path = path[split_index:]
                circ_path = []
                max_iter = int(max(np.floor(len(front_path) / 2), np.floor(len(back_path) / 2)))
                reverse = False
                for i in range(0, max_iter):
                    if split_index % 2 != 0:
                        if not reverse:
                            if len(front_path) >= (2 * i + 1):
                                circ_path.extend([front_path[2 * i], front_path[2 * i + 1]])
                            if len(back_path) >= (2 * i + 1):
                                circ_path.extend([back_path[2 * i], back_path[2 * i + 1]])
                        else:
                            if len(front_path) >= (2 * i + 1):
                                circ_path.extend([front_path[2 * i + 1], front_path[2 * i]])
                            if len(back_path) >= (2 * i + 1):
                                circ_path.extend([back_path[2 * i + 1], back_path[2 * i]])
                    else:
                        if not reverse:
                            if len(front_path) >= (2 * i + 1):
                                circ_path.extend([front_path[2 * i], front_path[2 * i + 1]])
                            if len(back_path) >= (2 * i + 1):
                                circ_path.extend([back_path[2 * i + 1], back_path[2 * i]])
                        else:
                            if len(front_path) >= (2 * i + 1):
                                circ_path.extend([front_path[2 * i + 1], front_path[2 * i]])
                            if len(back_path) >= (2 * i + 1):
                                circ_path.extend([back_path[2 * i], back_path[2 * i + 1]])
                    reverse = not reverse

                circ_path = [
                    (
                        int(round(circ_path[i][0], 0)),
                        int(round(circ_path[i][1], 0)),
                        int(round(circ_path[i + 1][0], 0)),
                        int(round(circ_path[i + 1][1], 0)),
                    )
                    for i in range(len(circ_path) - 1)
                ]
                circ_paths.append(circ_path)
            self.best_case = turns(list(circ_paths))

    def CalcRealBinaryReg(self, BinaryRobotRegion, rows, cols):
        # 배열을 크기 2로 확장
        temp = np.kron(BinaryRobotRegion, np.ones((2, 2)))
        # temp 배열의 값을 bool 타입으로 변환하여 RealBinaryRobotRegion 생성
        RealBinaryRobotRegion = temp.astype(bool)
        return RealBinaryRobotRegion

    def calculateMSTs(self, BinaryRobotRegions, droneNo, rows, cols, mode):
        MSTs = []
        for r in range(droneNo):
            k = Kruskal(rows, cols)
            k.initializeGraph(BinaryRobotRegions[r, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        if np.linalg.det(matrix) == 0:
            return None
        t, u = np.linalg.solve(matrix, edge_start - line_point)
        if 0 <= u <= 1:
            return line_point + t * line_direction
        return None


def select(
    manager: Manager,
) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
    # 영역 좌표 변환
    top_l, top_r, bot_r, bot_l = convert_boundary_lla_to_enu(
        lat0=manager.lat0, lon0=manager.lon0, alt0=manager.alt0, boundary_as_lla=manager.mp_input.boundary
    )
    # 감시임무 가능한 비행체 반환
    avs_list, avs_pos_list = get_avail_avs(
        avs_to_avail_task=manager.avs_to_available_task_dict,
        criteria=["S", "s"],
        avs_info_dict=manager.mp_input.avs_info_dict,
    )
    # 임무계획 결과 초기화
    mp_result_dict: Dict[int, SurveillanceSelectionResult] = {}

    start_time = time.time()

    mesh_xs, mesh_ys = create_mesh(manager, top_l, top_r, bot_r, bot_l)
    org_xs = np.linspace(top_l[0], top_r[0], mesh_xs.shape[0])
    org_ys = np.linspace(bot_r[1], top_r[1], mesh_ys.shape[0])
    modi_xs = np.linspace(top_l[0], top_r[0], int(mesh_xs.shape[0] / 4))
    modi_ys = np.linspace(bot_r[1], top_r[1], int(mesh_ys.shape[0] / 4))
    mesh_x, mesh_y = np.meshgrid(modi_xs, modi_ys)
    grid = [mesh_x.shape[0], mesh_y.shape[0]]
    in_pos = []
    for avs_pos in avs_pos_list:
        xv = avs_pos[0]
        yv = avs_pos[1]
        x_idx = np.argmin(np.sqrt((mesh_x[0, :] - xv) ** 2))
        y_idx = np.argmin(np.sqrt((mesh_y[:, 0] - yv) ** 2))
        idx = y_idx * mesh_x.shape[0] + x_idx
        if idx in in_pos:
            x_id_temperature = 40
            while True:
                y_idx = random.randint(0, mesh_y.shape[0])
                x_idx = random.randint(0, min(mesh_y.shape[0], x_id_temperature))
                idx = y_idx * mesh_x.shape[0] + x_idx
                if (idx not in in_pos) and (idx < mesh_x.shape[0] * mesh_y.shape[0]):
                    break
        in_pos.append(y_idx * mesh_x.shape[0] + x_idx)
    obs_pos = []
    portions = ((1 / len(avs_list)) * np.ones(len(avs_list))).tolist()
    nep = True
    vis = False
    mrpp = MultiRobotPathPlanner_ADD2(grid[0], grid[1], nep, in_pos, portions, obs_pos, vis)
    # Convert DARP path to S9 path
    for i in range(len(avs_pos_list)):
        grid_path = mrpp.best_case.paths[i]
        xx = []
        yy = []
        for j in range(len(grid_path)):
            tmp_x = org_xs[grid_path[j][1] * 2]
            xx.append(tmp_x)
            tmp_y = org_ys[grid_path[j][0] * 2]
            yy.append(tmp_y)
        corner_pts = get_coordinate_array(xx, yy, 800.0)  # TODO
        mp_result_dict.update(
            {
                avs_list[i]: SurveillanceSelectionResult(
                    avs_id=avs_list[i],
                    sys_group_id=0,
                    smp_mode=1,
                    speed=27,  # TODO
                    waypoint_count=corner_pts.shape[0],
                    waypoints=convert_waypoints_enu_to_lla(
                        waypoint_array_as_enu=corner_pts, lat0=manager.lat0, lon0=manager.lon0, alt0=manager.alt0
                    ),
                )
            }
        )

    return time.time() - start_time, mp_result_dict, deepcopy(manager.mp_input)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-grid",
        default=(10, 10),
        type=int,
        nargs=2,
        help="Dimensions of the Grid (default: (10, 10))",
    )
    argparser.add_argument(
        "-obs_pos",
        default=[5, 6, 7],
        nargs="*",
        type=int,
        help="Obstacles Positions (default: None)",
    )
    argparser.add_argument(
        "-in_pos",
        default=[0, 3, 9],
        nargs="*",
        type=int,
        help="Initial Positions of the robots (default: (1, 3, 9))",
    )
    argparser.add_argument(
        "-nep",
        action="store_true",
        help="Not Equal Portions shared between the Robots in the Grid (default: False)",
    )
    argparser.add_argument(
        "-portions",
        default=[0.2, 0.3, 0.5],
        nargs="*",
        type=float,
        help="Portion for each Robot in the Grid (default: (0.2, 0.7, 0.1))",
    )
    argparser.add_argument(
        "-vis",
        default=False,
        action="store_true",
        help="Visualize results (default: False)",
    )
    args = argparser.parse_args()

    args.grid = [98, 98]
    args.in_pos = [10, 50, 90]
    args.obs_pos = [5, 15, 25, 35, 45, 55, 65, 75, 85]
    args.portions = [0.3, 0.3, 0.4]
    args.nep = True
    args.vis = True
    MultiRobotPathPlanner_ADD2(
        args.grid[0],
        args.grid[1],
        args.nep,
        args.in_pos,
        args.portions,
        args.obs_pos,
        args.vis,
    )
