import os
import random
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from policies.surv.algorithms.ADD.calculate_trajectories import CalculateTrajectories
from policies.surv.algorithms.ADD.darp import Darp
from policies.surv.algorithms.ADD.darp_old import DarpOld
from policies.surv.algorithms.ADD.kruskal import Kruskal
from policies.surv.algorithms.ADD.turns import Turns
from policies.surv.algorithms.ADD.visualization import visualize_paths
from policies.surv.algorithms.algo_utils import create_mesh, get_coordinate_array
from utils.coordinates import convert_boundary_lla_to_enu, convert_waypoints_enu_to_lla
from utils.recognition import get_avail_avs

seed = 50
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


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


class MultiRobotPathPlanner_ADD:

    def __init__(
        self,
        darp: str,
        num_rows: int,
        num_cols: int,
        not_equal: bool = False,
        init_positions: List[int] = [1, 2, 3, 4],
        portions: List[float] = [0.25, 0.25, 0.25, 0.25],
        obs_positions: List[int] = [5, 6, 7],
        visualization: bool = False,
        max_iter: int = 80000,
        cc_variation: float = 0.01,
        random_level: float = 0.0001,
        d_cells: int = 10,
        importance: bool = False,
    ):

        start_time = time.time()
        # Initialize DARP
        if darp == "old":
            self.darp = DarpOld(
                nx=num_rows,
                ny=num_cols,
                notEqualPortions=not_equal,
                given_initial_positions=init_positions,
                given_portions=portions,
                obstacles_positions=obs_positions,
                visualization=visualization,
                MaxIter=max_iter,
                CCvariation=cc_variation,
                randomLevel=random_level,
                dcells=d_cells,
                importance=importance,
            )
        elif darp == "new":
            self.darp = Darp(
                num_rows=num_rows,
                num_cols=num_cols,
                not_equal=not_equal,
                given_init_positions=init_positions,
                given_portions=portions,
                given_obs_positions=obs_positions,
                visualization=visualization,
                max_iter=max_iter,
                cc_variation=cc_variation,
                random_level=random_level,
                d_cells=d_cells,
                importance=importance,
            )
        else:
            raise ValueError(f"지원하지 않은 알고리즘: {darp}")

        # Divide areas based on robots initial positions
        self.is_success, self.iter = self.darp.divide_regions()

        # Check if solution was found
        if not self.is_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            # Iterate for 4 different ways to join edges in MST
            self.mode_to_drone_turns = []
            AllRealPaths_dict = {}
            subCellsAssignment_dict = {}
            for mode in range(4):
                MSTs = self.calculateMSTs(
                    self.darp.binary_uav_regions, self.darp.num_uavs, self.darp.num_rows, self.darp.num_cols, mode
                )
                AllRealPaths = []
                for r in range(self.darp.num_uavs):
                    ct = CalculateTrajectories(self.darp.num_rows, self.darp.num_cols, MSTs[r])
                    ct.initializeGraph(
                        self.CalcRealBinaryReg(
                            self.darp.binary_uav_regions[r], self.darp.num_rows, self.darp.num_cols
                        ),
                        True,
                    )
                    ct.RemoveTheAppropriateEdges()
                    ct.CalculatePathsSequence(
                        4 * self.darp.init_positions[r][0] * self.darp.num_cols + 2 * self.darp.init_positions[r][1]
                    )
                    AllRealPaths.append(ct.PathSequence)

                subCellsAssignment = np.zeros((2 * self.darp.num_rows, 2 * self.darp.num_cols))
                for i in range(self.darp.num_rows):
                    for j in range(self.darp.num_cols):
                        subCellsAssignment[2 * i][2 * j] = self.darp.assg_mat[i][j]
                        subCellsAssignment[2 * i + 1][2 * j] = self.darp.assg_mat[i][j]
                        subCellsAssignment[2 * i][2 * j + 1] = self.darp.assg_mat[i][j]
                        subCellsAssignment[2 * i + 1][2 * j + 1] = self.darp.assg_mat[i][j]

                drone_turns = Turns(AllRealPaths)
                drone_turns.count_turns()
                drone_turns.find_avg_and_std()
                self.mode_to_drone_turns.append(drone_turns)

                AllRealPaths_dict[mode] = AllRealPaths
                subCellsAssignment_dict[mode] = subCellsAssignment

            # Find mode with the smaller number of turns
            averge_turns = [x.avg for x in self.mode_to_drone_turns]
            self.min_mode = averge_turns.index(min(averge_turns))

            # Combine all modes to get one mode with the least available turns for each drone
            combined_modes_paths = []
            combined_modes_turns = []

            for r in range(self.darp.num_uavs):
                min_turns = sys.maxsize
                temp_path = []
                for mode in range(4):
                    if self.mode_to_drone_turns[mode].turns[r] < min_turns:
                        temp_path = self.mode_to_drone_turns[mode].paths[r]
                        min_turns = self.mode_to_drone_turns[mode].turns[r]
                combined_modes_paths.append(temp_path)
                combined_modes_turns.append(min_turns)

            self.best_case = Turns(combined_modes_paths)
            self.best_case.turns = combined_modes_turns
            self.best_case.find_avg_and_std()

            # Retrieve number of cells per robot for the best case configuration
            best_case_num_paths = [len(x) for x in self.best_case.paths]

            # visualize best case
            if self.darp.visualization:
                image = visualize_paths(
                    self.best_case.paths, subCellsAssignment_dict[self.min_mode], self.darp.num_uavs, self.darp.colors
                )
                image.visualize_paths("Combined Modes")

            self.execution_time = time.time() - start_time

            print(f"\nResults:")
            print(f"Number of cells per robot: {best_case_num_paths}")
            print(f"Minimum number of cells in robots paths: {min(best_case_num_paths)}")
            print(f"Maximum number of cells in robots paths: {max(best_case_num_paths)}")
            print(f"Average number of cells in robots paths: {np.mean(np.array(best_case_num_paths))}")
            print(f"\nTurns Analysis: {self.best_case}")

    def CalcRealBinaryReg(self, BinaryRobotRegion, rows, cols):
        temp = np.zeros((2 * rows, 2 * cols))
        RealBinaryRobotRegion = np.zeros((2 * rows, 2 * cols), dtype=bool)
        for i in range(2 * rows):
            for j in range(2 * cols):
                temp[i, j] = BinaryRobotRegion[(int(i / 2))][(int(j / 2))]
                if temp[i, j] == 0:
                    RealBinaryRobotRegion[i, j] = False
                else:
                    RealBinaryRobotRegion[i, j] = True
        return RealBinaryRobotRegion

    def calculateMSTs(self, BinaryRobotRegions, droneNo, rows, cols, mode):
        MSTs = []
        for r in range(droneNo):
            k = Kruskal(rows, cols)
            k.initializeGraph(BinaryRobotRegions[r, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs


def select(manager: Manager) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
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
    mrpp = MultiRobotPathPlanner_ADD(
        darp="new",
        num_rows=grid[0],
        num_cols=grid[1],
        not_equal=nep,
        init_positions=in_pos,
        portions=portions,
        obs_positions=obs_pos,
        visualization=vis,
    )
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
    roi_area = [[-5000, 5000, 0.0], [5000, 5000, 0.0], [5000, -5000, 0.0], [-5000, -5000, 0.0]]
    num = 50
    avail_avs = [i for i in range(1, num + 1)]

    def generate_random_points(N):
        # 주어진 직사각형 범위의 x, y 최소값과 최대값을 계산합니다.
        x_min = -5000
        x_max = 5000
        y_min = -5000
        y_max = 5000

        points = []
        for _ in range(N):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = 0.0  # z 좌표는 0으로 고정
            points.append([x, y, z])

        return points

    avail_avs_pos = generate_random_points(len(avail_avs))

    xs = np.linspace(roi_area[0][0], roi_area[1][0], 200)
    ys = np.linspace(roi_area[1][1], roi_area[2][1], 200)
    mesh_xs, mesh_ys = np.meshgrid(xs, ys)

    org_xs = np.linspace(roi_area[0][0], roi_area[1][0], int(np.shape(mesh_xs)[0]))
    org_ys = np.linspace(roi_area[1][1], roi_area[2][1], int(np.shape(mesh_ys)[0]))
    modi_xs = np.linspace(roi_area[0][0], roi_area[1][0], int(np.shape(mesh_xs)[0] / 4))
    modi_ys = np.linspace(roi_area[1][1], roi_area[2][1], int(np.shape(mesh_ys)[0] / 4))
    mesh_x, mesh_y = np.meshgrid(modi_xs, modi_ys)

    grid = [np.shape(mesh_x)[0], np.shape(mesh_y)[0]]
    in_pos = []
    tmp_xid = []
    tmp_yid = []
    coordinates = []
    for p in range(0, np.shape(avail_avs_pos)[0]):
        xv = avail_avs_pos[p][0]
        yv = avail_avs_pos[p][1]
        x_idx = np.argmin(np.sqrt((mesh_x[0, :] - xv) ** 2))
        y_idx = np.argmin(np.sqrt((mesh_y[:, 0] - yv) ** 2))
        coordinates.append([x_idx, y_idx])
        idx = y_idx * np.shape(mesh_x)[0] + x_idx
        if idx in in_pos:
            while True:
                y_idx = random.randint(0, np.shape(mesh_y)[0])
                idx = y_idx * np.shape(mesh_x)[0] + x_idx
                if idx not in in_pos:
                    break
        in_pos.append(y_idx * np.shape(mesh_x)[0] + x_idx)
    obs_pos = []
    portions = ((1 / len(avail_avs)) * np.ones(len(avail_avs))).tolist()
    nep = True
    vis = False

    MultiRobotPathPlanner_ADD(
        darp="copy",
        num_rows=grid[0],
        num_cols=grid[1],
        not_equal=nep,
        init_positions=in_pos,
        portions=portions,
        obs_positions=[],
        visualization=vis,
    )
