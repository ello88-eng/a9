import random
import time
from copy import deepcopy
from typing import Dict

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from skimage.draw import polygon
from skimage.transform import resize

from commu.sys.receiver.data.gcs2mps import AvsInfo
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from config.avs_config import SmpMode
from config.mp_config import (
    ALT_REF,
    D_CELLS,
    FOV_LEN,
    MIN_CLEARANCE,
    SURV_TEMPERATURE,
    SURV_TURN_RADIUS,
)
from manager.manager import Manager
from policies.surv.algorithms.ADD4.darp import Darp
from policies.surv.algorithms.ADD4.kruskal import Kruskal
from policies.surv.algorithms.ADD4.polygon_grid_projector import PolygonGridProjector
from policies.surv.algorithms.ADD4.turns import turns
from policies.surv.algorithms.algo_utils import create_mesh, get_coordinate_array
from utils.coordinates import true_round
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
        value = true_round(value)
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


class MultiRobotPathPlanner_ADD4:
    def __init__(
        self,
        nx,
        ny,
        notEqualPortions,
        initial_positions,
        portions,
        obs_pos,
        visualization,
        map_roi,
        turn_radius,
        fov_len,
        MaxIter=80000,
        CCvariation=0.01,
        randomLevel=0.0001,
        dcells=2,
        importance=False,
    ):
        # Initialize DARP
        self.darp_instance = Darp(
            nx,
            ny,
            notEqualPortions,
            initial_positions,
            portions,
            obs_pos,
            visualization,
            max_iter=MaxIter,
            cc_variation=CCvariation,
            random_level=randomLevel,
            d_cells=dcells,
            importance=importance,
        )

        # Divide areas based on robots initial positions
        self.DARP_success, self.iterations = self.darp_instance.divide_regions()

        # Check if solution was found
        if not self.DARP_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            print("DARP Success.")
            # bin size calculation
            try:
                l_bin = self.calculate_grid_size(map_roi, nx, ny)
            except:
                l_bin = 50
            finally:
                l_bin = 50
            # spaicing_alpha (convex hull shrink)

            s_alpha = turn_radius / l_bin
            s_alpha = 0

            # Iterate for 4 different ways to join edges in MST
            # Make convex hull -------------------------------
            resize_shape = (100, 100)
            resized_grid = resize(
                self.darp_instance.assg_mat, resize_shape, order=0, preserve_range=True, anti_aliasing=False
            ).astype(int)
            hull_grids = []
            shull_grids = []  # shrinked hull grids
            spacing = s_alpha
            circ_paths = []
            self.best_case = dict()
            self.best_case["paths"] = []
            for value in range(self.darp_instance.num_uavs):
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

                # 임무가능영역 확장 또는 축소 선택
                # 임무가능영역 축소
                scoords = coords - unit_vectors * spacing
                # 임무가능영역 확장
                # scoords = coords + unit_vectors * spacing
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
                minor_axis = eigenvectors[:, np.argmin(eigenvalues)]

                # max grid num in minor axis direction
                hull_points = scoords[shull.vertices]
                projected_points = hull_points @ minor_axis
                max_grid = np.max(projected_points) - np.min(projected_points)
                overlap_margin = 0.2
                s_beta = (fov_len / l_bin) * (1 - overlap_margin)

                perpendicular_axis = np.array([-major_axis[1], major_axis[0]])
                line_count = true_round(max_grid / s_beta + 10)
                line_spacing = s_beta
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
                split_index = true_round(np.floor(len(path) / 2))
                if split_index % 2 != 0:
                    front_path = path[: split_index + 1]
                    back_path = path[split_index + 1 :]
                else:
                    front_path = path[:split_index]
                    back_path = path[split_index:]
                circ_path = []
                max_iter = true_round(max(np.floor(len(front_path) / 2), np.floor(len(back_path) / 2)))
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

                # circ_path = np.array(circ_path)
                circ_path = [
                    (
                        true_round(round(circ_path[i][0], 0)),
                        true_round(round(circ_path[i][1], 0)),
                        true_round(round(circ_path[i + 1][0], 0)),
                        true_round(round(circ_path[i + 1][1], 0)),
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

    def calculate_grid_size(self, corners, nx, ny):
        left_top, right_top, right_bottom, left_bottom = corners.values()

        height_left = euclidean(left_top, left_bottom)
        height_right = euclidean(right_top, right_bottom)
        grid_height = min(height_left, height_right) / ny

        width_top = euclidean(left_top, right_top)
        width_bottom = euclidean(left_bottom, right_bottom)
        grid_width = min(width_top, width_bottom) / nx

        return np.round(np.mean([grid_height, grid_width]))


def select(manager: Manager) -> tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
    # 알고리즘 시작 시간
    start_time = time.time()

    # 영역 좌표 변환
    # top_left, top_right, bottom_right, bottom_left = convert_boundary_lla_to_enu(
    #     lat0=manager.lat0, lon0=manager.lon0, alt0=manager.alt0, boundary_as_lla=manager.mp_input.boundary.vertices
    # )
    top_left, top_right, bottom_right, bottom_left = manager.mp_input.boundary.vertices

    # 감시임무 가능한 비행체 반환
    # ADD v4 알고리즘은 LLA 좌표 사용
    avs_list, avs_pos_list = get_avail_avs(
        avs_to_avail_task=manager.avs_to_available_task_dict,
        criteria=["S", "s"],
        avs_info_dict=manager.mp_input_lla.avs_info_dict,
    )
    # 임무계획 결과 초기화
    mp_result_dict: Dict[int, SurveillanceSelectionResult] = {}

    # 감시 선별을 위한 입력정보 전처리
    mesh_xs, mesh_ys = create_mesh(
        manager=manager, top_left=top_left, top_right=top_right, bottom_right=bottom_right, bottom_left=bottom_left
    )
    org_xs = np.linspace(top_left[1], top_right[1], mesh_xs.shape[0])
    org_ys = np.linspace(bottom_right[0], top_right[0], mesh_ys.shape[0])
    modi_xs = np.linspace(top_left[1], top_right[1], true_round(mesh_xs.shape[0] / 4))
    modi_ys = np.linspace(bottom_right[0], top_right[0], true_round(mesh_ys.shape[0] / 4))
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
            while True:
                random.seed(time.time())
                x_rand = random.randint(5, min(mesh_y.shape[0], SURV_TEMPERATURE))
                y_rand = random.randint(5, mesh_y.shape[0])
                if random.random() < 0.5:
                    x_idx += x_rand
                    y_idx += y_rand
                else:
                    x_idx -= x_rand
                    y_idx -= y_rand
                idx = y_idx * mesh_x.shape[0] + x_idx
                if (idx not in in_pos) and (idx < mesh_x.shape[0] * mesh_y.shape[0]):
                    break
                else:
                    x_idx = np.argmin(np.sqrt((mesh_x[0, :] - xv) ** 2))
                    y_idx = np.argmin(np.sqrt((mesh_y[:, 0] - yv) ** 2))
        in_pos.append(true_round(y_idx * mesh_x.shape[0] + x_idx))
    print(in_pos)

    # 클래스 초기화
    pgp = PolygonGridProjector(grid_size=grid[0])

    # obs_pos
    # `use_hull`은 필요 시 True, # Default : False
    obs_pos, _, _ = pgp.get_obstacle_position_from_polygon_latlon(
        [top_left, bottom_left, top_right, bottom_right],
        manager.mp_input.polygon_area.vertices,
        manager.mp_input.grid_coverage,
        use_hull=False,
    )
    # in_pos를 자유공간으로 투영 (장애물에서 3칸 이상)
    in_pos_safe = pgp.project_in_pos_with_clearance(obs_pos=obs_pos, in_pos=in_pos, min_clearance=MIN_CLEARANCE)
    print(in_pos_safe)

    portions = ((1 / len(avs_list)) * np.ones(len(avs_list))).tolist()
    nep = True
    vis = False
    map_roi = dict()
    map_roi["left_top"] = top_left
    map_roi["right_top"] = top_right
    map_roi["right_bottom"] = bottom_right
    map_roi["left_bottom"] = bottom_left
    mrpp = MultiRobotPathPlanner_ADD4(
        nx=grid[0],
        ny=grid[1],
        notEqualPortions=nep,
        initial_positions=in_pos_safe,
        portions=portions,
        obs_pos=obs_pos,
        visualization=vis,
        map_roi=map_roi,
        turn_radius=SURV_TURN_RADIUS,
        fov_len=FOV_LEN,
        dcells=D_CELLS,
    )
    # Convert DARP path to S9 path
    for i in range(len(avs_pos_list)):
        grid_path = mrpp.best_case.paths[i]
        xx = []
        yy = []
        for j in range(len(grid_path)):
            tmp_x = org_xs[grid_path[j][0] * 2]
            # tmp_x = org_xs[grid_path[j][1]]
            xx.append(tmp_x)
            tmp_y = org_ys[grid_path[j][1] * 2]
            # tmp_y = org_ys[grid_path[j][0]]
            yy.append(tmp_y)
        # waypoint_array = get_corner_coordinate_array(xx, yy, ALT_REF)
        waypoint_array = get_coordinate_array(yy, xx, ALT_REF)
        # waypoint_array_as_lla = convert_waypoints_enu_to_lla(
        #     waypoint_array_as_enu=waypoint_array, lat0=manager.lat0, lon0=manager.lon0, alt0=manager.alt0
        # )
        waypoint_array[:, -1] = manager.mp_input.mission_init_info.surv_alt

        avs_info: AvsInfo = manager.mp_input.avs_info_dict.get(avs_list[i])
        mp_result_dict.update(
            {
                avs_info.avs_id: SurveillanceSelectionResult(
                    avs_id=avs_info.avs_id,
                    system_group_id=avs_info.system_group_id,
                    smp_mode=SmpMode.SURV.value,
                    speed=manager.mp_input.mission_init_info.speed,
                    waypoint_count=waypoint_array.shape[0],
                    waypoints=waypoint_array,
                )
            }
        )

    return time.time() - start_time, mp_result_dict, deepcopy(manager.mp_input)
