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
import matplotlib.pyplot as plt
import math

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


class MultiRobotPathPlanner_ADD5:
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
        self.turn_radius = turn_radius
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
            self.hull_grids = []
            self.shull_grids = []  # shrinked hull grids
            spacing = s_alpha
            self.circ_paths = []
            self.best_case = dict()
            self.best_case["paths"] = []
            # temp =======================
            all_coords_list = []
            all_hull_coords_list = []  
            all_paths_list = []
            # temp =======================

            for value in range(self.darp_instance.num_uavs):
                mask = resized_grid == value
                coords = np.column_stack(np.where(mask))

                if coords.shape[0] < 3:
                    self.circ_paths.append([])
                    self.hull_grids.append(np.full(resized_grid.shape, -1))
                    self.shull_grids.append(np.full(resized_grid.shape, -1))
                    continue

                hull = ConvexHull(coords)
                hull_coords = coords[hull.vertices]
                hull_grid = np.full(resized_grid.shape, -1)
                rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], resized_grid.shape)
                hull_grid[rr, cc] = value
                self.hull_grids.append(hull_grid)

                shull_grid = np.full(resized_grid.shape, -1)
                shull_grid[rr, cc] = value
                self.shull_grids.append(shull_grid)

                hull_edges = []
                for simplex in hull.simplices:
                    hull_edges.append((coords[simplex[0]], coords[simplex[1]]))
                
                mean = np.mean(coords, axis=0)
                centroid_points = coords - mean
                major_axis = np.array([1.0, 0.0]) # 가로 방향은 x, y 위치를 바꾸기
                minor_axis = np.array([0.0, 1.0])

                hull_points = coords[hull.vertices]
                projected_points = hull_points @ minor_axis
                max_grid = np.max(projected_points) - np.min(projected_points)
                overlap_margin = 0.2
                s_beta = (fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1

                perpendicular_axis = np.array([-major_axis[1], major_axis[0]])
                line_count = int(max_grid / s_beta + 1) if s_beta > 0 else 1
                line_spacing = s_beta
                lines = []

                for i in range(-line_count // 2, line_count // 2 + 1):
                    offset = perpendicular_axis * (i * line_spacing)
                    point_on_line = mean + offset
                    lines.append((point_on_line, major_axis))

                intersections = []
                for line_point, direction in lines:
                    direction = direction / np.linalg.norm(direction)
                    line_intersections = []
                    for edge_start, edge_end in hull_edges:
                        intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                        if intersection is not None:
                            line_intersections.append(intersection)
                    
                    if len(line_intersections) > 1:
                        line_intersections.sort(key=lambda p: np.dot(p - line_point, direction))
                        intersections.append((line_intersections[0], line_intersections[-1]))
                intersections_improved = self.improved_intersection_logic(lines, hull_edges, hull_coords)
                # self.compare_intersections(intersections, intersections_improved)

                path = []
                if len(intersections_improved) > 0:
                    mid_point = len(intersections_improved) // 2
                    max_lines = max(mid_point, len(intersections_improved) - mid_point)
                    
                    for i in range(max_lines):
                        # 1파트 라인
                        if i < mid_point:
                            pair1 = intersections_improved[i]
                            path.extend([pair1[0], pair1[1]])
                        
                        # 2파트 라인
                        part2_idx = mid_point + i
                        if part2_idx < len(intersections_improved):
                            pair2 = intersections_improved[part2_idx]
                            
                            # 남는 라인들은 정방향으로 처리
                            if i >= mid_point:  # 1파트가 끝난 후 남은 라인들
                                path.extend([pair2[0], pair2[1]])  # 정방향
                            else:
                                path.extend([pair2[1], pair2[0]])  # 역방향

                # 데이터 저장 (temp)
                all_coords_list.append(coords)
                all_hull_coords_list.append(hull_coords)
                all_paths_list.append(path)
                # self.plot_all_robots_paths_separately(all_coords_list, all_hull_coords_list, all_paths_list)
                
                self.circ_paths.append(path)
            
            self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

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
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            if 0 <= u <= 1:
                return line_point + t * line_direction
        except np.linalg.LinAlgError:
            return None
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

    # 기존 코드의 intersection 부분을 다음과 같이 수정:
    def improved_intersection_logic(self, lines, hull_edges, hull_coords):
        """
        개선된 교점 계산 로직
        """
        intersections = []
        
        for line_point, direction in lines:
            direction = direction / np.linalg.norm(direction)
            line_intersections = []
            
            # 1. 기존 방식으로 모든 교점 찾기
            for edge_start, edge_end in hull_edges:
                intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                if intersection is not None:
                    line_intersections.append(intersection)
            
            # 2. Hull 꼭짓점과 가까운 교점들 필터링
            line_intersections = self.filter_hull_vertex_intersections(
                line_intersections, hull_coords, min_distance_threshold=1.5
            )
            
            # 3. 중복 교점 제거
            line_intersections = self.remove_duplicate_intersections(
                line_intersections, min_distance_between_points=0.5
            )
            
            # 4. 정확히 2개의 교점만 유지
            final_intersections = self.ensure_exactly_two_intersections(
                line_intersections, line_point, direction
            )
            
            # 5. 유효한 교점 쌍만 추가
            if len(final_intersections) == 2:
                # 라인 방향에 따라 정렬하여 일관된 순서 보장
                projections = [np.dot(p - line_point, direction) for p in final_intersections]
                if projections[0] > projections[1]:
                    final_intersections = final_intersections[::-1]
                
                intersections.append(tuple(final_intersections))
        
        return intersections

    def compare_intersections(self, intersections, intersections_improved):
        """
        intersections와 intersections_improved를 간단히 비교 시각화
        
        Args:
            intersections: 기존 방식 교점 pairs [(point1, point2), ...]
            intersections_improved: 개선된 방식 교점 pairs [(point1, point2), ...]
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 왼쪽: 기존 intersections
        ax1.set_title(f'Original Intersections\n({len(intersections)} pairs)')
        for i, pair in enumerate(intersections):
            x_vals = [pair[0][0], pair[1][0]]
            y_vals = [pair[0][1], pair[1][1]]
            
            # 교점 표시
            ax1.scatter(x_vals, y_vals, color='blue', s=60, marker='o', zorder=5)
            # 교점 연결선
            ax1.plot(x_vals, y_vals, 'blue', linewidth=2, alpha=0.7, zorder=3)
            # 선 번호
            ax1.annotate(f'{i}', ((x_vals[0] + x_vals[1]) / 2, (y_vals[0] + y_vals[1]) / 2),
                         fontsize=10, color='darkblue', ha='center', zorder=6)
            # 점 번호 (pair별로 0,1 표시)
            ax1.annotate(f'{i}-0', (x_vals[0], y_vals[0]), fontsize=8, color='navy',
                         xytext=(3, 3), textcoords='offset points')
            ax1.annotate(f'{i}-1', (x_vals[1], y_vals[1]), fontsize=8, color='navy',
                         xytext=(3, -10), textcoords='offset points')
        
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 오른쪽: 개선된 intersections_improved
        ax2.set_title(f'Improved Intersections\n({len(intersections_improved)} pairs)')
        for i, pair in enumerate(intersections_improved):
            x_vals = [pair[0][0], pair[1][0]]
            y_vals = [pair[0][1], pair[1][1]]
            
            # 교점 표시
            ax2.scatter(x_vals, y_vals, color='green', s=60, marker='o', zorder=5)
            # 교점 연결선
            ax2.plot(x_vals, y_vals, 'green', linewidth=2, alpha=0.7, zorder=3)
            # 선 번호
            ax2.annotate(f'{i}', ((x_vals[0] + x_vals[1]) / 2, (y_vals[0] + y_vals[1]) / 2),
                         fontsize=10, color='darkgreen', ha='center', zorder=6)
            # 점 번호
            ax2.annotate(f'{i}-0', (x_vals[0], y_vals[0]), fontsize=8, color='darkgreen',
                         xytext=(3, 3), textcoords='offset points')
            ax2.annotate(f'{i}-1', (x_vals[1], y_vals[1]), fontsize=8, color='darkgreen',
                         xytext=(3, -10), textcoords='offset points')
        
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        # 간단한 통계 출력
        print(f"Original: {len(intersections)} pairs")
        print(f"Improved: {len(intersections_improved)} pairs")
        print(f"Difference: {len(intersections_improved) - len(intersections)}")
        
        plt.show()

    def plot_single_robot_path(self, coords, hull_coords, path, robot_value):
        """
        단일 로봇의 경로를 화살표로 간단하게 시각화
        
        Args:
            coords: 로봇 영역의 격자 좌표들
            hull_coords: Hull 꼭짓점 좌표들  
            path: 계산된 경로 점들 [(x1,y1), (x2,y2), ...]
            robot_value: 로봇 번호 (0, 1, 2...)
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        robot_color = colors[robot_value % len(colors)]
        
        # 영역 점들 표시
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.4, s=20, 
                  color=robot_color, label=f'UAV {robot_value+1} Region')
        
        # Hull 경계 그리기
        hull_x = [hull_coords[i][0] for i in range(len(hull_coords))] + [hull_coords[0][0]]
        hull_y = [hull_coords[i][1] for i in range(len(hull_coords))] + [hull_coords[0][1]]
        ax.plot(hull_x, hull_y, 'black', linewidth=2, label='Convex Hull')
        
        # Hull 꼭짓점 표시
        ax.scatter(hull_coords[:, 0], hull_coords[:, 1], color='red', s=100, 
                  marker='s', edgecolor='black', linewidth=1, label='Hull Vertices', zorder=10)
        
        # 경로를 화살표로 표시
        if len(path) > 1:
            for i in range(len(path) - 1):
                start_point = path[i]
                end_point = path[i + 1]
                
                # 화살표 그리기
                ax.annotate('', xy=end_point, xytext=start_point,
                           arrowprops=dict(arrowstyle='->', color='darkred', 
                                         lw=1.5, alpha=0.8))
                
                # 시작점 번호 표시 (너무 많으면 일부만)
                if i % 3 == 0 or i < 5:  # 처음 5개 + 3개마다 하나씩
                    ax.annotate(f'{i}', start_point, fontsize=16, color='darkred',
                               xytext=(2, 2), textcoords='offset points', zorder=9)
            
            # 마지막 점 번호
            ax.annotate(f'{len(path)-1}', path[-1], fontsize=16, color='darkred',
                       xytext=(2, 2), textcoords='offset points', zorder=9)
        
        ax.set_title(f'UAV {robot_value+1} Path ({len(path)} points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"UAV {robot_value+1}: {len(path)} path points")
    
    def plot_all_robots_paths_separately(self, all_coords, all_hull_coords, all_paths):
        """
        모든 로봇의 경로를 각각 별도 그림으로 표시
        
        Args:
            all_coords: [coords_robot0, coords_robot1, coords_robot2, ...]
            all_hull_coords: [hull_coords_robot0, hull_coords_robot1, ...]  
            all_paths: [path_robot0, path_robot1, path_robot2, ...]
        """
        
        for robot_idx in range(len(all_paths)):
            if len(all_paths[robot_idx]) > 0:
                self.plot_single_robot_path(
                    all_coords[robot_idx], 
                    all_hull_coords[robot_idx], 
                    all_paths[robot_idx], 
                    robot_idx
                )



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
