import random
import time
from copy import deepcopy
from typing import Dict

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
# skimage.draw에서 disk 함수를 추가로 import 합니다.
from skimage.draw import polygon, disk
from skimage.transform import resize
import matplotlib.pyplot as plt
import math


# ==========================================================================================
# 기존 코드의 의존성을 해결하기 위한 모의(mock) 클래스 및 함수
# ==========================================================================================
class MockManager:
    def __init__(self):
        class Boundary:
            vertices = [
                (36.3, 127.3),
                (36.3, 127.4),
                (36.4, 127.4),
                (36.4, 127.3),
            ]
        class PolygonArea:
            # 이 부분은 이제 사용되지 않습니다.
            vertices = []
        class MissionInitInfo:
            surv_alt = 100
            speed = 10
        class AvsInfo:
            def __init__(self, avs_id, system_group_id):
                self.avs_id = avs_id
                self.system_group_id = system_group_id

        self.mp_input = type("MissionPlanInput", (), {})()
        self.mp_input.boundary = Boundary()
        self.mp_input.polygon_area = PolygonArea()
        self.mp_input.mission_init_info = MissionInitInfo()
        self.mp_input.avs_info_dict = {
            1: AvsInfo(1, 1),
            2: AvsInfo(2, 1),
            3: AvsInfo(3, 1),
        }
        self.avs_to_available_task_dict = {1: ["S"], 2: ["S"], 3: ["S"]}
        self.mp_input_lla = self
        self.avs_info_dict = {
            1: {"pos": (36.31, 127.31, 100)},
            2: {"pos": (36.38, 127.32, 100)},
            3: {"pos": (36.35, 127.38, 100)},
        }

class SmpMode:
    SURV = "SURV"

class SurveillanceSelectionResult:
    def __init__(self, avs_id, system_group_id, smp_mode, speed, waypoint_count, waypoints):
        self.avs_id = avs_id
        self.system_group_id = system_group_id
        self.smp_mode = smp_mode
        self.speed = speed
        self.waypoint_count = waypoint_count
        self.waypoints = waypoints

# 상수 정의
ALT_REF = 100
FOV_LEN = 50
SURV_TURN_RADIUS = 25

def get_avail_avs(avs_to_avail_task, criteria, avs_info_dict):
    avs_list = list(avs_to_avail_task.keys())
    avs_pos_list = [avs_info_dict.get(i)["pos"][:2] for i in avs_list]
    return avs_list, avs_pos_list

def create_mesh(manager, top_left, top_right, bottom_right, bottom_left):
    xs = np.linspace(top_left[1], top_right[1], 100)
    ys = np.linspace(bottom_right[0], top_right[0], 100)
    return np.meshgrid(xs, ys)

def get_coordinate_array(xs, ys, alt):
    return np.vstack([xs, ys, np.full_like(xs, alt)]).T

# ==========================================================================================
# ✨ 새로운 기능: 무작위 장애물 영역 생성 함수
# ==========================================================================================
def generate_random_obstacle_shape(grid_size: int) -> list:
    """
    그리드 내에 무작위 모양(원 또는 다각형)의 장애물 영역을 생성합니다.

    :param grid_size: 그리드의 한 변의 크기
    :return: 장애물 위치 인덱스 리스트
    """
    shape_type = random.choice(['polygon', 'circle'])
    obs_coords = []

    if shape_type == 'polygon':
        # 3~8개의 꼭짓점을 가진 다각형을 생성합니다.
        num_vertices = random.randint(3, 8)
        center_x = grid_size / 2
        center_y = grid_size / 2
        
        # 다각형이 너무 작거나 크지 않도록 반지름 범위를 설정합니다.
        radius_min = grid_size / 6
        radius_max = grid_size / 3
        
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        vertices_r = []
        vertices_c = []
        for angle in angles:
            radius = random.uniform(radius_min, radius_max)
            r = int(center_y + radius * math.sin(angle))
            c = int(center_x + radius * math.cos(angle))
            vertices_r.append(max(0, min(grid_size - 1, r)))
            vertices_c.append(max(0, min(grid_size - 1, c)))
        
        rr, cc = polygon(vertices_r, vertices_c)
        obs_coords = np.array([rr, cc]).T

    elif shape_type == 'circle':
        # 무작위 중심점과 반지름을 가진 원을 생성합니다.
        radius = random.randint(int(grid_size / 8), int(grid_size / 4))
        center_r = random.randint(radius, grid_size - 1 - radius)
        center_c = random.randint(radius, grid_size - 1 - radius)
        
        rr, cc = disk((center_r, center_c), radius)
        obs_coords = np.array([rr, cc]).T
    
    # 장애물 좌표를 1차원 인덱스로 변환하여 반환합니다.
    obs_pos = [r * grid_size + c for r, c in obs_coords]
    return obs_pos

# ==========================================================================================
# 원본 코드 (일부 수정됨)
# ==========================================================================================

class Darp:
    def __init__(self, nx, ny, notEqualPortions, initial_positions, portions, obs_pos, visualization,
                 max_iter, cc_variation, random_level, d_cells, importance):
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.max_iter = max_iter

        self.grid = np.zeros((ny, nx))
        for obs in obs_pos:
            row, col = obs // nx, obs % nx
            if 0 <= row < ny and 0 <= col < nx:
                self.grid[row, col] = -1

        self.assg_mat = np.full((ny, nx), -1, dtype=int)
        free_cells = np.argwhere(self.grid != -1)
        
        initial_coords = np.array([[pos // nx, pos % nx] for pos in initial_positions])
        
        for y, x in free_cells:
            distances = np.linalg.norm(initial_coords - np.array([y, x]), axis=1)
            self.assg_mat[y, x] = np.argmin(distances)

    def divide_regions(self):
        print("Using Voronoi partitioning for region division.")
        return True, 0

class PolygonGridProjector:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def project_in_pos_with_clearance(self, obs_pos, in_pos, min_clearance):
        return in_pos

def turns(paths):
    return {"paths": paths}

class MultiRobotPathPlanner_ADD7:
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
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        
        # 기존 DARP 대신 간소화된 처리
        self.DARP_success = True  # 항상 성공으로 설정
        self.iterations = 0
        
        print("Using simplified rectangular patch division instead of DARP.")
        
        # Open space 계산
        total_cells = set(range(nx * ny))
        obs_set = set(obs_pos)
        self.open_space_indices = list(total_cells - obs_set)
        self.open_space_coords = np.array([[idx % nx, idx // nx] for idx in self.open_space_indices])
        
        # 간소화된 영역 분할
        self.rectangular_patches = self.divide_into_rectangular_patches()
        
        # 분할 결과 시각화 (디버깅용)
        self.plot_rectangular_patches()
        
        # 기존 결과 구조 유지
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = dict()
        self.best_case["paths"] = []
        
        # 경로 생성 (기존 ConvexHull + intersection 로직 활용)
        self.generate_paths_for_patches()

    def divide_into_rectangular_patches(self):
        """Open space를 UAV 수와 동일한 개수의 직사각형으로 분할"""
        if len(self.open_space_coords) == 0:
            return [[] for _ in range(self.num_uavs)]
        
        # Open space의 bounding box 계산
        min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
        min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
        
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
        
        # UAV 수만큼 직사각형으로 분할 (긴 직선 경로를 위해 가로로 분할)
        patch_width = total_width / self.num_uavs
        patch_height = total_height
        
        # UAV 수만큼 동일한 크기의 직사각형 생성
        patches = []
        
        for uav_idx in range(self.num_uavs):
            patch_min_x = min_x + uav_idx * patch_width
            patch_max_x = min_x + (uav_idx + 1) * patch_width
            patch_min_y = min_y
            patch_max_y = max_y
            
            rectangle_info = {
                'bounds': (patch_min_x, patch_max_x, patch_min_y, patch_max_y),
                'corners': np.array([
                    [patch_min_x, patch_min_y],  # 좌하
                    [patch_max_x, patch_min_y],  # 우하
                    [patch_max_x, patch_max_y],  # 우상
                    [patch_min_x, patch_max_y]   # 좌상
                ]),
                'edges': [
                    (np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_min_y])),  # 하단
                    (np.array([patch_max_x, patch_min_y]), np.array([patch_max_x, patch_max_y])),  # 우측
                    (np.array([patch_max_x, patch_max_y]), np.array([patch_min_x, patch_max_y])),  # 상단
                    (np.array([patch_min_x, patch_max_y]), np.array([patch_min_x, patch_min_y]))   # 좌측
                ]
            }
            patches.append(rectangle_info)
        
        return patches

    def generate_paths_for_patches(self):
        """각 직사각형 패치에 대해 직접 intersection 계산으로 경로 생성"""
        l_bin = self.calculate_grid_size(None, self.nx, self.ny)
        
        for value in range(self.num_uavs):
            rectangle_info = self.rectangular_patches[value]
            
            if not rectangle_info or 'bounds' not in rectangle_info:
                self.circ_paths.append([])
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
                continue
            
            # 직사각형 정보 추출
            min_x, max_x, min_y, max_y = rectangle_info['bounds']
            rectangle_edges = rectangle_info['edges']
            
            # 그리드 생성 (기존 구조 유지용)
            patch_grid = np.full((self.ny, self.nx), -1)
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if 0 <= x < self.nx and 0 <= y < self.ny:
                        patch_grid[y, x] = value
            
            self.hull_grids.append(patch_grid)
            self.shull_grids.append(patch_grid.copy())
            
            # 직사각형에서 직접 intersection 경로 생성
            path = self.generate_rectangle_intersection_path(rectangle_info, l_bin)
            self.circ_paths.append(path)
        
        # 결과 포맷팅 (기존 구조 유지)
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def generate_rectangle_intersection_path(self, rectangle_info, l_bin):
        """직사각형 경계에서 직접 intersection을 계산하여 경로 생성"""
        min_x, max_x, min_y, max_y = rectangle_info['bounds']
        
        # 직사각형 중심점
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 긴 직선 경로를 위해 주축을 가로 방향으로 설정
        major_axis = np.array([1.0, 0.0])  # 가로 방향 (X축)
        
        # 직사각형의 세로 길이
        rectangle_height = max_y - min_y
        
        # 스캔 라인 간격 계산
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        # 스캔 라인 개수
        line_count = int(rectangle_height / s_beta) + 1
        if line_count < 1:
            line_count = 1
        
        # 실제 라인 간격 (직사각형 높이에 맞춤)
        if line_count > 1:
            actual_spacing = rectangle_height / (line_count - 1)
        else:
            actual_spacing = 0
        
        # 지그재그 패턴으로 경로 생성 (직사각형 경계 사용)
        path = []
        
        for i in range(line_count):
            line_y = min_y + i * actual_spacing
            
            # 직사각형의 왼쪽과 오른쪽 경계점
            left_point = np.array([min_x, line_y])
            right_point = np.array([max_x, line_y])
            
            # 지그재그 패턴
            if i % 2 == 0:  # 짝수 번째: 왼쪽 → 오른쪽
                path.extend([left_point, right_point])
            else:  # 홀수 번째: 오른쪽 → 왼쪽
                path.extend([right_point, left_point])
        
        return path

    def plot_rectangular_patches(self):
        """직사각형 분할 결과를 open space와 함께 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Open space 점들 표시
        if len(self.open_space_coords) > 0:
            ax.scatter(self.open_space_coords[:, 0], self.open_space_coords[:, 1], 
                      c='lightblue', s=10, alpha=0.6, label='Open Space')
        
        # 장애물 점들 표시
        if len(self.obs_pos) > 0:
            obs_coords = np.array([[idx % self.nx, idx // self.nx] for idx in self.obs_pos])
            ax.scatter(obs_coords[:, 0], obs_coords[:, 1], 
                      c='red', s=10, alpha=0.8, label='Obstacles')
        
        # 각 UAV의 직사각형 패치 표시
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for uav_idx, rectangle_info in enumerate(self.rectangular_patches):
            if rectangle_info and 'bounds' in rectangle_info:
                min_x, max_x, min_y, max_y = rectangle_info['bounds']
                
                # 직사각형 그리기
                rect = patches.Rectangle(
                    (min_x, min_y), 
                    max_x - min_x, 
                    max_y - min_y,
                    linewidth=2, 
                    edgecolor=colors[uav_idx % len(colors)], 
                    facecolor='none',
                    label=f'UAV {uav_idx + 1} Rectangle'
                )
                ax.add_patch(rect)
                
                # 직사각형 중심에 UAV 번호 표시
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                ax.text(center_x, center_y, f'UAV {uav_idx + 1}', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 디버깅 정보 출력
                print(f"UAV {uav_idx + 1}: Rectangle bounds = ({min_x:.1f}, {max_x:.1f}, {min_y:.1f}, {max_y:.1f})")
        
        # 초기 위치 표시
        if len(self.initial_positions) > 0:
            initial_coords = np.array([[pos % self.nx, pos // self.nx] for pos in self.initial_positions])
            ax.scatter(initial_coords[:, 0], initial_coords[:, 1], 
                      c='black', s=100, marker='X', label='Initial Positions', zorder=10)
        
        ax.set_xlim(-1, self.nx)
        ax.set_ylim(-1, self.ny)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f'Rectangular Patches Division ({self.num_uavs} UAVs)')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        plt.tight_layout()
        plt.show()
        
        # 요약 정보 출력
        print(f"\n=== Rectangular Patches Summary ===")
        print(f"Number of UAVs: {self.num_uavs}")
        print(f"Grid size: {self.nx} x {self.ny}")
        print(f"Open space points: {len(self.open_space_coords)}")
        print(f"Obstacle points: {len(self.obs_pos)}")
        
        if len(self.open_space_coords) > 0:
            min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
            min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
            total_width = max_x - min_x + 1
            total_height = max_y - min_y + 1
            patch_width = total_width / self.num_uavs
            
            print(f"Open space bounding box: ({min_x}, {max_x}, {min_y}, {max_y})")
            print(f"Total dimensions: {total_width} x {total_height}")
            print(f"Each rectangle width: {patch_width:.2f}")
            print(f"Each rectangle height: {total_height}")
        print("=====================================\n")

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """직선과 선분의 교점 계산"""
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        
        # 행렬식이 0이면 평행선
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:  # 수치적 안정성을 위해 작은 값 사용
            return None
            
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            # u가 [0,1] 범위에 있어야 선분과 교차
            if 0 <= u <= 1:
                intersection = line_point + t * line_direction
                return intersection
        except np.linalg.LinAlgError:
            return None
        return None

    def calculate_grid_size(self, corners, nx, ny):
        """기존 grid size 계산 로직 (호환성 유지)"""
        return 50
# class MultiRobotPathPlanner_ADD7:
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
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        
        # 기존 DARP 대신 간소화된 처리
        self.DARP_success = True  # 항상 성공으로 설정
        self.iterations = 0
        
        print("Using simplified rectangular patch division instead of DARP.")
        
        # Open space 계산
        total_cells = set(range(nx * ny))
        obs_set = set(obs_pos)
        self.open_space_indices = list(total_cells - obs_set)
        self.open_space_coords = np.array([[idx % nx, idx // nx] for idx in self.open_space_indices])
        
        # 간소화된 영역 분할
        self.rectangular_patches = self.divide_into_rectangular_patches()
        
        # 기존 결과 구조 유지
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = dict()
        self.best_case["paths"] = []
        
        # 경로 생성 (기존 ConvexHull + intersection 로직 활용)
        self.generate_paths_for_patches()

    def divide_into_rectangular_patches(self):
        """Open space를 UAV 수와 동일한 개수의 직사각형으로 분할"""
        if len(self.open_space_coords) == 0:
            return [[] for _ in range(self.num_uavs)]
        
        # Open space의 bounding box 계산
        min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
        min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
        
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
        
        # UAV 수만큼 직사각형으로 분할 (긴 직선 경로를 위해 가로로 분할)
        patch_width = total_width / self.num_uavs
        patch_height = total_height
        
        # UAV 수만큼 동일한 크기의 직사각형 생성
        patches = []
        
        for uav_idx in range(self.num_uavs):
            patch_min_x = min_x + uav_idx * patch_width
            patch_max_x = min_x + (uav_idx + 1) * patch_width
            patch_min_y = min_y
            patch_max_y = max_y
            
            rectangle_info = {
                'bounds': (patch_min_x, patch_max_x, patch_min_y, patch_max_y),
                'corners': np.array([
                    [patch_min_x, patch_min_y],  # 좌하
                    [patch_max_x, patch_min_y],  # 우하
                    [patch_max_x, patch_max_y],  # 우상
                    [patch_min_x, patch_max_y]   # 좌상
                ]),
                'edges': [
                    (np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_min_y])),  # 하단
                    (np.array([patch_max_x, patch_min_y]), np.array([patch_max_x, patch_max_y])),  # 우측
                    (np.array([patch_max_x, patch_max_y]), np.array([patch_min_x, patch_max_y])),  # 상단
                    (np.array([patch_min_x, patch_max_y]), np.array([patch_min_x, patch_min_y]))   # 좌측
                ]
            }
            patches.append(rectangle_info)
        
        return patches

    def generate_paths_for_patches(self):
        """각 직사각형 패치에 대해 직접 intersection 계산으로 경로 생성"""
        l_bin = self.calculate_grid_size(None, self.nx, self.ny)
        
        for value in range(self.num_uavs):
            rectangle_info = self.rectangular_patches[value]
            
            if not rectangle_info or 'bounds' not in rectangle_info:
                self.circ_paths.append([])
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
                continue
            
            # 직사각형 정보 추출
            min_x, max_x, min_y, max_y = rectangle_info['bounds']
            rectangle_edges = rectangle_info['edges']
            
            # 그리드 생성 (기존 구조 유지용)
            patch_grid = np.full((self.ny, self.nx), -1)
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if 0 <= x < self.nx and 0 <= y < self.ny:
                        patch_grid[y, x] = value
            
            self.hull_grids.append(patch_grid)
            self.shull_grids.append(patch_grid.copy())
            
            # 직사각형에서 직접 intersection 경로 생성
            path = self.generate_rectangle_intersection_path(rectangle_info, l_bin)
            self.circ_paths.append(path)
        
        # 결과 포맷팅 (기존 구조 유지)
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def generate_rectangle_intersection_path(self, rectangle_info, l_bin):
        """직사각형 경계에서 직접 intersection을 계산하여 경로 생성"""
        min_x, max_x, min_y, max_y = rectangle_info['bounds']
        
        # 직사각형 중심점
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 긴 직선 경로를 위해 주축을 가로 방향으로 설정
        major_axis = np.array([1.0, 0.0])  # 가로 방향 (X축)
        
        # 직사각형의 세로 길이
        rectangle_height = max_y - min_y
        
        # 스캔 라인 간격 계산
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        # 스캔 라인 개수
        line_count = int(rectangle_height / s_beta) + 1
        if line_count < 1:
            line_count = 1
        
        # 실제 라인 간격 (직사각형 높이에 맞춤)
        if line_count > 1:
            actual_spacing = rectangle_height / (line_count - 1)
        else:
            actual_spacing = 0
        
        # 지그재그 패턴으로 경로 생성 (직사각형 경계 사용)
        path = []
        
        for i in range(line_count):
            line_y = min_y + i * actual_spacing
            
            # 직사각형의 왼쪽과 오른쪽 경계점
            left_point = np.array([min_x, line_y])
            right_point = np.array([max_x, line_y])
            
            # 지그재그 패턴
            if i % 2 == 0:  # 짝수 번째: 왼쪽 → 오른쪽
                path.extend([left_point, right_point])
            else:  # 홀수 번째: 오른쪽 → 왼쪽
                path.extend([right_point, left_point])
        
        return path

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """직선과 선분의 교점 계산"""
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        
        # 행렬식이 0이면 평행선
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:  # 수치적 안정성을 위해 작은 값 사용
            return None
            
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            # u가 [0,1] 범위에 있어야 선분과 교차
            if 0 <= u <= 1:
                intersection = line_point + t * line_direction
                return intersection
        except np.linalg.LinAlgError:
            return None
        return None

    def calculate_grid_size(self, corners, nx, ny):
        """기존 grid size 계산 로직 (호환성 유지)"""
        return 50
# class MultiRobotPathPlanner_ADD7:
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
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        
        # 기존 DARP 대신 간소화된 처리
        self.DARP_success = True  # 항상 성공으로 설정
        self.iterations = 0
        
        print("Using simplified rectangular patch division instead of DARP.")
        
        # Open space 계산
        total_cells = set(range(nx * ny))
        obs_set = set(obs_pos)
        self.open_space_indices = list(total_cells - obs_set)
        self.open_space_coords = np.array([[idx % nx, idx // nx] for idx in self.open_space_indices])
        
        # 간소화된 영역 분할
        self.rectangular_patches = self.divide_into_rectangular_patches()
        
        # 기존 결과 구조 유지
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = dict()
        self.best_case["paths"] = []
        
        # 경로 생성 (기존 ConvexHull + intersection 로직 활용)
        self.generate_paths_for_patches()

    def divide_into_rectangular_patches(self):
        """Open space를 UAV 수와 동일한 개수의 직사각형으로 분할"""
        if len(self.open_space_coords) == 0:
            return [[] for _ in range(self.num_uavs)]
        
        # Open space의 bounding box 계산
        min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
        min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
        
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
        
        # UAV 수만큼 직사각형으로 분할 (긴 직선 경로를 위해 가로로 분할)
        patch_width = total_width / self.num_uavs
        patch_height = total_height
        
        # UAV 수만큼 동일한 크기의 직사각형 생성
        patches = []
        
        for uav_idx in range(self.num_uavs):
            patch_min_x = min_x + uav_idx * patch_width
            patch_max_x = min_x + (uav_idx + 1) * patch_width
            patch_min_y = min_y
            patch_max_y = max_y
            
            rectangle_info = {
                'bounds': (patch_min_x, patch_max_x, patch_min_y, patch_max_y),
                'corners': np.array([
                    [patch_min_x, patch_min_y],  # 좌하
                    [patch_max_x, patch_min_y],  # 우하
                    [patch_max_x, patch_max_y],  # 우상
                    [patch_min_x, patch_max_y]   # 좌상
                ]),
                'edges': [
                    (np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_min_y])),  # 하단
                    (np.array([patch_max_x, patch_min_y]), np.array([patch_max_x, patch_max_y])),  # 우측
                    (np.array([patch_max_x, patch_max_y]), np.array([patch_min_x, patch_max_y])),  # 상단
                    (np.array([patch_min_x, patch_max_y]), np.array([patch_min_x, patch_min_y]))   # 좌측
                ]
            }
            patches.append(rectangle_info)
        
        return patches

    def generate_paths_for_patches(self):
        """각 직사각형 패치에 대해 직접 intersection 계산으로 경로 생성"""
        l_bin = self.calculate_grid_size(None, self.nx, self.ny)
        
        for value in range(self.num_uavs):
            rectangle_info = self.rectangular_patches[value]
            
            if not rectangle_info or 'bounds' not in rectangle_info:
                self.circ_paths.append([])
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
                continue
            
            # 직사각형 정보 추출
            min_x, max_x, min_y, max_y = rectangle_info['bounds']
            rectangle_edges = rectangle_info['edges']
            
            # 그리드 생성 (기존 구조 유지용)
            patch_grid = np.full((self.ny, self.nx), -1)
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if 0 <= x < self.nx and 0 <= y < self.ny:
                        patch_grid[y, x] = value
            
            self.hull_grids.append(patch_grid)
            self.shull_grids.append(patch_grid.copy())
            
            # 직사각형에서 직접 intersection 경로 생성
            path = self.generate_rectangle_intersection_path(rectangle_info, l_bin)
            self.circ_paths.append(path)
        
        # 결과 포맷팅 (기존 구조 유지)
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def generate_rectangle_intersection_path(self, rectangle_info, l_bin):
        """직사각형과 스캔 라인들의 intersection을 계산하여 경로 생성"""
        min_x, max_x, min_y, max_y = rectangle_info['bounds']
        rectangle_edges = rectangle_info['edges']
        
        # 직사각형 중심점
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center = np.array([center_x, center_y])
        
        # 긴 직선 경로를 위해 주축을 가로 방향으로 설정
        major_axis = np.array([1.0, 0.0])  # 가로 방향 (X축)
        minor_axis = np.array([0.0, 1.0])  # 세로 방향 (Y축)
        
        # 직사각형의 세로 길이 (minor axis 방향)
        rectangle_height = max_y - min_y
        
        # 스캔 라인 간격 계산
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        # 스캔 라인 개수 및 간격
        line_count = int(rectangle_height / s_beta) + 1
        if line_count < 1:
            line_count = 1
        
        # 실제 라인 간격 (직사각형 높이에 맞춤)
        if line_count > 1:
            actual_spacing = rectangle_height / (line_count - 1)
        else:
            actual_spacing = 0
        
        # 스캔 라인들 생성 (가로 방향 라인들)
        lines = []
        for i in range(line_count):
            line_y = min_y + i * actual_spacing
            line_point = np.array([center_x, line_y])
            lines.append((line_point, major_axis))
        
        # 각 스캔 라인과 직사각형 경계의 교점 계산
        intersections = []
        for line_point, direction in lines:
            line_intersections = []
            
            # 직사각형의 4개 모서리와 교점 계산
            for edge_start, edge_end in rectangle_edges:
                intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                if intersection is not None:
                    # 교점이 실제 직사각형 내부에 있는지 확인
                    if (min_x <= intersection[0] <= max_x and 
                        min_y <= intersection[1] <= max_y):
                        line_intersections.append(intersection)
            
            # 중복 교점 제거 및 정렬
            if len(line_intersections) >= 2:
                # X 좌표 기준으로 정렬
                line_intersections.sort(key=lambda p: p[0])
                # 가장 왼쪽과 오른쪽 점만 사용
                intersections.append((line_intersections[0], line_intersections[-1]))
        
        # 지그재그 패턴으로 경로 생성
        path = []
        reverse = False
        for pair in intersections:
            if reverse:
                path.extend([pair[1], pair[0]])  # 오른쪽에서 왼쪽으로
            else:
                path.extend([pair[0], pair[1]])  # 왼쪽에서 오른쪽으로
            reverse = not reverse
        
        return path

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """직선과 선분의 교점 계산"""
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        
        # 행렬식이 0이면 평행선
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:  # 수치적 안정성을 위해 작은 값 사용
            return None
            
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            # u가 [0,1] 범위에 있어야 선분과 교차
            if 0 <= u <= 1:
                intersection = line_point + t * line_direction
                return intersection
        except np.linalg.LinAlgError:
            return None
        return None

    def calculate_grid_size(self, corners, nx, ny):
        """기존 grid size 계산 로직 (호환성 유지)"""
        return 50
# class MultiRobotPathPlanner_ADD7:
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
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        
        # 기존 DARP 대신 간소화된 처리
        self.DARP_success = True  # 항상 성공으로 설정
        self.iterations = 0
        
        print("Using simplified rectangular patch division instead of DARP.")
        
        # Open space 계산
        total_cells = set(range(nx * ny))
        obs_set = set(obs_pos)
        self.open_space_indices = list(total_cells - obs_set)
        self.open_space_coords = np.array([[idx % nx, idx // nx] for idx in self.open_space_indices])
        
        # 간소화된 영역 분할
        self.rectangular_patches = self.divide_into_rectangular_patches()
        
        # 기존 결과 구조 유지
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = dict()
        self.best_case["paths"] = []
        
        # 경로 생성 (기존 ConvexHull + intersection 로직 활용)
        self.generate_paths_for_patches()

    def divide_into_rectangular_patches(self):
        """Open space를 UAV 수만큼 직사각형 패치로 분할"""
        if len(self.open_space_coords) == 0:
            return [[] for _ in range(self.num_uavs)]
        
        # Open space의 bounding box 계산
        min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
        min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
        
        # 직사각형 패치 크기 결정 (긴 직선 경로를 위해 세로가 더 긴 직사각형)
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
        
        # UAV 수에 따른 격자 분할 (가로 우선)
        grid_cols = int(np.ceil(np.sqrt(self.num_uavs)))
        grid_rows = int(np.ceil(self.num_uavs / grid_cols))
        
        patch_width = total_width / grid_cols
        patch_height = total_height / grid_rows
        
        patches = []
        uav_idx = 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if uav_idx >= self.num_uavs:
                    break
                    
                # 각 패치의 직사각형 경계 정의
                patch_min_x = min_x + col * patch_width
                patch_max_x = min_x + (col + 1) * patch_width
                patch_min_y = min_y + row * patch_height
                patch_max_y = min_y + (row + 1) * patch_height
                
                # 직사각형 정보를 딕셔너리로 저장 (좌표 배열 대신)
                rectangle_info = {
                    'bounds': (patch_min_x, patch_max_x, patch_min_y, patch_max_y),
                    'corners': np.array([
                        [patch_min_x, patch_min_y],  # 좌하
                        [patch_max_x, patch_min_y],  # 우하
                        [patch_max_x, patch_max_y],  # 우상
                        [patch_min_x, patch_max_y]   # 좌상
                    ]),
                    'edges': [
                        (np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_min_y])),  # 하단
                        (np.array([patch_max_x, patch_min_y]), np.array([patch_max_x, patch_max_y])),  # 우측
                        (np.array([patch_max_x, patch_max_y]), np.array([patch_min_x, patch_max_y])),  # 상단
                        (np.array([patch_min_x, patch_max_y]), np.array([patch_min_x, patch_min_y]))   # 좌측
                    ]
                }
                
                patches.append(rectangle_info)
                uav_idx += 1
        
        return patches

    def generate_paths_for_patches(self):
        """각 직사각형 패치에 대해 직접 intersection 계산으로 경로 생성"""
        l_bin = self.calculate_grid_size(None, self.nx, self.ny)
        
        for value in range(self.num_uavs):
            rectangle_info = self.rectangular_patches[value]
            
            if not rectangle_info or 'bounds' not in rectangle_info:
                self.circ_paths.append([])
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
                continue
            
            # 직사각형 정보 추출
            min_x, max_x, min_y, max_y = rectangle_info['bounds']
            rectangle_edges = rectangle_info['edges']
            
            # 그리드 생성 (기존 구조 유지용)
            patch_grid = np.full((self.ny, self.nx), -1)
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if 0 <= x < self.nx and 0 <= y < self.ny:
                        patch_grid[y, x] = value
            
            self.hull_grids.append(patch_grid)
            self.shull_grids.append(patch_grid.copy())
            
            # 직사각형에서 직접 intersection 경로 생성
            path = self.generate_rectangle_intersection_path(rectangle_info, l_bin)
            self.circ_paths.append(path)
        
        # 결과 포맷팅 (기존 구조 유지)
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def generate_rectangle_intersection_path(self, rectangle_info, l_bin):
        """직사각형과 스캔 라인들의 intersection을 계산하여 경로 생성"""
        min_x, max_x, min_y, max_y = rectangle_info['bounds']
        rectangle_edges = rectangle_info['edges']
        
        # 직사각형 중심점
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center = np.array([center_x, center_y])
        
        # 긴 직선 경로를 위해 주축을 가로 방향으로 설정
        major_axis = np.array([1.0, 0.0])  # 가로 방향 (X축)
        minor_axis = np.array([0.0, 1.0])  # 세로 방향 (Y축)
        
        # 직사각형의 세로 길이 (minor axis 방향)
        rectangle_height = max_y - min_y
        
        # 스캔 라인 간격 계산
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        # 스캔 라인 개수 및 간격
        line_count = int(rectangle_height / s_beta) + 1
        if line_count < 1:
            line_count = 1
        
        # 실제 라인 간격 (직사각형 높이에 맞춤)
        if line_count > 1:
            actual_spacing = rectangle_height / (line_count - 1)
        else:
            actual_spacing = 0
        
        # 스캔 라인들 생성 (가로 방향 라인들)
        lines = []
        for i in range(line_count):
            line_y = min_y + i * actual_spacing
            line_point = np.array([center_x, line_y])
            lines.append((line_point, major_axis))
        
        # 각 스캔 라인과 직사각형 경계의 교점 계산
        intersections = []
        for line_point, direction in lines:
            line_intersections = []
            
            # 직사각형의 4개 모서리와 교점 계산
            for edge_start, edge_end in rectangle_edges:
                intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                if intersection is not None:
                    # 교점이 실제 직사각형 내부에 있는지 확인
                    if (min_x <= intersection[0] <= max_x and 
                        min_y <= intersection[1] <= max_y):
                        line_intersections.append(intersection)
            
            # 중복 교점 제거 및 정렬
            if len(line_intersections) >= 2:
                # X 좌표 기준으로 정렬
                line_intersections.sort(key=lambda p: p[0])
                # 가장 왼쪽과 오른쪽 점만 사용
                intersections.append((line_intersections[0], line_intersections[-1]))
        
        # 지그재그 패턴으로 경로 생성
        path = []
        reverse = False
        for pair in intersections:
            if reverse:
                path.extend([pair[1], pair[0]])  # 오른쪽에서 왼쪽으로
            else:
                path.extend([pair[0], pair[1]])  # 왼쪽에서 오른쪽으로
            reverse = not reverse
        
        return path

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """직선과 선분의 교점 계산"""
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        
        # 행렬식이 0이면 평행선
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:  # 수치적 안정성을 위해 작은 값 사용
            return None
            
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            # u가 [0,1] 범위에 있어야 선분과 교차
            if 0 <= u <= 1:
                intersection = line_point + t * line_direction
                return intersection
        except np.linalg.LinAlgError:
            return None
        return None

    def calculate_grid_size(self, corners, nx, ny):
        """기존 grid size 계산 로직 (호환성 유지)"""
        return 50
# class MultiRobotPathPlanner_ADD7:
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
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        
        # 기존 DARP 대신 간소화된 처리
        self.DARP_success = True  # 항상 성공으로 설정
        self.iterations = 0
        
        print("Using simplified rectangular patch division instead of DARP.")
        
        # Open space 계산
        total_cells = set(range(nx * ny))
        obs_set = set(obs_pos)
        self.open_space_indices = list(total_cells - obs_set)
        self.open_space_coords = np.array([[idx % nx, idx // nx] for idx in self.open_space_indices])
        
        # 간소화된 영역 분할
        self.rectangular_patches = self.divide_into_rectangular_patches()
        
        # 기존 결과 구조 유지
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = dict()
        self.best_case["paths"] = []
        
        # 경로 생성 (기존 ConvexHull + intersection 로직 활용)
        self.generate_paths_for_patches()

    def divide_into_rectangular_patches(self):
        """Open space를 UAV 수만큼 직사각형 패치로 분할"""
        if len(self.open_space_coords) == 0:
            return [[] for _ in range(self.num_uavs)]
        
        # Open space의 bounding box 계산
        min_x, max_x = np.min(self.open_space_coords[:, 0]), np.max(self.open_space_coords[:, 0])
        min_y, max_y = np.min(self.open_space_coords[:, 1]), np.max(self.open_space_coords[:, 1])
        
        # 직사각형 패치 크기 결정 (긴 직선 경로를 위해 세로가 더 긴 직사각형)
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
        
        # UAV 수에 따른 격자 분할 (가로 우선)
        grid_cols = int(np.ceil(np.sqrt(self.num_uavs)))
        grid_rows = int(np.ceil(self.num_uavs / grid_cols))
        
        patch_width = total_width / grid_cols
        patch_height = total_height / grid_rows
        
        patches = []
        uav_idx = 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if uav_idx >= self.num_uavs:
                    break
                    
                # 각 패치의 경계 계산
                patch_min_x = min_x + col * patch_width
                patch_max_x = min_x + (col + 1) * patch_width
                patch_min_y = min_y + row * patch_height
                patch_max_y = min_y + (row + 1) * patch_height
                
                # 패치 내 좌표들 생성 (경계를 약간 넘어가도 허용)
                patch_coords = []
                x_range = np.arange(int(patch_min_x), int(patch_max_x) + 1)
                y_range = np.arange(int(patch_min_y), int(patch_max_y) + 1)
                
                for x in x_range:
                    for y in y_range:
                        if 0 <= x < self.nx and 0 <= y < self.ny:
                            patch_coords.append([x, y])
                
                patches.append(np.array(patch_coords) if patch_coords else np.array([]))
                uav_idx += 1
        
        return patches

    def generate_paths_for_patches(self):
        """각 패치에 대해 기존 ConvexHull + intersection 로직을 사용하여 경로 생성"""
        l_bin = self.calculate_grid_size(None, self.nx, self.ny)
        
        for value in range(self.num_uavs):
            coords = self.rectangular_patches[value]
            
            if len(coords) < 3:
                self.circ_paths.append([])
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
                continue
            
            # 기존 ConvexHull 로직 적용
            try:
                hull = ConvexHull(coords)
                hull_coords = coords[hull.vertices]
                
                # Hull grid 생성
                hull_grid = np.full((self.ny, self.nx), -1)
                rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], (self.ny, self.nx))
                valid_mask = (0 <= rr) & (rr < self.ny) & (0 <= cc) & (cc < self.nx)
                hull_grid[rr[valid_mask], cc[valid_mask]] = value
                self.hull_grids.append(hull_grid)
                
                shull_grid = np.full((self.ny, self.nx), -1)
                shull_grid[rr[valid_mask], cc[valid_mask]] = value
                self.shull_grids.append(shull_grid)
                
                # Hull edges 계산
                hull_edges = []
                for simplex in hull.simplices:
                    hull_edges.append((coords[simplex[0]], coords[simplex[1]]))
                
                # 기존 intersection 로직 적용
                path = self.generate_intersection_path(coords, hull_coords, hull_edges, l_bin)
                self.circ_paths.append(path)
                
            except Exception as e:
                print(f"Warning: ConvexHull failed for UAV {value}: {e}")
                # ConvexHull 실패시 단순 래스터 스캔으로 대체
                path = self.generate_simple_raster_path(coords)
                self.circ_paths.append(path)
                self.hull_grids.append(np.full((self.ny, self.nx), -1))
                self.shull_grids.append(np.full((self.ny, self.nx), -1))
        
        # 결과 포맷팅 (기존 구조 유지)
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def generate_intersection_path(self, coords, hull_coords, hull_edges, l_bin):
        """기존 intersection 로직을 사용한 경로 생성"""
        mean = np.mean(coords, axis=0)
        
        # 긴 직선 경로를 위해 주축을 가로 방향으로 설정
        major_axis = np.array([1.0, 0.0])  # 가로 방향
        minor_axis = np.array([0.0, 1.0])  # 세로 방향
        
        hull_points = coords[ConvexHull(coords).vertices] if len(coords) >= 3 else coords
        projected_points = hull_points @ minor_axis
        max_grid = np.max(projected_points) - np.min(projected_points) if len(projected_points) > 1 else 1
        
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        perpendicular_axis = np.array([-major_axis[1], major_axis[0]])
        line_count = int(max_grid / s_beta + 1) if s_beta > 0 else 1
        line_spacing = s_beta
        lines = []
        
        for i in range(-line_count // 2, line_count // 2 + 1):
            offset = perpendicular_axis * (i * line_spacing)
            point_on_line = mean + offset
            lines.append((point_on_line, major_axis))
        
        # Intersection 계산
        intersections = []
        for line_point, direction in lines:
            direction = direction / np.linalg.norm(direction)
            line_intersections = []
            for edge_start, edge_end in hull_edges:
                intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                if intersection is not None:
                    line_intersections.append(intersection)
            
            if len(line_intersections) >= 2:
                line_intersections.sort(key=lambda p: np.dot(p - line_point, direction))
                intersections.append((line_intersections[0], line_intersections[-1]))
        
        # 경로 생성 (지그재그 패턴)
        path = []
        reverse = False
        for pair in intersections:
            if reverse:
                path.extend([pair[1], pair[0]])
            else:
                path.extend(pair)
            reverse = not reverse
        
        return path

    def generate_simple_raster_path(self, coords):
        """ConvexHull 실패시 대체용 단순 래스터 경로"""
        if len(coords) == 0:
            return []
        
        # 좌표들을 y값 기준으로 그룹핑
        coords_sorted = coords[np.lexsort((coords[:, 0], coords[:, 1]))]
        
        path = []
        current_y = None
        row_points = []
        reverse = False
        
        for coord in coords_sorted:
            if current_y is None or coord[1] != current_y:
                # 새로운 행 시작
                if row_points:
                    if reverse:
                        row_points.reverse()
                    path.extend(row_points)
                    reverse = not reverse
                
                row_points = [coord]
                current_y = coord[1]
            else:
                row_points.append(coord)
        
        # 마지막 행 처리
        if row_points:
            if reverse:
                row_points.reverse()
            path.extend(row_points)
        
        return path

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """기존 line-edge intersection 로직"""
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
        """기존 grid size 계산 로직"""
        return 50
# class MultiRobotPathPlanner_ADD7:
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

        self.turn_radius = turn_radius
        self.DARP_success, self.iterations = self.darp_instance.divide_regions()

        if not self.DARP_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            print("DARP Success.")
            l_bin = self.calculate_grid_size(map_roi, nx, ny)

            s_alpha = turn_radius / l_bin
            s_alpha = 0

            resize_shape = (nx, ny)
            resized_grid = resize(
                self.darp_instance.assg_mat, resize_shape, order=0, preserve_range=True, anti_aliasing=False
            ).astype(int)
            
            self.hull_grids = []
            self.shull_grids = []
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
                # cov_matrix = np.cov(centroid_points.T)
                # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                # major_axis = eigenvectors[:, np.argmax(eigenvalues)]
                # minor_axis = eigenvectors[:, np.argmin(eigenvalues)]
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
                self.compare_intersections(intersections, intersections_improved)


                # path = []
                # reverse = False
                # for pair in intersections:
                #     if reverse:
                #         path.extend([pair[1], pair[0]])
                #     else:
                #         path.extend(pair)
                #     reverse = not reverse
                # 기존 코드를 다음과 같이 교체:

                
                path = []
                if len(intersections) > 0:
                    mid_point = len(intersections) // 2
                    max_lines = max(mid_point, len(intersections) - mid_point)
                    
                    for i in range(max_lines):
                        # 1파트 라인
                        if i < mid_point:
                            pair1 = intersections[i]
                            path.extend([pair1[0], pair1[1]])
                        
                        # 2파트 라인
                        part2_idx = mid_point + i
                        if part2_idx < len(intersections):
                            pair2 = intersections[part2_idx]
                            
                            # 남는 라인들은 정방향으로 처리
                            if i >= mid_point:  # 1파트가 끝난 후 남은 라인들
                                path.extend([pair2[0], pair2[1]])  # 정방향
                            else:
                                path.extend([pair2[1], pair2[0]])  # 역방향

                # 데이터 저장 (temp)
                all_coords_list.append(coords)
                all_hull_coords_list.append(hull_coords)
                all_paths_list.append(path)
                self.plot_all_robots_paths_separately(all_coords_list, all_hull_coords_list, all_paths_list)

                
                self.circ_paths.append(path)
            
            self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

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


    def filter_hull_vertex_intersections(self, line_intersections, hull_coords, min_distance_threshold=1.5):
        """
        Hull 꼭짓점과 너무 가까운 교점들을 필터링합니다.
        
        Args:
            line_intersections: 라인과 hull edge들의 교점 리스트
            hull_coords: Hull 꼭짓점들의 좌표
            min_distance_threshold: 꼭짓점과의 최소 거리 (격자 단위)
        
        Returns:
            filtered_intersections: 필터링된 교점 리스트
        """
        if len(line_intersections) <= 2:
            return line_intersections
        
        filtered_intersections = []
        
        for intersection in line_intersections:
            # 각 hull 꼭짓점과의 거리 계산
            distances_to_vertices = []
            for vertex in hull_coords:
                distance = np.linalg.norm(intersection - vertex)
                distances_to_vertices.append(distance)
            
            # 가장 가까운 꼭짓점과의 거리가 threshold보다 크면 유지
            min_distance_to_vertex = min(distances_to_vertices)
            if min_distance_to_vertex > min_distance_threshold:
                filtered_intersections.append(intersection)
        
        return filtered_intersections
    
    def remove_duplicate_intersections(self,intersections, min_distance_between_points=0.5):
        """
        너무 가까운 교점들을 제거하여 중복을 방지합니다.
        
        Args:
            intersections: 교점 리스트
            min_distance_between_points: 교점들 사이의 최소 거리
        
        Returns:
            unique_intersections: 중복이 제거된 교점 리스트
        """
        if len(intersections) <= 1:
            return intersections
        
        unique_intersections = [intersections[0]]
        
        for intersection in intersections[1:]:
            # 이미 추가된 교점들과의 거리 확인
            is_duplicate = False
            for existing_point in unique_intersections:
                distance = np.linalg.norm(intersection - existing_point)
                if distance < min_distance_between_points:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_intersections.append(intersection)
        
        return unique_intersections
    
    def ensure_exactly_two_intersections(self, intersections, line_point, direction):
        """
        각 라인에서 정확히 2개의 교점만 유지하도록 보장합니다.
        
        Args:
            intersections: 필터링된 교점 리스트
            line_point: 라인 위의 한 점
            direction: 라인 방향 벡터
        
        Returns:
            final_intersections: 정확히 2개의 교점 (또는 빈 리스트)
        """
        if len(intersections) < 2:
            return []  # 유효한 교점이 2개 미만이면 이 라인은 제외
        elif len(intersections) == 2:
            return intersections
        else:
            # 3개 이상인 경우, 라인 방향으로 가장 먼 두 점 선택
            # 라인 위에서의 projection 값으로 정렬
            projections = []
            for point in intersections:
                projection = np.dot(point - line_point, direction)
                projections.append((projection, point))
            
            projections.sort(key=lambda x: x[0])
            
            # 가장 먼 두 점 선택 (첫 번째와 마지막)
            return [projections[0][1], projections[-1][1]]
    
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
        return 50

# ==========================================================================================
# 스크립트 실행 부분
# ==========================================================================================
if __name__ == "__main__":
    mock_manager = MockManager()
    execution_time, results = select(mock_manager)

    print(f"\nExecution time: {execution_time:.4f} seconds")
    print("\nGenerated Results:")
    for avs_id, result in results.items():
        print(f"  - AVS ID: {avs_id}, Waypoint Count: {result.waypoint_count}")