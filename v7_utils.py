import random
import math
import numpy as np
from skimage.draw import polygon, disk


# ==========================================================================================
# Mock 클래스들 (기존 의존성 해결)
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


# ==========================================================================================
# 상수 정의
# ==========================================================================================
ALT_REF = 500
FOV_LEN = 50
SURV_TURN_RADIUS = 350


# ==========================================================================================
# 유틸리티 함수들
# ==========================================================================================
def get_avail_avs(avs_to_avail_task, criteria, avs_info_dict):
    """사용 가능한 UAV 리스트와 위치 반환"""
    avs_list = list(avs_to_avail_task.keys())
    avs_pos_list = [avs_info_dict.get(i)["pos"][:2] for i in avs_list]
    return avs_list, avs_pos_list


def create_mesh(manager, top_left, top_right, bottom_right, bottom_left):
    """메시 생성 (호환성 유지용)"""
    xs = np.linspace(top_left[1], top_right[1], 100)
    ys = np.linspace(bottom_right[0], top_right[0], 100)
    return np.meshgrid(xs, ys)


def get_coordinate_array(xs, ys, alt):
    """좌표 배열 생성 (호환성 유지용)"""
    return np.vstack([xs, ys, np.full_like(xs, alt)]).T


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


def generate_mission_area_and_obstacles(grid_size, num_uavs):
    """
    임무 영역과 장애물 영역을 생성하고, UAV 시작 위치를 배치합니다.
    
    Args:
        grid_size: 그리드 크기
        num_uavs: UAV 개수
    
    Returns:
        tuple: (obs_pos, safe_in_pos) - 장애물 위치, 안전한 시작 위치
    """
    # 1. 임무 영역 생성
    mission_area_indices = generate_random_obstacle_shape(grid_size=grid_size)
    
    # 2. 전체 그리드에서 임무 영역을 제외한 나머지를 장애물로 정의
    total_cells = set(range(grid_size * grid_size))
    mission_area_set = set(mission_area_indices)
    obs_pos = list(total_cells - mission_area_set)
    
    # 3. UAV 시작 위치를 임무 영역 내에 배치
    if len(mission_area_indices) < num_uavs:
        raise ValueError(f"임무 영역이 너무 작아 {num_uavs}대의 UAV를 배치할 수 없습니다.")
    
    safe_in_pos = random.sample(mission_area_indices, num_uavs)
    
    return obs_pos, safe_in_pos


def convert_coordinates_to_grid_indices(avs_pos_list, boundary_vertices, grid_size):
    """
    위도/경도 좌표를 그리드 인덱스로 변환합니다.
    
    Args:
        avs_pos_list: UAV 위치 리스트 [(lat, lon), ...]
        boundary_vertices: 경계 좌표 [(lat, lon), ...]
        grid_size: 그리드 크기
    
    Returns:
        list: 그리드 인덱스 리스트
    """
    top_left, top_right, bottom_right, bottom_left = boundary_vertices
    
    min_lon, max_lon = top_left[1], top_right[1]
    min_lat, max_lat = bottom_left[0], top_left[0]
    
    in_pos = []
    for lat, lon in avs_pos_list:
        x_idx = int(((lon - min_lon) / (max_lon - min_lon)) * (grid_size - 1))
        y_idx = int(((lat - min_lat) / (max_lat - min_lat)) * (grid_size - 1))
        x_idx = max(0, min(grid_size - 1, x_idx))
        y_idx = max(0, min(grid_size - 1, y_idx))
        idx = y_idx * grid_size + x_idx
        in_pos.append(idx)
    
    return in_pos