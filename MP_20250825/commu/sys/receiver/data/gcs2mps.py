from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplanRequest:
    """비주기"""

    # 비행체 ID (0: 전체, >= 1: 개별 비행체)
    avs_id: int = -1
    # 재계획 요청 신호 (0: None, 1: False, 2: True)
    need_request: int = -1


@dataclass
class MissionPlanCommand:
    """비주기"""

    # 사용자 명령 (0: None, 1: 정찰 요청)
    reconn_request: int = -1
    # 표적 전역 ID
    target_id: int = -1


@dataclass
class Boundary:
    """비주기"""

    vertices: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))


@dataclass
class AvsInfo:
    """주기"""

    avs_id: int = -1
    system_group_id: int = -1
    smp_group_id: int = -1
    is_avs_arrival: int = -1
    is_system_group_arrival: int = -1
    system_mode: int = -1
    flight_control_mode: int = -1
    implement_mode: int = -1
    position: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    attitude: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    sensing_range: int = -1
    fov_vert: float = -1.0
    fov_horz: float = -1.0
    soc: float = -1.0
    voltage: float = -1.0
    hop_count: int = -1


@dataclass
class MissionPlanTarget:
    """비주기"""

    target_id: int = -1
    waypoint_offset_lat: float = -1.0
    waypoint_offset_lon: float = -1.0


@dataclass
class CircularMissionArea:
    """비주기"""

    center_lat: float = -1.0
    center_lon: float = -1.0
    radius: int = -1


@dataclass
class PolygonMissionArea:
    """비주기"""

    vertices: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))


@dataclass
class MissionInitialInfo:
    """비주기"""

    # 감시임무 고도
    surv_alt: int = -1
    # 정찰임무 고도
    reconn_alt: int = -1
    # 최저 전압 기준
    min_voltage: float = -1.0
    # 감시임무 속도
    speed: float = -1.0
    # 임무영역 도착 기준 (0: None, 1: Individual, 2: Group)
    arrival_param: int = -1
    # 감시경로 생성 방법 (0: None, 1: 사전 생성, 2: 실시간 생성)
    surv_path_gen_method: int = -1
    # 임무계획파일 저장 경로
    mp_file_path: str = ""
    # 정찰 선회반경
    loiter_radius: int = -1
    # 정찰 선회방향
    loiter_direction: int = -1
    # 표적융합결과 정답 모의 여부 (1: On, 2: Off)
    is_simulation: int = -1
    # 감시경로 선회속성
    surv_turn_type: int = -1


@dataclass
class MissionState:
    """비주기"""

    # 임무 상태 (0: None, 1: 시작, 2: 종료)
    state: int = -1
