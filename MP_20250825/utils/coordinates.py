import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pymap3d.enu import enu2geodetic, geodetic2enu

from commu.sys.receiver.data.fs2mps import TargetFusionResult
from commu.sys.receiver.data.gcs2mps import AvsInfo, Boundary
from config.mp_config import ALT_REF

LatLonType = Tuple[float, float]
LatLonAltType = Tuple[float, float, float]


def convert_waypoints_enu_to_lla(
    waypoint_array_as_enu: NDArray[np.float64], lat0: float, lon0: float, alt0: float
) -> NDArray[np.float64]:
    waypoint_array_as_lla = []
    if waypoint_array_as_enu.size > 3:
        for wp in waypoint_array_as_enu:
            east, north, up = wp
            waypoint_array_as_lla.append(np.array(enu2geodetic(east, north, up, lat0, lon0, alt0)))
    else:
        east, north, up = waypoint_array_as_enu
        waypoint_array_as_lla.append(np.array(enu2geodetic(east, north, up, lat0, lon0, alt0)))

    return np.array(waypoint_array_as_lla)


def split_origin(boundary: Boundary) -> LatLonAltType:
    lat0, lon0, alt0 = boundary[0][0], boundary[0][1], ALT_REF

    return lat0, lon0, alt0


def sort_coordinates_as_clockwise(coords: List[LatLonType]) -> List[LatLonType]:
    """
    임의 개수의 (lat, lon) 좌표를 시계방향으로 정렬하고,
    시작점을 '좌상단(최대 위도, 위도 동일 시 최소 경도)'으로 회전해 반환합니다.

    - 위도(lat) ↑ 이 북쪽, 경도(lon) ↓ 이 서쪽
    - 반시계가 아닌 '시계방향' 순서로 반환
    - 동일점/중복점이 있으면 결과에 그대로 남습니다.

    Parameters
    ----------
    coords : List[(lat, lon)]
        3개 이상 권장. (2개 이하는 입력 그대로 반환)

    Returns
    -------
    List[(lat, lon)]
        좌상단부터 시작하는 시계방향 정렬 결과
    """
    n = len(coords)
    if n <= 2:
        return coords[:]  # 그대로

    # 무게중심(centroid) 계산 (x=lon, y=lat)
    mean_lat = sum(lat for lat, _ in coords) / n
    mean_lon = sum(lon for _, lon in coords) / n

    # 각도 계산 후 시계방향 정렬
    # atan2(y, x): 여기서 y=lat - mean_lat, x=lon - mean_lon
    # 일반적으로 각도 오름차순은 반시계(CCW). 시계(CW)로 하려면 내림차순.
    def angle(p: LatLonType) -> float:
        lat, lon = p
        return math.atan2(lat - mean_lat, lon - mean_lon)

    # 시계방향
    cw_sorted = sorted(coords, key=angle, reverse=True)

    # 시작점을 '좌상단'으로 회전
    # 좌상단: (최대 위도, 동률 시 최소 경도)
    def topleft_key(p: LatLonType):
        lat, lon = p
        return (-lat, lon)  # 위도 큰 게 우선이므로 -lat, 경도는 작을수록(서쪽) 우선

    start_idx = min(range(len(cw_sorted)), key=lambda i: topleft_key(cw_sorted[i]))

    # rotate
    return cw_sorted[start_idx:] + cw_sorted[:start_idx]


def convert_boundary_lla_to_enu(
    lat0: float, lon0: float, alt0: float, boundary_as_lla: NDArray[np.float64]
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    top_left, top_right, bottom_right, bottom_left = [
        np.array(geodetic2enu(b[0], b[1], 0, lat0, lon0, alt0)[:2]) for b in boundary_as_lla
    ]

    return top_left, top_right, bottom_right, bottom_left


def convert_avs_pos_lla_to_enu(infos: Dict[int, AvsInfo], lat0: float, lon0: float, alt0: float) -> Dict[int, AvsInfo]:
    for avs_id, info in infos.items():
        lat, lon, alt = info.position
        east, north, up = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        infos[avs_id].position = np.array([east, north, up])

    return infos


def convert_trg_pos_lla_to_enu(
    fus_out_dict: Dict[int, TargetFusionResult], lat0: float, lon0: float, alt0: float
) -> Dict[int, TargetFusionResult]:
    # 전역 표적 위치 변환
    for glb_trg_id, trg_fus_out in fus_out_dict.items():
        lat, lon, alt = trg_fus_out.position
        east, north, up = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        fus_out_dict[glb_trg_id].position = np.array([east, north, up])

        # 로컬 표적 위치 변환
        for loc_trg_id, loc_info in enumerate(trg_fus_out.local_info_list):
            loc_lat, loc_lon, loc_alt = loc_info.local_position
            loc_east, loc_north, loc_up = geodetic2enu(loc_lat, loc_lon, loc_alt, lat0, lon0, alt0)
            fus_out_dict[glb_trg_id].local_info_list[loc_trg_id].local_position = np.array(
                [loc_east, loc_north, loc_up]
            )

    return fus_out_dict


def true_round(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
