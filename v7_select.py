import time
import numpy as np
from typing import Dict

from v7_utils import (
    MockManager, SmpMode, SurveillanceSelectionResult,
    FOV_LEN, SURV_TURN_RADIUS,
    get_avail_avs, generate_mission_area_and_obstacles,
    convert_coordinates_to_grid_indices
)
from v7_mrpp import MultiRobotPathPlanner_ADD7


def select(manager: MockManager) -> tuple[float, Dict[int, SurveillanceSelectionResult]]:
    """
    다중 로봇 경로 계획을 위한 메인 선택 함수
    
    Args:
        manager: MockManager 인스턴스
    
    Returns:
        tuple: (실행 시간, 결과 딕셔너리)
    """
    start_time = time.time()
    
    # 경계 좌표 추출
    boundary_vertices = manager.mp_input.boundary.vertices
    top_left, top_right, bottom_right, bottom_left = boundary_vertices
    
    # 사용 가능한 UAV 정보 가져오기
    avs_list, avs_pos_list = get_avail_avs(
        manager.avs_to_available_task_dict, 
        ["S"], 
        manager.mp_input_lla.avs_info_dict
    )
    
    # 결과 딕셔너리 초기화
    mp_result_dict: Dict[int, SurveillanceSelectionResult] = {}
    
    # 그리드 설정
    grid_size = 50 
    grid = [grid_size, grid_size]
    
    # 임무 영역과 장애물 생성
    obs_pos, safe_in_pos = generate_mission_area_and_obstacles(grid_size, len(avs_list))
    
    # UAV별 영역 비율 설정 (균등 분할)
    portions = ((1 / len(avs_list)) * np.ones(len(avs_list))).tolist()
    nep = True  # notEqualPortions
    vis = False  # visualization
    
    # 맵 ROI 설정
    map_roi = {
        "left_top": top_left, 
        "right_top": top_right,
        "right_bottom": bottom_right, 
        "left_bottom": bottom_left
    }
    
    # 다중 로봇 경로 계획 실행
    mrpp = MultiRobotPathPlanner_ADD7(
        nx=grid[0], 
        ny=grid[1], 
        notEqualPortions=nep,
        initial_positions=safe_in_pos, 
        portions=portions, 
        obs_pos=obs_pos,
        visualization=vis, 
        map_roi=map_roi,
        turn_radius=SURV_TURN_RADIUS, 
        fov_len=FOV_LEN,
    )
    
    # 결과 시각화
    mrpp.plot_visualization(grid_size, obs_pos, safe_in_pos, avs_pos_list)
    
    # 결과 생성 (기존 포맷 유지)
    for i in range(len(avs_list)):
        # 임시 waypoint 배열 생성 (실제 구현에서는 경로를 사용)
        waypoint_array = np.random.rand(10, 3)
        
        # UAV 정보 가져오기
        avs_info = manager.mp_input.avs_info_dict.get(avs_list[i])
        
        # 결과 객체 생성
        mp_result_dict[avs_info.avs_id] = SurveillanceSelectionResult(
            avs_id=avs_info.avs_id,
            system_group_id=avs_info.system_group_id,
            smp_mode=SmpMode.SURV,
            speed=manager.mp_input.mission_init_info.speed,
            waypoint_count=waypoint_array.shape[0],
            waypoints=waypoint_array,
        )

    return time.time() - start_time, mp_result_dict


def main():
    """메인 실행 함수"""
    mock_manager = MockManager()
    execution_time, results = select(mock_manager)

    print(f"\nExecution time: {execution_time:.4f} seconds")
    print("\nGenerated Results:")
    for avs_id, result in results.items():
        print(f"  - AVS ID: {avs_id}, Waypoint Count: {result.waypoint_count}")


if __name__ == "__main__":
    main()