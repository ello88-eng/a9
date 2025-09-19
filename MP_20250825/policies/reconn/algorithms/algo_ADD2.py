import math
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import ReconnaissanceSelectionResult
from config.mp_config import REPLAN_MODE
from manager.manager import Manager
from utils.coordinates import convert_waypoints_enu_to_lla
from utils.recognition import get_avail_avs, get_avail_trg


class ADD2ReconnaissanceAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

    def find_max_distance(self, a, b, indices: List[int]):
        max_distance = -1  # 가장 긴 거리 (초기값)
        max_index = -1  # 가장 긴 거리의 인덱스

        for idx in indices:
            # a[idx]와 b 사이의 거리 계산
            distance = self.euclidean_distance(a[indices.index(idx)], b)
            if distance > max_distance:
                max_distance = distance
                max_index = idx

        return max_index

    def find_min_distance(self, a, b, indices: List[int]):
        min_distance = float("inf")  # 가장 긴 거리 (초기값)
        min_index = -1  # 가장 긴 거리의 인덱스
        for idx in indices:
            # a[idx]와 b 사이의 거리 계산
            distance = self.euclidean_distance(a[indices.index(idx)], b)
            if distance < min_distance:
                min_distance = distance
                min_index = idx
        return min_index

    def select(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult], MissionPlanInput]:
        # 알고리즘 시작 시간
        start_time = time.time()

        #
        mp_selection_result_dict: Dict[int, ReconnaissanceSelectionResult] = {}

        #
        total_num_cluster = len(avs_list) if len(avs_list) < 5 else 5
        trg_to_avs_dict = deepcopy(self.manager.trg_to_tracker_dict)

        # 임무 재계획 모드가 `수동`이면 운용자 요청 기준으로 표적 선정
        if REPLAN_MODE == "Manual":
            if self.manager.mp_input.mp_cmd_dict:
                # 정찰이 필요한 표적 확인
                trg_list = [
                    trg_id
                    for trg_id, mp_cmd in self.manager.mp_input.mp_cmd_dict.items()
                    if mp_cmd.reconn_request == 1
                ]
                trg_pos_list = [
                    self.manager.mp_input.trg_fus_res_dict[trg_id].position.tolist() for trg_id in trg_list
                ]

        # 임무 재계획 모드가 `자동`이면 표적 상태를 기준으로 표적 선정
        else:
            trg_list, trg_pos_list = get_avail_trg(
                trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
                trg_state_dict=self.manager.trg_to_state_dict,
                expected_state=0,
            )

        avs_list, avs_pos_list = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["R"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 가용하지 않은 비행체는 `trg_to_avs_dict`에서 제거
        for temp_trg_id, temp_avs_list in trg_to_avs_dict.items():
            trg_to_avs_dict[temp_trg_id] = [item for item in temp_avs_list if item in avs_list]

        possible_group_num = len(avs_list) // total_num_cluster
        if len(trg_list) > possible_group_num:
            num_to_remove = len(trg_list) - possible_group_num
            del trg_list[-num_to_remove:]
            del trg_pos_list[-num_to_remove:]

        # Only one avs is assigned to each target
        avs_ids_dict: Dict[int, List[int]] = {}
        for i, trg_id in enumerate(trg_list):
            key = trg_id
            avs_ids_dict[key] = []
            if (trg_id in trg_to_avs_dict) and trg_to_avs_dict[trg_id]:
                detecting_avs_list = trg_to_avs_dict[trg_id]
                detect_min_idx = self.find_min_distance(
                    avs_pos_list, trg_pos_list[trg_list.index(trg_id)], detecting_avs_list
                )
                avs_ids_dict[key].append(detect_min_idx)
                # remove ids from the bank ----
                for key, value in trg_to_avs_dict.items():
                    if detect_min_idx in value:
                        value.remove(detect_min_idx)
                if detect_min_idx in avs_list:
                    avs_list.remove(detect_min_idx)
            else:
                # This 'else' is for the situation that
                # the single avs detect two targets simultaneouly.
                min_idx = self.find_min_distance(avs_pos_list, trg_pos_list[trg_list.index(trg_id)], avs_list)
                avs_ids_dict[key].append(min_idx)
                # remove ids from the bank ----
                for key, value in trg_to_avs_dict.items():
                    if min_idx in value:
                        value.remove(min_idx)
                if min_idx in avs_list:
                    avs_list.remove(min_idx)

        num_cluster = total_num_cluster - 1  # Already 1 avs was assigned (total nuber = 5)
        for i in range(num_cluster):
            for _, trg_id in enumerate(trg_list):
                key = trg_id
                min_idx = self.find_min_distance(
                    avs_pos_list,
                    trg_pos_list[trg_list.index(trg_id)],
                    avs_list,
                )
                avs_ids_dict[key].append(min_idx)
                for key, value in trg_to_avs_dict.items():
                    if min_idx in value:
                        value.remove(min_idx)
                if min_idx in avs_list:
                    avs_list.remove(min_idx)

        for key, value in avs_ids_dict.items():
            avs_ids = value
            for avs_id in avs_ids:
                trg_pt = np.array(trg_pos_list[trg_list.index(key)])

                wps = trg_pt.copy()
                wps[-1] += 300

                mp_selection_result_dict.update(
                    {
                        avs_id: ReconnaissanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=key,
                            num_avs_per_group=5,
                            smp_mode=2,
                            speed=27,
                            waypoint_count=1,
                            waypoints=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=wps,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            turn_radius=300,
                            turn_dir=0,
                            trg_pt=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=trg_pt,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=key,
                            attack_prio=0,
                            is_bd=0,
                        )
                    }
                )
        return time.time() - start_time, mp_selection_result_dict, deepcopy(self.manager.mp_input)


class ADD2PartialReconnAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult], MissionPlanInput]:
        #
        mp_selection_result_dict: Dict[int, ReconnaissanceSelectionResult] = {}

        # 정찰이 필요한 표적 확인
        trg_list, _ = get_avail_trg(
            trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
            trg_state_dict=self.manager.trg_to_state_dict,
            expected_state=1,
        )
        start_time = time.time()

        for trg_id in trg_list:
            for avs_id in self.manager.smp_group_id_to_avs_dict[trg_id]:
                trg_pt = np.array(self.manager.mp_input.trg_fus_res_dict[trg_id].position)
                wps = trg_pt.copy()
                wps[-1] += 300
                mp_selection_result_dict.update(
                    {
                        avs_id: ReconnaissanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=self.manager.avs_to_smp_group_id_dict[avs_id],
                            num_avs_per_group=5,
                            smp_mode=2,
                            speed=27,
                            waypoint_count=1,
                            waypoints=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=wps,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            turn_radius=300,
                            turn_dir=0,
                            trg_pt=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=trg_pt,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=0,
                            is_bd=0,
                        )
                    }
                )
        return time.time() - start_time, mp_selection_result_dict, deepcopy(self.manager.mp_input)
