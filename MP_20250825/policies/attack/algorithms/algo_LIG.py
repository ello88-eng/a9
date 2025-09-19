import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from utils.coordinates import convert_waypoints_enu_to_lla


class LIGAttackAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        # 타격 알고리즘 결과 초기화
        mp_output_dict: Dict[int, SurveillanceSelectionResult] = {}

        start_time = time.time()

        # 타격 대기 결과 초기화
        rd_to_attack: Dict[int, Dict[str, List[int]]] = {}
        # 표적 융합 결과에 존재하는 표적에 대해 반복
        for trg_id, fus_out in self.manager.mp_input.trg_fus_res_dict.items():
            #
            wps = fus_out.position.copy()
            wps[-1] += 300

            if self.manager.trg_to_state_dict[trg_id] == 1:
                # 해당 표적에 대해 타격 대기 결과 초기화
                rd_to_attack[trg_id] = {}
                rd_to_attack[trg_id]["total_avs"] = []
                rd_to_attack[trg_id]["track_avs"] = []
                rd_to_attack[trg_id]["priority"] = []

                # 이전 임무계획 결과를 사용하는 경우
                if self.manager.use_prev_mp_output:
                    # 이전 임무계획 결과가 없는 경우 동작하지 않음
                    if not hasattr(self.manager, "prev_mp_output_dict"):
                        return None, None, None
                    # TODO : 20241216 : 추적 상태 확인
                    else:
                        return None, None, None
                # 이전 임무계획 결과를 사용하지 않는 경우
                else:
                    #
                    for avs_id, info in self.manager.mp_input.avs_info_dict.items():
                        if info.avs_system_mode == 2:
                            rd_to_attack[trg_id]["track_avs"].append(avs_id)
                            dist = np.linalg.norm(
                                fus_out.position[:2] - self.manager.mp_input.avs_info_dict[avs_id].position[:2]
                            )
                            rd_to_attack[trg_id]["priority"].append(dist)
                            if dist < 500:
                                rd_to_attack[trg_id]["track_avs"].append(avs_id)

                #
                sorted_pairs = sorted(
                    zip(
                        rd_to_attack[trg_id]["priority"],
                        rd_to_attack[trg_id]["track_avs"],
                    )
                )
                _, avs_sorted = zip(*sorted_pairs)

                #
                if len(rd_to_attack[trg_id]["track_avs"]) > 4:
                    max_ats = 3
                else:
                    max_ats = 2

                #
                num_ats = 1
                for avs_id in avs_sorted:
                    #
                    if num_ats > max_ats:
                        #
                        mp_output_dict[avs_id] = SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=(
                                self.manager.avs_to_smp_group_id_dict[avs_id]
                                if avs_id in self.manager.avs_to_smp_group_id_dict
                                else trg_id
                            ),
                            num_avs_per_group=len(rd_to_attack[trg_id]["track_avs"]),
                            smp_mode=4,
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
                                waypoint_array_as_enu=fus_out.position,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=0,
                            is_bd=1,
                        )
                    else:
                        mp_output_dict[avs_id] = SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=(
                                self.manager.avs_to_smp_group_id_dict[avs_id]
                                if avs_id in self.manager.avs_to_smp_group_id_dict
                                else trg_id
                            ),
                            num_avs_per_group=len(rd_to_attack[trg_id]["track_avs"]),
                            smp_mode=4,
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
                                fus_out.position,
                                self.manager.lat0,
                                self.manager.lon0,
                                self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=num_ats,
                            is_bd=0,
                        )
                    #
                    num_ats += 1
        return time.time() - start_time, mp_output_dict, deepcopy(self.manager.mp_input)


class LIGPartialAttackAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        # 타격 알고리즘 결과 초기화
        mp_output_dict: Dict[int, SurveillanceSelectionResult] = {}

        start_time = time.time()

        # 타격 대기 결과 초기화
        rd_to_attack: Dict[int, Dict[str, List[int]]] = {}
        # 표적 융합 결과에 존재하는 표적에 대해 반복
        for trg_id, fus_out in self.manager.mp_input.trg_fus_res_dict.items():
            #
            wps = fus_out.position.copy()
            wps[-1] += 300

            if self.manager.trg_to_state_dict[trg_id] == 1:
                # 해당 표적에 대해 타격 대기 결과 초기화
                rd_to_attack[trg_id] = {}
                rd_to_attack[trg_id]["total_avs"] = []
                rd_to_attack[trg_id]["track_avs"] = []
                rd_to_attack[trg_id]["priority"] = []

                # 이전 임무계획 결과를 사용하는 경우
                if self.manager.use_prev_mp_output:
                    # 이전 임무계획 결과가 없는 경우 동작하지 않음
                    if not hasattr(self.manager, "prev_mp_output_dict"):
                        return None, None, None
                    # TODO : 20241216 : 추적 상태 확인
                    else:
                        return None, None, None
                # 이전 임무계획 결과를 사용하지 않는 경우
                else:
                    #
                    for avs_id, info in self.manager.mp_input.avs_info_dict.items():
                        if info.avs_system_mode == 2:
                            rd_to_attack[trg_id]["track_avs"].append(avs_id)
                            dist = np.linalg.norm(
                                fus_out.position[:2] - self.manager.mp_input.avs_info_dict[avs_id].position[:2]
                            )
                            rd_to_attack[trg_id]["priority"].append(dist)
                            if dist < 500:
                                rd_to_attack[trg_id]["track_avs"].append(avs_id)

                #
                sorted_pairs = sorted(
                    zip(
                        rd_to_attack[trg_id]["priority"],
                        rd_to_attack[trg_id]["track_avs"],
                    )
                )
                _, avs_sorted = zip(*sorted_pairs)

                #
                if len(rd_to_attack[trg_id]["track_avs"]) > 4:
                    max_ats = 3
                else:
                    max_ats = 2

                #
                num_ats = 1
                for avs_id in avs_sorted:
                    #
                    if num_ats > max_ats:
                        #
                        mp_output_dict[avs_id] = SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=(
                                self.manager.avs_to_smp_group_id_dict[avs_id]
                                if avs_id in self.manager.avs_to_smp_group_id_dict
                                else trg_id
                            ),
                            num_avs_per_group=len(rd_to_attack[trg_id]["track_avs"]),
                            smp_mode=4,
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
                                waypoint_array_as_enu=fus_out.position,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=0,
                            is_bd=1,
                        )
                    else:
                        mp_output_dict[avs_id] = SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=(
                                self.manager.avs_to_smp_group_id_dict[avs_id]
                                if avs_id in self.manager.avs_to_smp_group_id_dict
                                else trg_id
                            ),
                            num_avs_per_group=len(rd_to_attack[trg_id]["track_avs"]),
                            smp_mode=4,
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
                                fus_out.position,
                                self.manager.lat0,
                                self.manager.lon0,
                                self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=num_ats,
                            is_bd=0,
                        )
                    #
                    num_ats += 1
        return time.time() - start_time, mp_output_dict, deepcopy(self.manager.mp_input)
