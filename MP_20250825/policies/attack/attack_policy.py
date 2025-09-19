from typing import Dict, List, Tuple, Union

import numpy as np

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from policies.attack.algorithms.algo_ADD import (
    ADDAttackAlgorithmRunner,
    ADDPartialAttackAlgorithmRunner,
)
from policies.attack.algorithms.algo_LIG import (
    LIGAttackAlgorithmRunner,
    LIGPartialAttackAlgorithmRunner,
)
from utils.logger import logger
from utils.recognition import get_avail_avs


class AttackSelector:

    def __init__(self, algo: str, manager: Manager) -> None:
        self.algo = algo
        self.runners = {"ADD": ADDAttackAlgorithmRunner, "LIG": LIGAttackAlgorithmRunner}
        self.manager = manager

    def is_runnable(self) -> bool:
        # 타격임무 가능한 비행체
        avs_list, _ = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["A"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 타격 대기 결과 초기화
        rd_to_attack: Dict[int, Dict[str, List[int]]] = {}

        # 표적 융합 결과에 존재하는 표적에 대해 반복
        for trg_id, fus_out in self.manager.mp_input.trg_fus_res_dict.items():
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
                        return False
                    else:
                        return False
                # 이전 임무계획 결과를 사용하지 않는 경우
                else:
                    #
                    for avs_id, info in self.manager.mp_input.avs_info_dict.items():
                        if info.system_mode == 2:
                            rd_to_attack[trg_id]["track_avs"].append(avs_id)
                            dist = np.linalg.norm(
                                fus_out.position[:2] - self.manager.mp_input.avs_info_dict[avs_id].position[:2]
                            )
                            rd_to_attack[trg_id]["priority"].append(dist)
                            if dist < 500:
                                rd_to_attack[trg_id]["track_avs"].append(avs_id)
                sorted_pairs = sorted(zip(rd_to_attack[trg_id]["priority"], rd_to_attack[trg_id]["track_avs"]))

                # 타격할 표적이 없는 경우
                if len(sorted_pairs) == 0:
                    return False

        # 임무 할당이 가능한 비행체가 없는 경우
        if self.manager.avs_to_available_task_dict == {}:
            return False

        # 타격임무 할당이 가능한 비행체가 없는 경우
        if not avs_list:
            return False

        # 정찰 상태인 표적이 없는 경우
        if 1 not in self.manager.trg_to_state_dict.values():
            return False

        return True

    def execute_runner(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        if self.algo in self.runners:
            logger.info(f"타격 선별 실행 : {self.algo} 알고리즘")
            runner: Union[ADDAttackAlgorithmRunner, LIGAttackAlgorithmRunner] = self.runners[self.algo](self.manager)
            return runner.select()
        else:
            raise ValueError(f"지원 불가 타격 선별 : {self.algo} 알고리즘")

    def calculate_performance(
        self, mp_selection_result_dict: Dict[int, SurveillanceSelectionResult], mp_input: MissionPlanInput
    ) -> Dict[str, float]:
        criteria: float = 4.2e-8
        perfs: Dict[str, float] = {}

        # TAS : 표적까지의 거리 편향
        delta_dists: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            if output.is_bd == 0:
                dist = mp_input.avs_info_dict[avs].position - mp_input.trg_fus_res_dict[output.trg_id].position
                delta_dists.append(dist)
        perfs.update({"tas_avs_dist": np.array(delta_dists).mean() - criteria})

        # BD : 표적까지의 평균 거리
        dists: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            if output.is_bd == 1:
                dists.append(mp_input.avs_info_dict[avs].position - mp_input.trg_fus_res_dict[output.trg_id].position)
        perfs.update({"bd_avs_dist": np.array(dists).mean()})

        # None, None
        perfs.update({"perf_0": None, "perf_1": None})

        return perfs


class PartialAttackSelector:

    def __init__(self, algo: str, manager: Manager) -> None:
        self.algo = algo
        self.runners = {"ADD": ADDPartialAttackAlgorithmRunner, "LIG": LIGPartialAttackAlgorithmRunner}
        self.manager = manager

    def is_runnable(self) -> bool:
        # 타격임무 가능한 비행체
        avs_list, _ = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["A"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 타격 대기 결과 초기화
        rd_to_attack: Dict[int, Dict[str, List[int]]] = {}

        # 표적 융합 결과에 존재하는 표적에 대해 반복
        for trg_id, fus_out in self.manager.mp_input.trg_fus_res_dict.items():
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
                        return False
                    else:
                        return False

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
                sorted_pairs = sorted(zip(rd_to_attack[trg_id]["priority"], rd_to_attack[trg_id]["track_avs"]))

                # 타격할 표적이 없는 경우
                if len(sorted_pairs) == 0:
                    return False

        # 임무 할당이 가능한 비행체가 없는 경우
        if self.manager.avs_to_available_task_dict == {}:
            return False

        # 타격임무 할당이 가능한 비행체가 없는 경우
        if not avs_list:
            return False

        # 정찰 상태인 표적이 없는 경우
        if 1 not in self.manager.trg_to_state_dict.values():
            return False

        return True

    def execute_runner(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult]]:
        if self.algo in self.runners:
            logger.info(f"부분 타격 선별 실행 : {self.algo} 알고리즘")
            runner: Union[ADDPartialAttackAlgorithmRunner, LIGPartialAttackAlgorithmRunner] = self.runners[self.algo](
                self.manager
            )
            return runner.select()
        else:
            raise ValueError(f"지원 불가 부분 타격 선별 : {self.algo} 알고리즘")

    def calculate_performance(
        self, mp_selection_result_dict: Dict[int, SurveillanceSelectionResult], mp_input: MissionPlanInput
    ) -> Dict[str, float]:
        criteria: float = 4.2e-8
        perfs: Dict[str, float] = {}

        # TAS : 표적까지의 거리 편향
        delta_dists: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            if output.is_bd == 0:
                dist = mp_input.avs_info_dict[avs].position - mp_input.trg_fus_res_dict[output.trg_id].position
                delta_dists.append(dist)
        perfs.update({"tas_avs_dist": np.array(delta_dists).mean() - criteria})

        # BD : 표적까지의 평균 거리
        dists: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            if output.is_bd == 1:
                dists.append(mp_input.avs_info_dict[avs].position - mp_input.trg_fus_res_dict[output.trg_id].position)
        perfs.update({"bd_avs_dist": np.array(dists).mean()})

        # None, None
        perfs.update({"perf_0": None, "perf_1": None})

        return perfs
