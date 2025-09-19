from itertools import combinations
from typing import Dict, List, Tuple, Union

import numpy as np

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import ReconnaissanceSelectionResult
from config.mp_config import REPLAN_MODE
from manager.manager import Manager
from policies.reconn.algorithms.algo_ADD import (
    ADDPartialReconnaissanceAlgorithmRunner,
    ADDReconnaissanceAlgorithmRunner,
)
from policies.reconn.algorithms.algo_ADD2 import ADD2ReconnaissanceAlgorithmRunner
from policies.reconn.algorithms.algo_LIG import (
    LIGPartialReconnaissanceAlgorithmRunner,
    LIGReconnaissanceAlgorithmRunner,
)
from utils.logger import logger
from utils.recognition import get_avail_avs, get_avail_trg


class ReconnSelector:

    def __init__(self, algo: str, manager: Manager) -> None:
        self.algo = algo
        self.runners = {
            "ADD": ADDReconnaissanceAlgorithmRunner,
            "ADD2": ADD2ReconnaissanceAlgorithmRunner,
            "LIG": LIGReconnaissanceAlgorithmRunner,
        }
        self.manager = manager

    def is_runnable(self) -> bool:
        # 임무 재계획 모드가 `수동`인 경우에는 운용자 요청 기준으로 판단
        if REPLAN_MODE == "Manual":
            if not self.manager.mp_input.mp_cmd_dict:
                return False
            else:
                return True

        # 정찰 필요 표적 확인
        trg_list, _ = get_avail_trg(
            trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
            trg_state_dict=self.manager.trg_to_state_dict,
            expected_state=0,
        )
        avs_list, _ = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["R"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 표적 및 비행체 개수가 없는 경우
        if len(trg_list) < 1 or len(avs_list) < 1:
            return False

        # 임무 할당이 가능한 비행체가 없는 경우
        if self.manager.avs_to_available_task_dict == {}:
            return False

        return True

    def execute_runner(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult], MissionPlanInput]:
        if self.algo in self.runners:
            logger.info(f"정찰 선별 실행 : {self.algo} 알고리즘")
            runner: Union[
                ADDReconnaissanceAlgorithmRunner, ADD2ReconnaissanceAlgorithmRunner, LIGReconnaissanceAlgorithmRunner
            ] = self.runners[self.algo](self.manager)
            return runner.select()
        else:
            raise ValueError(f"지원 불가 정찰 선별 : {self.algo} 알고리즘")

    def calculate_performance(
        self, mp_selection_result_dict: Dict[int, ReconnaissanceSelectionResult], mp_input: MissionPlanInput
    ) -> Dict[str, float]:
        # perfs: Dict[str, float] = {}

        # # 표적까지 거리의 평균/분산
        # trg_dists: List[float] = []
        # for avs_id, mp_selection_result in mp_selection_result_dict.items():
        #     trg_dists.append(
        #         np.array(mp_input.avs_info_dict[avs_id].position)
        #         - np.array(mp_input.trg_fus_res_dict[mp_selection_result.target_id].position)
        #     )
        # perfs.update({"norm_trg_dist": float(np.array(trg_dists).mean() / np.array(trg_dists).var())})

        # # 비행체 간 거리의 평균/분산
        # pos_mat: List[float] = []
        # for avs_id, mp_selection_result in mp_selection_result_dict.items():
        #     pos_mat.append(mp_input.avs_info_dict[avs_id].position)
        # pairs = list(combinations(np.array(pos_mat), 2))
        # pair_dists: np.ndarray[float] = np.array([np.linalg.norm(pair[0] - pair[1]) for pair in pairs])
        # perfs.update({"norm_avs_dist": pair_dists.mean() / pair_dists.var()})

        # # 비행체 별 베어링 오차의 평균/분산
        # angles: List[float] = []
        # for avs_id, mp_selection_result in mp_selection_result_dict.items():
        #     los = np.array(mp_input.avs_info_dict[avs_id].position) - np.array(
        #         mp_input.trg_fus_res_dict[mp_selection_result.target_id].position
        #     )
        #     los /= np.linalg.norm(los)
        #     angles.append(np.degrees(np.arccos(np.clip(np.dot(los, np.array([0, 0, 1])), -1.0, 1.0))))
        # perfs.update({"bearing_angs": np.array(angles).mean() / np.array(angles).var()})

        # # None
        # perfs.update({"perf_0": None})

        # return perfs
        return {"norm_trg_dist": 0, "norm_avs_dist": 0, "bearing_angs": 0, "perf_0": 0}


class PartialReconnaissanceSelector:

    def __init__(self, algo: str, manager: Manager) -> None:
        self.algo = algo
        self.runners = {"ADD": ADDPartialReconnaissanceAlgorithmRunner, "LIG": LIGPartialReconnaissanceAlgorithmRunner}
        self.manager = manager

    def is_runnable(self) -> bool:
        # 정찰 필요 표적 확인
        trg_list, _ = get_avail_trg(
            trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
            trg_state_dict=self.manager.trg_to_state_dict,
            expected_state=1,
        )
        avs_list, _ = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["A"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 표적 및 비행체 개수가 1개 미만인 경우
        if len(trg_list) < 1 and len(avs_list) < 1:
            return False

        # 임무 할당이 가능한 비행체가 없는 경우
        if self.manager.avs_to_available_task_dict == {}:
            return False

        return True

    def execute_runner(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult]]:
        if self.algo in self.runners:
            logger.info(f"부분 정찰 선별 실행 : {self.algo} 알고리즘")
            runner: Union[ADDPartialReconnaissanceAlgorithmRunner, LIGPartialReconnaissanceAlgorithmRunner] = (
                self.runners[self.algo](self.manager)
            )
            return runner.select()
        else:
            raise ValueError(f"지원 불가 부분 정찰 선별 : {self.algo} 알고리즘")

    def calculate_performance(
        self, mp_selection_result_dict: Dict[int, ReconnaissanceSelectionResult], mp_input: MissionPlanInput
    ) -> Dict[str, float]:
        perfs: Dict[str, float] = {}

        # 표적까지 거리의 평균/분산
        trg_dists: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            trg_dists.append(
                np.array(mp_input.avs_info_dict[avs].position)
                - np.array(mp_input.trg_fus_res_dict[output.trg_id].position)
            )
        perfs.update({"norm_trg_dist": np.array(trg_dists).mean() / np.array(trg_dists).var()})

        # 비행체 간 거리의 평균/분산
        pos_mat: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            pos_mat.append(mp_input.avs_info_dict[avs].position)
        pairs = list(combinations(np.array(pos_mat), 2))
        pair_dists: np.ndarray[float] = np.array([np.linalg.norm(pair[0] - pair[1]) for pair in pairs])
        perfs.update({"norm_avs_dist": pair_dists.mean() / pair_dists.var()})

        # 비행체 별 베어링 오차의 평균/분산
        angles: List[float] = []
        for avs, output in mp_selection_result_dict.items():
            los = np.array(mp_input.avs_info_dict[avs].position) - np.array(
                mp_input.trg_fus_res_dict[output.trg_id].position
            )
            los /= np.linalg.norm(los)
            angles.append(np.degrees(np.arccos(np.clip(np.dot(los, np.array([0, 0, 1])), -1.0, 1.0))))
        perfs.update({"bearing_angs": np.array(angles).mean() / np.array(angles).var()})

        # 그 외
        perfs.update({"perf_0": None})

        return perfs
