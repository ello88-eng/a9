from typing import Dict, List, Tuple, Union

import numpy as np

from config.avs_config import SmpMode
from config.mp_config import (
    GEN_SURV_PATH,
    MISSION_MODE_TO_NAME_DICT,
    REPLAN_MODE,
    TRG_POS_CHANGE_THRESHOLD,
    TRG_TRACK_THRESHOLD,
)
from manager.manager import Manager
from replanner.algorithms.algo_ADD import ADDReplanAlgorithmRunner
from replanner.algorithms.algo_LIG import LIGReplanAlgorithmRunner
from utils.logger import logger


class ReplanSelector:

    def __init__(self, algo: str, manager: Manager, trg_track_cnt_dict: Dict[int, int]):
        self.algo = algo
        self.manager = manager
        self.runners = {"ADD": ADDReplanAlgorithmRunner, "LIG": LIGReplanAlgorithmRunner}

        self.need_full_replan = False
        self.need_partial_replan = False

        self.trg_track_cnt_dict = trg_track_cnt_dict

    def _check_avs_count_change(self) -> None:
        """비행체 임무상태별 개수 비교"""
        if hasattr(self.manager, "prev_mp_input"):
            prev_avs_cnt = len(self.manager.prev_mp_input.avs_info_dict)
            curr_avs_cnt = len(self.manager.mp_input.avs_info_dict)

            if prev_avs_cnt != curr_avs_cnt:
                logger.info(f"비행체 수 변경: {prev_avs_cnt} -> {curr_avs_cnt}")
                self.need_full_replan = True
            else:
                logger.info(f"비행체 수 유지: {prev_avs_cnt} -> {curr_avs_cnt}")

    def _check_trg_count_change(self) -> None:
        """표적융합결과 기반 표적 개수 비교"""
        if hasattr(self.manager, "prev_mp_input"):
            accum_trg_set = set(self.manager.accum_trg_fus_res.keys())
            curr_trg_set = set(self.manager.mp_input.trg_fus_res_dict.keys())

            if len(curr_trg_set - accum_trg_set) > 0:
                logger.info(f"신규 표적 식별: {list(curr_trg_set - accum_trg_set)}")
                self.need_full_replan = True
            else:
                logger.info(f"표적 개수 유지/감소: {list(accum_trg_set)} -> {list(curr_trg_set)}")

    def _check_trg_state_change(self) -> None:
        """표적 상태 변경 비교"""
        # 이전/현재 시점의 표적 전역 ID의 집합
        prev_trg_set = {trg_id for trg_id in self.manager.prev_trg_to_state_dict}
        curr_trg_set = {trg_id for trg_id in self.manager.trg_to_state_dict}

        # 전역 ID가 동일한 표적 중에서 상태가 변화된 표적에 대해 이전/현재 상태 리턴
        changed_trg_to_state_dict = {
            trg_id: (self.manager.prev_trg_to_state_dict[trg_id], self.manager.trg_to_state_dict[trg_id])
            for trg_id in prev_trg_set & curr_trg_set
            if self.manager.prev_trg_to_state_dict[trg_id] != self.manager.trg_to_state_dict[trg_id]
        }
        if changed_trg_to_state_dict:
            for trg_id, (prev_trg_state, curr_trg_state) in changed_trg_to_state_dict.items():
                logger.info(f"표적 T{trg_id}의 상태 변경: {prev_trg_state} -> {curr_trg_state}")
            self.need_full_replan = True
        else:
            if not self.need_full_replan:
                logger.info(f"표적 상태 변경 없음")

    def _check_trg_position_change(self) -> None:
        """동일한 표적 전역 ID에 대해 표적 위치가 변경되었는지 확인"""
        if hasattr(self.manager, "prev_mp_input"):
            # 이전/현재 시점의 표적 전역 ID의 집합
            accum_trg_set = set(self.manager.accum_trg_fus_res.keys())
            curr_trg_set = set(self.manager.mp_input.trg_fus_res_dict.keys())

            self.moved_trg_dict = []
            if curr_trg_set.issubset(accum_trg_set):
                for trg_id in curr_trg_set:
                    accum_trg_pos = self.manager.accum_trg_fus_res[trg_id].position
                    curr_trg_pos = self.manager.mp_input.trg_fus_res_dict[trg_id].position
                    # 표적의 위치 변화가 기준 값 이상인 경우
                    if np.linalg.norm(accum_trg_pos - curr_trg_pos) >= TRG_POS_CHANGE_THRESHOLD:
                        logger.info(f"표적 T{trg_id} 위치 변경: {accum_trg_pos} -> {curr_trg_pos}")
                        self.moved_trg_dict.append(trg_id)
            else:
                self.need_full_replan = True

            if self.moved_trg_dict:
                self.need_partial_replan = True

    def _check_track_success(self) -> None:
        for trg_id, accum_trg_fus_res in self.manager.accum_trg_fus_res.items():
            if self.manager.trg_to_state_dict[trg_id] == 1:
                # smp_group_id == target_id 인 avs_id 집합
                avs_ids_by_smp_set = {
                    avs.avs_id
                    for avs in self.manager.mp_input.avs_info_dict.values()
                    if avs.smp_group_id == trg_id and avs.implement_mode == SmpMode.RECONN
                }

                # local_info_list에 있는 avs_id 집합
                avs_ids_in_local_set = {local_info.avs_id for local_info in accum_trg_fus_res.local_info_list}

                # 완전히 동일한지 비교
                self.trg_track_cnt_dict[trg_id] += int(avs_ids_by_smp_set == avs_ids_in_local_set)

                # 추적 성공이 `TRG_TRACK_THRESHOLD` 횟수를 넘으면 추적 완료로 전환
                if self.trg_track_cnt_dict[trg_id] >= TRG_TRACK_THRESHOLD:
                    logger.info(f"표적 ID: T{trg_id} 추적 완료 전환")
                    self.need_full_replan = True

    def judge_replan_or_not(self, step: int, manager: Manager) -> Tuple[bool, bool]:
        """임무계획 입력 기반 임무 재계획 여부 및 수준 판단"""

        # 운용자 재계획 요청이 있는 경우
        if any(req.need_request == 2 for req in manager.mp_input.replan_req_dict.values()):
            logger.info("임무 재계획 요청 수신")
            self.need_full_replan = True

        # 운용자 정찰 요청이 있는 경우
        if any(mp_cmd.reconn_request == 1 for mp_cmd in manager.mp_input.mp_cmd_dict.values()):
            logger.info("운용자 정찰 요청 수신")
            self.need_full_replan = True

        #
        if GEN_SURV_PATH and manager.mp_input.mission_init_info.surv_path_gen_method == 1:
            self.need_full_replan = True
            manager.mp_input.mission_init_info.surv_path_gen_method = 0

        # 임무 재계획 모드가 `수동`인 경우
        if REPLAN_MODE == "Manual":
            return self.need_full_replan, self.need_partial_replan

        # 표적 추적 완료 여부 확인
        self._check_track_success()

        # 최초 실행인 경우
        if step == 0:
            self.need_full_replan = True

        # 어느 비행체라도 할당된 임무(감시, 정찰, 타격)가 없으면 전체 재계획 실행
        if any(mode == MISSION_MODE_TO_NAME_DICT[0] for mode in self.manager.avs_to_implement_mode_dict.values()):
            self.need_full_replan = True

        # 비행체 개수 변화 여부 확인
        self._check_avs_count_change()
        # 표적 개수 변화 여부 확인
        self._check_trg_count_change()
        # 표적 상태 변화 여부 확인
        self._check_trg_state_change()

        if not self.need_full_replan:
            # 표적 위치변화 여부 확인
            self._check_trg_position_change()

        return self.need_full_replan, self.need_partial_replan

    def run_full_replan(self) -> Dict[int, List[str]]:
        if self.algo in self.runners:
            logger.info(f"재계획 선별 실행: {self.algo} 알고리즘")
            runner: Union[ADDReplanAlgorithmRunner, LIGReplanAlgorithmRunner] = self.runners[self.algo](self.manager)

            return runner.select()
        else:
            raise ValueError(f"지원하지 않은 알고리즘: {self.algo}")

    def run_partial_replan(self) -> Dict[int, int]:
        # 정찰 또는 타격임무에 대해 비행체 별 할당된 표적 전역 ID
        trg_to_avs_dict = {}

        for trg_id, curr_fus_out in self.manager.mp_input.trg_fus_res_dict.items():
            accum_fus_out = self.manager.accum_trg_fus_res[trg_id]

            # 이전에 해당 표적이 없었던 경우
            if accum_fus_out is None:
                continue

            # 이전 및 현재 시점에서 동일 표적의 위치 변화
            pos_diff = np.linalg.norm(curr_fus_out.position[2:] - accum_fus_out.position[2:])
            # 위치 변화가 100m 이상인 경우
            avs_ids = []
            if pos_diff >= 100:
                for loc_info in curr_fus_out.local_info_list:
                    avs_ids.append(loc_info.avs_id)
                dict.update(trg_to_avs_dict, {trg_id: avs_ids})
            avs_ids = []

        return trg_to_avs_dict
