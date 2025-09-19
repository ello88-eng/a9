import time
from types import FunctionType
from typing import Dict, Union

from commu.sys.sender.sys_sender import (
    send_attack_selection_result,
    send_mp_result_info,
    send_reconn_selection_result,
    send_surv_selection_result,
)
from config.main_config import MainArgs
from config.sys_msg_config import MP_RESULT_SEND_INTERVAL
from factory.mp_factory import MissionPlanService
from manager.manager import Manager
from policies.attack.attack_policy import AttackSelector, PartialAttackSelector
from policies.reconn.reconn_policy import PartialReconnaissanceSelector, ReconnSelector
from policies.surv.surv_policy import SurvSelector
from replanner.replan_policy import ReplanSelector
from utils.logger import logger

SelectorType = Union[
    PartialAttackSelector, PartialReconnaissanceSelector, AttackSelector, ReconnSelector, SurvSelector
]


class Planner:
    def __init__(self, args: MainArgs) -> None:
        self.args = args

        # 전체 재계획 / 부분 재계획 선별 클래스 매핑
        self.selector_map_dict: Dict[str, Union[AttackSelector, ReconnSelector, SurvSelector]] = {
            "Attack": AttackSelector,
            "Reconn": ReconnSelector,
            "Surv": SurvSelector,
        }
        self.partial_selector_map_dict: Dict[str, Union[PartialAttackSelector, PartialReconnaissanceSelector]] = {
            "Attack": PartialAttackSelector,
            "Reconn": PartialReconnaissanceSelector,
        }

        # 임무계획 감시/정찰/타격 선별 결과 전송 메서드 매핑
        self.sender_fn_dict: Dict[str, FunctionType] = {
            "Attack": send_attack_selection_result,
            "Reconn": send_reconn_selection_result,
            "Surv": send_surv_selection_result,
        }
        self.trg_state_map_dict = {"Attack": 3, "Track": 2, "Reconn": 1}
        self.step = 0
        self.heartbeat_dict: Dict[str, int] = {"Info": 1, "Attack": 1, "Reconn": 1, "Surv": 1}
        self.mission_to_cnt_dict = {
            "Full Replan": 0,
            "Partial Replan": 0,
            "Surv": 0,
            "Reconn": 0,
            "Partial Reconn": 0,
            "Attack": 0,
            "Partial Attack": 0,
        }

        self.need_full_replan = False
        self.need_partial_replan = False

        self.trg_track_cnt_dict: Dict[int, int] = {}

    def _run_pipeline(self, mission: str, manager: Manager, partial: bool, show_output: bool) -> None:
        # 전체 재계획 / 부분 재계획 선별 공통 실행 파이프라인
        if partial and mission == "Surv":
            # Surv는 부분 재계획 없음
            return

        selector_map = self.partial_selector_map_dict if partial else self.selector_map_dict
        if mission not in selector_map:
            raise ValueError(f"`MISSION_LIST`에 {mission} 추가 필요")

        selector_cls: SelectorType = selector_map.get(mission)
        if selector_cls is None:
            return
        selector: SelectorType = selector_cls(algo=manager.mission_to_algo_dict.get(mission), manager=manager)

        if not selector.is_runnable():
            return

        logger.info(f"==================== STEP: {self.step} ====================")

        # 실행
        delta_t, mp_selection_result_dict, mp_input = selector.execute_runner()
        # 실행 횟수 카운트
        if partial:
            self.mission_to_cnt_dict[f"Partial {mission}"] += 1
        else:
            self.mission_to_cnt_dict[mission] += 1

        if not mp_selection_result_dict:
            self.step += 1
            return

        # 감시/정찰/타격 선별 알고리즘 성능 계산
        perf_dict = selector.calculate_performance(
            mp_selection_result_dict=mp_selection_result_dict, mp_input=mp_input
        )

        # 감시/정찰/타격 선별 결과 메모리에 저장
        manager.update_mp_selection_result_logger(
            mission=mission, delta_t=delta_t, mp_selection_result_dict=mp_selection_result_dict, perf_dict=perf_dict
        )
        # 메모리에 저장된 감시/정찰/타격 선별 결과를 JSON에 저장
        manager.save_mp_selection_result_as_json(mission)

        # 표적 상태 업데이트
        state = self.trg_state_map_dict.get(mission)
        if state is not None:
            manager.update_trg_state(mp_selection_result_dict=mp_selection_result_dict, state=state)

        # 그룹별 임무계획 결과 정보 생성
        mp_result_info_batch = MissionPlanService.create_mp_result_info_batch(
            mission=mission, mp_selection_result_dict=mp_selection_result_dict
        )

        # 결과 전송
        for mp_result_info in mp_result_info_batch:
            # 그룹 임무계획 결과 전송
            send_mp_result_info(
                args=self.args, mp_result_info=mp_result_info, heartbeat=self.heartbeat_dict.get("Info")
            )
            # 하트비트 업데이트
            self.heartbeat_dict["Info"] = (self.heartbeat_dict.get("Info") + 1) % 65535
            # 임무계획 결과 전송 인터벌
            time.sleep(MP_RESULT_SEND_INTERVAL)

            # 임무계획 감시/정찰/타격 생성 결과 전송 함수 선택
            selection_result_sender_fn = self.sender_fn_dict[mission]
            # 그룹 내 비행체 추출
            for avs_id in mp_result_info.avs_list:
                # 그룹 내 비행체별 임무계획 감시/정찰/타격 생성 결과 전송
                selection_result_sender_fn(
                    args=self.args,
                    result=mp_selection_result_dict.get(avs_id),
                    show_output=show_output,
                    heartbeat=self.heartbeat_dict.get(mission),
                    src=manager.src,
                )
                # 임무계획 결과 전송 인터벌
                time.sleep(MP_RESULT_SEND_INTERVAL)
                # 하트비트 업데이트
                self.heartbeat_dict[mission] = (self.heartbeat_dict.get(mission) + 1) % 65535

        # 임무계획 스텝 업데이트
        self.step += 1

    def run_selector(self, mission: str, manager: Manager, show_output: bool = False) -> None:
        self._run_pipeline(mission=mission, manager=manager, partial=False, show_output=show_output)

    def run_partial_selector(self, mission: str, manager: Manager, show_output: bool = False) -> None:
        self._run_pipeline(mission=mission, manager=manager, partial=True, show_output=show_output)

    def _judge_replan_or_not(self, manager: Manager) -> None:
        # 임무 별 실행횟수 확인
        logger.info(f"임무별 실행 횟수: {self.mission_to_cnt_dict}")

        for trg_id in manager.accum_trg_fus_res.keys():
            # 없으면 추가
            if not trg_id in self.trg_track_cnt_dict.keys():
                self.trg_track_cnt_dict[trg_id] = 0

        # 임무 재계획 여부 및 수준 판단
        replan_selector = ReplanSelector(
            algo=manager.mission_to_algo_dict.get("Replan"),
            manager=manager,
            trg_track_cnt_dict=self.trg_track_cnt_dict,
        )

        self.need_full_replan, self.need_partial_replan = replan_selector.judge_replan_or_not(
            step=self.step, manager=manager
        )

        # 전체 임무 재계획이 필요한 경우
        if self.need_full_replan:
            logger.info("전체 임무 재계획 실행")
            manager.avs_to_available_task_dict = replan_selector.run_full_replan()

        # 부분 재계획이 필요한 경우
        elif self.need_partial_replan:
            logger.info("부분 임무 재계획 실행")
            manager.trg_to_tracker_dict = replan_selector.run_partial_replan()

        # 임무 재계획이 필요없는 경우
        else:
            logger.info("임무 재계획 없음")

    def plan(self, manager: Manager, show_output: bool = False) -> None:
        # 재계획 여부 및 수준에 따라 비행체 임무 할당
        for mission in manager.mission_list:
            logger.info(f"==================== {mission} 선별 검토 시작 ====================")

            # 재계획 여부 및 수준 판단 + 비행체별 수행가능 임무 할당
            self._judge_replan_or_not(manager)

            # 전체 재계획이 필요한 경우
            if self.need_full_replan:
                self.run_selector(mission=mission, manager=manager, show_output=show_output)
            # 부분 재계획이 필요한 경우
            if self.need_partial_replan:
                self.run_partial_selector(mission=mission, manager=manager, show_output=show_output)

            logger.info(f"==================== {mission} 선별 검토 종료 ====================")
