from typing import Dict, Tuple

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from policies.surv.algorithms import algo_ADD, algo_ADD2, algo_ADD3, algo_ADD4, algo_LIG
from utils.logger import logger
from utils.recognition import get_avail_avs


class SurvSelector:

    def __init__(self, algo: str, manager: Manager) -> None:
        self.algo = algo
        self.runners = {
            "ADD": algo_ADD.select,
            "ADD2": algo_ADD2.select,
            "ADD3": algo_ADD3.select,
            "ADD4": algo_ADD4.select,
            "LIG": algo_LIG.select,
        }
        self.manager = manager

    def is_runnable(self) -> bool:
        # 감시임무 가능한 비행체 반환
        avs_list, _ = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["S", "s"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # 모든 비행체가 가능한 임무가 없거나 감시임무 가능한 비행체가 없는 경우
        if not self.manager.avs_to_available_task_dict or not avs_list:
            return False

        return True

    def execute_runner(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        if self.algo in self.runners:
            logger.info(f"감시 선별 실행 : {self.algo} 알고리즘")
            runner_fn = self.runners.get(self.algo, algo_ADD3.select)
            return runner_fn(self.manager)
        else:
            raise ValueError(f"지원 불가 감시 선별 : {self.algo} 알고리즘")

    def calculate_performance(
        self, mp_selection_result_dict: Dict[int, SurveillanceSelectionResult], mp_input: MissionPlanInput
    ) -> Dict[str, float]:
        return {"perf_0": None, "perf_1": None, "perf_2": None, "perf_3": None}
