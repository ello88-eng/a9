from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Generic, List, TypeVar, Union

from commu.sys.sender.data.sys_out import (
    AttackSelectionResult,
    BaseSelectionResult,
    MissionPlanResultInfo,
    ReconnaissanceSelectionResult,
    SurveillanceSelectionResult,
)
from config.avs_config import SmpMode

T = TypeVar("T", bound=BaseSelectionResult)

ReconnOrAttackType = Union[ReconnaissanceSelectionResult, AttackSelectionResult]


class MissionBatchFactory(ABC, Generic[T]):
    @abstractmethod
    def create_mission_plan_results(self, selection_result: Dict[int, T]) -> List[MissionPlanResultInfo]:
        pass


class ReconnOrAttackMissionFactory(MissionBatchFactory[ReconnOrAttackType]):
    def create_mission_plan_results(
        self, selection_result_dict: Dict[int, ReconnOrAttackType]
    ) -> List[MissionPlanResultInfo]:
        if not selection_result_dict:
            return []

        # smp_group_id 기준으로 키(비행기 번호) 묶기
        groups: Dict[int, List[int]] = defaultdict(list)
        for avs_id, selection_result in selection_result_dict.items():
            groups[selection_result.smp_group_id].append(avs_id)

        mp_result_info_list: List[MissionPlanResultInfo] = []
        for smp_group_id, avs_list in groups.items():
            avs_list_sorted = sorted(avs_list)
            mp_result_info_list.append(
                MissionPlanResultInfo(
                    smp_group_id=smp_group_id,
                    mission_info=selection_result.smp_mode,
                    avs_count=len(avs_list_sorted),
                    avs_list=avs_list_sorted,
                )
            )

        return mp_result_info_list


class SurveillanceMissionBatchFactory(MissionBatchFactory[SurveillanceSelectionResult]):
    def create_mission_plan_results(
        self, selection_result_dict: Dict[int, SurveillanceSelectionResult]
    ) -> List[MissionPlanResultInfo]:
        if not selection_result_dict:
            return []

        avs_list = sorted(selection_result_dict.keys())

        return [
            MissionPlanResultInfo(
                smp_group_id=0, mission_info=SmpMode.SURV.value, avs_count=len(avs_list), avs_list=avs_list
            )
        ]


# ------------------------------
# Service
# ------------------------------
class MissionPlanService:
    _factories = {
        "Attack": ReconnOrAttackMissionFactory(),
        "Reconn": ReconnOrAttackMissionFactory(),
        "Surv": SurveillanceMissionBatchFactory(),
    }

    @classmethod
    def create_mp_result_info_batch(
        cls, mission: str, mp_selection_result_dict: Dict[int, BaseSelectionResult]
    ) -> List[MissionPlanResultInfo]:
        factory: Union[ReconnOrAttackMissionFactory, SurveillanceMissionBatchFactory] = cls._factories.get(mission)

        return factory.create_mission_plan_results(mp_selection_result_dict)
