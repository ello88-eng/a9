from dataclasses import dataclass, field, fields
from typing import Dict

import numpy as np

from commu.sys.receiver.data.fs2mps import TargetFusionResult
from commu.sys.receiver.data.gcs2mps import (
    AvsInfo,
    Boundary,
    CircularMissionArea,
    MissionInitialInfo,
    MissionPlanCommand,
    MissionPlanTarget,
    MissionState,
    PolygonMissionArea,
    ReplanRequest,
)


@dataclass
class MissionPlanInput:
    replan_req_dict: Dict[int, ReplanRequest] = field(default_factory=lambda: {})
    mp_cmd_dict: Dict[int, MissionPlanCommand] = field(default_factory=lambda: {})
    num_avs: int = -1
    num_tgr: int = -1
    boundary: Boundary = field(default_factory=lambda: Boundary())
    avs_info_dict: Dict[int, AvsInfo] = field(default_factory=lambda: {})
    mp_trg_dict: Dict[int, MissionPlanTarget] = field(default_factory=lambda: {})
    trg_fus_res_dict: Dict[int, TargetFusionResult] = field(default_factory=lambda: {})
    grid_coverage: np.ndarray[np.uint8] = field(default_factory=lambda: np.zeros((1,)))
    circular_area: CircularMissionArea = field(default_factory=lambda: CircularMissionArea())
    polygon_area: PolygonMissionArea = field(default_factory=lambda: PolygonMissionArea())
    mission_init_info: MissionInitialInfo = field(default_factory=lambda: MissionInitialInfo())
    mission_state: MissionState = field(default_factory=lambda: MissionState())

    def update_replan_request(self, replan_req_dict: Dict[int, ReplanRequest]) -> None:
        self.replan_req_dict.update(replan_req_dict)

    def update_mission_plan_command(self, mp_cmd_dict: Dict[int, MissionPlanCommand]) -> None:
        self.mp_cmd_dict.update(mp_cmd_dict)

    def update_boundary(self, boundary: Boundary) -> None:
        self.boundary = boundary

    def update_avs_info(self, info: Dict[int, AvsInfo]) -> None:
        self.avs_info_dict.update(info)
        self.num_avs = len(self.avs_info_dict)

    def update_mission_plan_target(self, mp_target_dict: Dict[int, MissionPlanTarget]) -> None:
        self.mp_trg_dict.update(mp_target_dict)

    def update_target_fusion_result(self, target_fusion_result_dict: Dict[int, TargetFusionResult]) -> None:
        self.trg_fus_res_dict.update(target_fusion_result_dict)
        self.num_tgr = len(self.trg_fus_res_dict)

    def update_grid_coverage_data(self, grid_coverage: np.ndarray[np.uint8]) -> None:
        self.grid_coverage = grid_coverage

    def update_circular_mission_area(self, circular_area: CircularMissionArea) -> None:
        self.circular_area = circular_area

    def update_polygon_mission_area(self, polygon_area: PolygonMissionArea) -> None:
        self.polygon_area = polygon_area

    def update_mission_initial_info(self, mission_init_info: MissionInitialInfo) -> None:
        self.mission_init_info = mission_init_info

    def update_mission_state(self, mission_state: MissionState) -> None:
        self.mission_state = mission_state

    def reset_periodically(self) -> None:
        self.replan_req_dict = {}
        self.mp_cmd_dict = {}
        self.num_avs = 0
        self.num_tgr = 0
        self.avs_info_dict = {}
        self.mp_trg_dict = {}
        self.trg_fus_res_dict = {}
        self.grid_coverage = np.zeros((1,))
