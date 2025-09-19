import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from json.encoder import JSONEncoder
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np

from commu.sys.receiver.data.fs2mps import TargetFusionResult, TargetLocalInfo
from commu.sys.receiver.data.gcs2mps import AvsInfo, MissionPlanTarget
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import (
    AttackSelectionResult,
    ReconnaissanceSelectionResult,
    SurveillanceSelectionResult,
    TrackSelectionResult,
)
from config.mp_config import MISSION_MODE_TO_NAME_DICT
from utils.coordinates import (
    convert_avs_pos_lla_to_enu,
    convert_trg_pos_lla_to_enu,
    split_origin,
)
from utils.logger import logger
from utils.timer import get_timestamp


class DataToJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(
            obj,
            (
                MissionPlanInput,
                SurveillanceSelectionResult,
                ReconnaissanceSelectionResult,
                AttackSelectionResult,
                TargetFusionResult,
                MissionPlanTarget,
                TargetLocalInfo,
                AvsInfo,
            ),
        ):
            return asdict(obj)

        if isinstance(obj, np.int64):
            return int(obj)

        return super(DataToJsonEncoder, self).default(obj)


# 커스텀 타입 정의
SelectionResultType = Union[SurveillanceSelectionResult, ReconnaissanceSelectionResult, AttackSelectionResult]
LoggerType = Union[int, MissionPlanInput, Dict[str, Union[str, SelectionResultType, Dict[str, float]]]]


class Manager:

    def __init__(
        self,
        mission_list: List[str] = ["Attack", "Reconn", "Surv"],
        num_rows: int = 200,
        num_columns: int = 200,
        replan_algo: str = "ADD",
        surv_algo: str = "ADD",
        reconn_algo: str = "ADD",
        attack_algo: str = "ADD",
        scenario_fp: str = "",
        use_mp_trg: bool = False,
        use_prev_mp_result: bool = False,
        src_id: int = 1,
    ) -> None:
        # 수행할 임무 리스트
        self.mission_list = mission_list

        # 임무계획 결과 초기화
        self.mp_io_log: Dict[str, LoggerType] = {
            "timestamp": None,
            "state": None,
            "result": {"mission": None, "algorithm": None, "output": None, "perf": {}, "barchart": {}},
        }

        # 비행체 별 할당 가능 임무
        self.avs_to_available_task_dict: Dict[int, List[str]] = {}
        # 가로 및 세로 그리드 개수
        self.num_rows = num_rows
        self.num_columns = num_columns

        # 임무 별 사용할 알고리즘 정의
        self.mission_to_algo_dict = {
            "Replan": replan_algo,
            "Surv": surv_algo,
            "Reconn": reconn_algo,
            "Attack": attack_algo,
        }

        # SMP 그룹 딕셔너리 초기화
        self.avs_to_smp_group_id_dict: Dict[int, int] = {}
        self.smp_group_id_to_avs_dict: Dict[int, Set[int]] = {}

        # 임무계획 입력 및 결과를 저장할 위치 설정
        base_dir = Path("/mnt/analysis")
        date_now = datetime.now().strftime("%Y%m%d")
        time_now = datetime.now().strftime("%H%M%S")

        # 시나리오 경로가 없으면 임시로 `TEST`로 만듬
        if not scenario_fp:
            scenario_fp = "TEST"
        scenario_name = Path(scenario_fp).stem.split("_")[-1]
        tag = f"{date_now}_{time_now}_{scenario_name}"
        self.state_dir = base_dir / "state" / tag
        self.result_dirs = {m: base_dir / "result" / m.lower() / tag for m in ("Surv", "Reconn", "Attack")}
        # 디렉터리 일괄 생성
        for d in (*self.result_dirs.values(), self.state_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 지상체로부터 표적 상태를 수신할지 여부
        self.use_mp_trg = use_mp_trg

        # 표적별 상태 (식별됨: 0, 정찰 할당됨: 1, 정찰 완료됨: 2, 타격 할당됨: 3, 타격 완료됨: 4)
        self.trg_to_state_dict: Dict[int, int] = {}

        # 비행체별 임무 모드 (감시, 정찰, 추적, 타격)
        self.avs_to_implement_mode_dict: Dict[int, str] = {}

        # 표적별 식별한 비행체 ID 리스트 초기화
        self.trg_to_tracker_dict: Dict[int, List[int]] = {}

        # 이전 표적 상태 딕셔너리
        self.prev_trg_to_state_dict: Dict[int, int] = {}

        # 이전 비행체 임무모드 딕셔너리
        self.prev_avg_to_implement_mode_dict: Dict[int, str] = {}

        # 이전 임무계획 결과 활용 여부
        self.use_prev_mp_output = use_prev_mp_result

        # 모의환경 인덱스
        self.src = src_id

        # 표적융합결과 누적 딕셔너리
        self.accum_trg_fus_res: Dict[int, TargetFusionResult] = {}

        # 감시 커버리지 비율
        self.coverage = 0.0
        # 감시 커버리지 JSON 파일 초기화
        self._init_coverage_json()

    def _init_coverage_json(self):
        """Coverage JSON 파일이 없으면 기본 구조로 초기화"""
        self.coverage_fp = self.state_dir / "coverage.json"
        if not self.coverage_fp.exists():
            template = {
                "algorithm": self.mission_to_algo_dict["Surv"],
                "label": {"x": "timestamp", "y": "coverage"},
                "data": [{"timestamp": get_timestamp(), "coverage": 0.0}],
            }
            self._save_json(self.coverage_fp, template)

    def _load_json(self, fp: Path) -> dict:
        """JSON 파일 로드"""
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, fp: Path, obj: dict):
        """JSON 파일 저장"""
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

    def _compute_coverage(self, grid_data: np.ndarray) -> float:
        """0, 1 배열에서 1의 비율 계산"""
        return float(np.asarray(grid_data).mean())

    def _append_coverage(self, fp: Path, coverage: float):
        """현재 시각과 coverage를 JSON의 data 리스트에 추가"""
        obj: Dict[str, List[Dict]] = self._load_json(fp)
        obj["data"].append({"timestamp": get_timestamp(), "coverage": coverage})
        self._save_json(fp=fp, obj=obj)

    def _update_coverage_json(self, grid_data: np.ndarray):
        """주기적으로 배열 생성 → coverage 계산 → JSON 업데이트 반복"""
        coverage = self._compute_coverage(grid_data)
        self._append_coverage(fp=self.coverage_fp, coverage=coverage)

    def save_input_as_json(self) -> None:
        logger.info(f"임무계획 입력정보 저장 : {self.state_dir}/timestamp_{get_timestamp()}.json")
        with open(f"{self.state_dir}/timestamp_{get_timestamp()}.json", "w") as f:
            json.dump({"state": self.mp_input_lla}, f, indent=4, cls=DataToJsonEncoder)

    def update_input(self, mp_input: MissionPlanInput) -> None:
        # 이전 임무계획 입력 정보를 신규 임무계획 입력 정보로 업데이트
        if not hasattr(self, "mp_input"):
            self.mp_input = deepcopy(mp_input)
        else:
            self.prev_mp_input = deepcopy(self.mp_input)
            self.mp_input = deepcopy(mp_input)

        # 실시간 및 누적 표적융합결과 정보 도시
        logger.info(f"실시간 표적융합결과 (전역 ID): {mp_input.trg_fus_res_dict.keys()}")
        self.accum_trg_fus_res.update(self.mp_input.trg_fus_res_dict)
        logger.info(f"누적 표적융합결과 (전역 ID): {self.accum_trg_fus_res.keys()}")

        # 임무계획 입력을 LLA 좌표로 저장하기 위한 raw 데이터
        self.mp_input_lla = deepcopy(mp_input)
        self.mp_io_log["state"] = deepcopy(mp_input)

        # lat0, lon0, alt0 중 하나라도 정의되어 있지 않을 때만 split_origin() 메서드 실행
        if not (hasattr(self, "lat0") and hasattr(self, "lon0") and hasattr(self, "alt0")):
            self.lat0, self.lon0, self.alt0 = split_origin(mp_input.boundary.vertices)

        # 좌표 변환
        self.mp_input.avs_info_dict = convert_avs_pos_lla_to_enu(
            infos=mp_input.avs_info_dict, lat0=self.lat0, lon0=self.lon0, alt0=self.alt0
        )
        self.mp_input.trg_fus_res_dict = convert_trg_pos_lla_to_enu(
            fus_out_dict=mp_input.trg_fus_res_dict, lat0=self.lat0, lon0=self.lon0, alt0=self.alt0
        )
        # 감시 커버리지 JSON 파일 업데이트
        self._update_coverage_json(deepcopy(mp_input.grid_coverage))

        # 비행체 자동임무 모드 정보가 비어있으면, 입력된 자동임무 모드로 업데이트
        for avs_id, avs_info in self.mp_input.avs_info_dict.items():
            # 비행체별 자동임무 모드 테이블로 업데이트
            self.avs_to_implement_mode_dict[avs_id] = MISSION_MODE_TO_NAME_DICT.get(avs_info.implement_mode)
        logger.info(f"AVS별 자동임무 모드: {self.avs_to_implement_mode_dict}")

        # 표적융합결과 기반 자체 표적상태 및 추적 상태 업데이트
        for trg_id in self.mp_input.trg_fus_res_dict.keys():
            # 해당 표적이 최초 식별인 경우
            self.trg_to_state_dict.setdefault(trg_id, 0)
            # 실시간 표적 융합 중인 비행체 ID 리스트 업데이트
            tracker_list = [local_info.avs_id for local_info in self.mp_input.trg_fus_res_dict[trg_id].local_info_list]
            self.trg_to_tracker_dict[trg_id] = list(set(tracker_list))

    def update_trg_state(
        self,
        mp_selection_result_dict: Dict[
            int, Union[ReconnaissanceSelectionResult, TrackSelectionResult, AttackSelectionResult]
        ],
        state: int,
    ) -> None:
        # 정찰 임무가 할당되면 1로 변경
        # 추적에 성공하면 2로 변경
        # 타격 임무가 할당되면 3으로 변경
        # TODO: 타격이 완료되면 4로 변경
        for mp_selection_result in mp_selection_result_dict.values():
            self.trg_to_state_dict.update({mp_selection_result.target_id: state})

    def save_mp_selection_result_as_json(self, mission: str) -> None:
        # 정해진 위치에 임무계획 결과 저장
        with open(Path(self.result_dirs[mission] / f"timestamp_{get_timestamp()}.json"), "w") as fp:
            json.dump(self.mp_io_log, fp, indent=4, cls=DataToJsonEncoder)
        # 저장 후 성능지표 초기화
        self.mp_io_log["result"]["perf"] = {}
        self.mp_io_log["result"]["barchart"] = {}

    def save_prev_mp_selection_result(self) -> None:
        self.prev_mp_output_dict: SelectionResultType = deepcopy(self.mp_io_log["result"]["output"])

    def update_mp_selection_result_logger(
        self,
        mission: str,
        delta_t: float,
        mp_selection_result_dict: Dict[int, SelectionResultType],
        perf_dict: Dict[str, float],
    ) -> None:
        # 저장할 임무계획 결과 JSON 만들기
        self.mp_io_log["timestamp"] = get_timestamp()
        self.mp_io_log["result"]["mission"] = mission
        self.mp_io_log["result"]["algorithm"] = self.mission_to_algo_dict.get(mission)
        self.mp_io_log["result"]["output"] = mp_selection_result_dict
        self.mp_io_log["result"]["perf"].update({"delta_t": delta_t})
        self.mp_io_log["result"]["perf"].update(perf_dict)
        self.mp_io_log["result"]["barchart"].update(
            {"label": {"x": "x_label", "y": "y_label"}, "data": [{"x": "perf_3", "y": 0.3}]}
        )

        # 임무 계획에 따라 모드 최신화
        for avs_id, mp_selection_result in mp_selection_result_dict.items():
            self.avs_to_implement_mode_dict.update(
                {avs_id: MISSION_MODE_TO_NAME_DICT.get(mp_selection_result.smp_mode)}
            )

            # 정찰 및 타격 임무인 경우, 그룹 정보 업데이트
            if mission in ["Attack", "Reconn"]:
                self.avs_to_smp_group_id_dict.update({avs_id: mp_selection_result.smp_group_id})

        # 정찰 및 타격 임무인 경우, 그룹 정보 업데이트
        if mission in ["Attack", "Reconn"]:
            self.smp_group_id_to_avs_dict = {}
            for avs_id, smp_group_id in self.avs_to_smp_group_id_dict.items():
                if smp_group_id not in self.smp_group_id_to_avs_dict:
                    self.smp_group_id_to_avs_dict[smp_group_id] = set()
                self.smp_group_id_to_avs_dict[smp_group_id].add(avs_id)

        # 타격 임무를 위한 직전 임무계획 결과 보관
        self.save_prev_mp_selection_result()
