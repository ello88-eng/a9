from enum import IntEnum, auto

from config.sim_msg_config import SimulationStatus

# 감시패턴 생성 여부
GEN_SURV_PATH: bool = True

# 임무계획 알고리즘
ALGO_TO_NAME_DICT = {"Replan": "ADD", "Surv": "ADD4", "Reconn": "LIG", "Attack": "ADD"}

# 비행체별 자동임무 모드 테이블
MISSION_MODE_TO_NAME_DICT = {0: "이동", 1: "감시", 2: "정찰", 3: "추적", 4: "타격"}

# 알고리즘 환경설정
# 감시 선별 알고리즘
NUM_ROWS: int = 200
NUM_COLS: int = 200

# 감시경로 위치 온도
SURV_TEMPERATURE: int = 40

# 재계획 모드
REPLAN_MODE: str = "Manual"
# REPLAN_MODE: str = "Auto"

#
D_CELLS: int = 5

#
MIN_CLEARANCE: int = 2

#
DARP_THRESHOLD: int = 5

# 고도 기준값
ALT_REF: int = 0

# 감시경로 선회 반경
SURV_TURN_RADIUS: int = 300

# 촬영영역 고려 여부
USE_GRID_COVERAGE: bool = False

# FoV 길이
FOV_LEN: int = 400

# 정찰 선별 그룹 수
MAX_GROUPS_OF_5: int = None

# 촬영영역 다운샘플링 기준
GRID_COVERAGE_THRESHOLD: float = 0.75

# 이동 표적에 대한 부분 재계획 거리 기준
TRG_POS_CHANGE_THRESHOLD = 100

# 표적 추적 완료 기준
TRG_TRACK_THRESHOLD: int = 2

# 임무계획 입력 확인 여부
SHOW_INPUTS: bool = False

# 임무계획 출력 확인 여부
SHOW_OUTPUT: bool = False

# 오프라인 강화학습 데이터 확인 여부
SHOW_OFFLINE: bool = False

# GCS 표적정보 활용 여부
USE_MP_TRG: bool = False

# 이전 임무계획 출력 활용 여부
USE_PREV_MP_OUTPUT: bool = False

# 임무계획 입력 모의
STATE_FP = ""

# 임무 재계획 판단 주기
REPLAN_INTERVAL = 5.0
TRAIN_INTERVAL = 0.5

# 임무유형 설정
MISSION_LIST = ["Attack", "Reconn", "Surv"]


# 체계 임무 상태
class SystemMissionState(IntEnum):
    START = 1
    FINISH = 2


SYS_STATUS_TRANSITION = {
    (SimulationStatus.INIT_COMPLETED.value, SystemMissionState.START.value): SimulationStatus.START_SIM.value,
    (SimulationStatus.START_SIM.value, SystemMissionState.FINISH.value): SimulationStatus.STOP_SIM.value,
}
