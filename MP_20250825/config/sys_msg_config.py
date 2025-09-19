from enum import IntEnum, unique

from config.mp_config import NUM_COLS, NUM_ROWS


@unique
class SystemMessageId(IntEnum):
    REPLAN_REQUEST = 0x7B03
    MISSION_PLAN_CMD = 0x7B04
    BOUNDARY = 0x7B05
    AVS_INFO = 0x7B06
    MISSION_PLAN_TARGET = 0x7B07
    SW_VERSION_REQ = 0x7B09
    CIRCULAR_AREA = 0x7B0A
    POLYGON_AREA = 0x7B0B
    MISSION_INITIAL_INFO = 0x7B0C
    MISSION_STATE = 0x7B0D
    MISSION_PLAN_RESULT_ACK = 0x7B0E
    TARGET_FUSION_RESULT = 0x7A01
    SURV_SELECTION_RESULT = 0x7A02
    RECONN_SELECTION_RESULT = 0x7A05
    ATTACK_SELECTION_RESULT = 0x7A06
    MISSION_PLAN_RESULT_INFO = 0x7A07
    SMP_STATUS = 0x7C03  # To 융합모듈 (IDD)
    SW_VERSION_RES = 0x7C04  # To 융합모듈 (IDD)
    GRID_COVERAGE_DATA = 0x7C05  # From 융합모듈 (IDD)


# 처리 함수 이름 매핑
UPDATER_FN_DICT = {
    SystemMessageId.REPLAN_REQUEST.value: "update_replan_request",
    SystemMessageId.MISSION_PLAN_CMD.value: "update_mission_plan_command",
    SystemMessageId.BOUNDARY.value: "update_boundary",
    SystemMessageId.AVS_INFO.value: "update_avs_info",
    SystemMessageId.MISSION_PLAN_TARGET.value: "update_mission_plan_target",
    SystemMessageId.CIRCULAR_AREA.value: "update_circular_mission_area",
    SystemMessageId.POLYGON_AREA.value: "update_polygon_mission_area",
    SystemMessageId.MISSION_INITIAL_INFO.value: "update_mission_initial_info",
    SystemMessageId.MISSION_STATE.value: "update_mission_state",
    SystemMessageId.TARGET_FUSION_RESULT.value: "update_target_fusion_result",
    SystemMessageId.GRID_COVERAGE_DATA.value: "update_grid_coverage_data",
}


# 메시지 헤더 포맷 정의 (ESICD v1.6)
SYS_HEADER_FMT = "> BHH"

# 메시지 전송 주기
CBIT_INTERVAL: int = 1

# 임무계획 결과 전송 인터벌 [초]
MP_RESULT_SEND_INTERVAL = 0.1

# ESICD v1.6 기준 메시지 정의
PAYLOAD_TO_FMT = {
    "heartbeat": "> H",
    "replan_request": "> H H",
    "mission_plan_cmd": "> H H",
    "boundary": "> ii ii ii ii",
    "avs_info": "> H HH HH HHB iiH hhH I HH BHB",
    "mission_plan_target": "> H ii",
    "sw_version_req": "> B",
    "circular_area": "> iiH",
    "polygon_area": {"corner_count": "> H", "vertex": "ii"},
    "mission_initial_info": "> HH H h B B 100s HB B B",
    "mission_state": "> B",
    "mission_plan_result_ack": "> B",
    "target_fusion_result": {
        "global_target_count": "> B",
        "timestamp": "> Q",
        "is_simulation": "> B",
        "global_target_info": "> H iiH B B",
        "local_target_count": "> B",
        "local_target_info": "> H H iiH B H",
    },
    "surv_selection_result": {"basic_info": "> H H B H", "waypoint_count": "> H", "waypoint": "iiH"},
    "reconn_selection_result": "> H HH B ii HH iiH H",
    "attack_selection_result": "> H HH B ii HH iiH H HH",
    "mission_plan_result_info": "> H H H",
    "smp_status": "> B",
    "sw_version_res": "> H BB",
    "grid_coverage_data": ">" + f"{round(NUM_ROWS * NUM_COLS / 8)}B",
}
