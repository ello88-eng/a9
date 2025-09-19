import struct
import time
import zlib
from typing import Optional, Tuple

import numpy as np

from commu.sys.sender.data.sys_out import (
    AttackSelectionResult,
    MissionPlanResultInfo,
    ReconnaissanceSelectionResult,
    SurveillanceSelectionResult,
)
from config.main_config import MainArgs
from config.mp_config import MISSION_MODE_TO_NAME_DICT
from config.resolution_config import CRC_16_TABLE, RES
from config.sim_msg_config import SIM_HEADER_FMT
from config.sw_config import SW_VERSION
from config.sys_msg_config import (
    CBIT_INTERVAL,
    PAYLOAD_TO_FMT,
    SYS_HEADER_FMT,
    SystemMessageId,
)
from utils.coordinates import true_round
from utils.logger import logger
from utils.server import sock
from utils.timer import get_timestamp


def compute_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        pos = ((crc >> 8) ^ byte) & 0xFF
        crc = ((crc << 8) ^ CRC_16_TABLE[pos]) & 0xFFFF

    return crc


def send_output_to_gui(payload: bytes, src: int, address: Tuple[str, int]):
    # TODO
    header = struct.pack(SIM_HEADER_FMT, 0xFE, 0x00, 0x00, src, 0x00, 0x03, get_timestamp())
    checksum = zlib.crc32(header + payload).to_bytes(length=4, byteorder="big")
    sock.sendto(header + payload + checksum, address)
    logger.info(f"GUI: {address}으로 임무계획 전송")


def send_smp_status(address: Tuple[str, int]):
    heartbeat = 1
    while True:
        header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.SMP_STATUS, 0x01)
        body = struct.pack(PAYLOAD_TO_FMT["smp_status"], 0x01)
        body += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)
        checksum = true_round(compute_crc16(header + body)).to_bytes(length=2, byteorder="big")
        sock.sendto(header + body + checksum, address)
        time.sleep(CBIT_INTERVAL)
        heartbeat = (heartbeat + 1) % 65535


def send_sw_version(address: Tuple[str, int], heartbeat: int):
    header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.SW_VERSION_RES, 0x01)
    body = struct.pack(PAYLOAD_TO_FMT["sw_version_res"], 0x00, SW_VERSION[0], SW_VERSION[1] * 10 + SW_VERSION[2])
    body += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)
    checksum = true_round(compute_crc16(header + body)).to_bytes(length=2, byteorder="big")
    sock.sendto(header + body + checksum, address)
    logger.info(f"{address}으로 SW 버전 v{SW_VERSION[0]}.{SW_VERSION[1] * 10 + SW_VERSION[2]} 전송")


def send_mp_result_info(args: MainArgs, mp_result_info: MissionPlanResultInfo, heartbeat: int = 0) -> None:
    body = struct.pack(
        PAYLOAD_TO_FMT["mission_plan_result_info"],
        mp_result_info.smp_group_id,
        mp_result_info.mission_info,
        mp_result_info.avs_count,
    )
    body += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)
    header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.MISSION_PLAN_RESULT_INFO, len(body))
    checksum = true_round(compute_crc16(header + body)).to_bytes(length=2, byteorder="big")
    sock.sendto(header + body + checksum, (args.addr_fm_sys_ip, args.addr_mps_port))
    logger.info(
        f"임무계획 결과 정보 전송: "
        f"SMP 그룹: G{mp_result_info.smp_group_id} / "
        f"임무 유형: {MISSION_MODE_TO_NAME_DICT[mp_result_info.mission_info]} / "
        f"비행체 대수: {mp_result_info.avs_count}"
    )


def send_surv_selection_result(
    args: MainArgs,
    result: SurveillanceSelectionResult,
    show_output: Optional[bool] = False,
    heartbeat: int = 0,
    src: int = 1,
) -> None:
    # 임무계획 출력을 보고싶은 경우
    if show_output:
        logger.info(result)

    #
    payload = struct.pack(
        PAYLOAD_TO_FMT["surv_selection_result"]["basic_info"],
        result.avs_id,
        result.system_group_id,
        result.smp_mode,
        true_round(result.speed),
    )

    #
    payload += struct.pack(PAYLOAD_TO_FMT["surv_selection_result"]["waypoint_count"], result.waypoint_count)

    #
    wps_fmt = ">" + PAYLOAD_TO_FMT["surv_selection_result"]["waypoint"] * result.waypoint_count
    result.waypoints[:, -1] += 900
    wps = result.waypoints / np.tile([RES.lat, RES.lon, RES.alt], (result.waypoints.shape[0], 1))
    payload += struct.pack(wps_fmt, *[true_round(p) for p in wps.flat])

    #
    payload += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)

    #
    header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.SURV_SELECTION_RESULT, len(payload))
    checksum = true_round(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")

    # 모의장치/실장비: 융합 모듈(지상체/지상체 모의)로 임무계획 결과 전송
    sock.sendto(header + payload + checksum, (args.addr_fm_sys_ip, args.addr_mps_port))

    # 임무계획 분석 도구로 임무계획 결과 전송
    # TODO send_output_to_gui(payload=payload, src=src, address=(args.addr_gui_ip, args.addr_gui_port))
    logger.info(f"감시선별 결과: 비행체 ID: A{result.avs_id} / 감시경로 항로점 개수: {result.waypoint_count}")


def send_reconn_selection_result(
    args: MainArgs,
    result: ReconnaissanceSelectionResult,
    show_output: Optional[bool] = False,
    heartbeat: int = 0,
    src: int = 1,
) -> None:
    # 임무계획 출력을 보고싶은 경우
    if show_output:
        logger.info(result)

    # 선회 중심점 Resolution 적용
    coordinate = result.coordinate / np.tile([RES.lat, RES.lon], (result.coordinate.shape[0], 1))

    # 표적 좌표 Resolution 적용
    result.target_position[:, -1] += 900
    trg_position = result.target_position / np.tile([RES.lat, RES.lon, RES.alt], (result.target_position.shape[0], 1))

    # 페이로드 생성
    payload = struct.pack(
        PAYLOAD_TO_FMT["reconn_selection_result"],
        result.avs_id,
        result.system_group_id,
        result.smp_group_id,
        result.smp_mode,
        *[true_round(coord) for coord in coordinate.flat],
        result.turning_radius,
        result.turning_direction,
        *[true_round(trg_pos) for trg_pos in trg_position.flat],
        result.target_id,
    )
    payload += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)

    # 헤더 생성
    header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.RECONN_SELECTION_RESULT, len(payload))
    checksum = true_round(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")

    # 모의장치/실장비: 융합 모듈(지상체/지상체 모의)로 임무계획 결과 전송
    sock.sendto(header + payload + checksum, (args.addr_fm_sys_ip, args.addr_mps_port))

    # 임무계획 분석 도구로 임무계획 결과 전송
    # TODO send_output_to_gui(payload=payload, src=src, address=(args.addr_gui_ip, args.addr_gui_port))
    logger.info(f"정찰선별 결과: 비행체 ID: A{result.avs_id} / 표적 ID: T{result.target_id}")


def send_attack_selection_result(
    args: MainArgs,
    result: AttackSelectionResult,
    show_output: Optional[bool] = False,
    heartbeat: int = 0,
    src: int = 1,
) -> None:
    # 임무계획 출력을 보고싶은 경우
    if show_output:
        logger.info(result)

    # 선회 중심점 Resolution 적용
    coordinate = result.coordinate / np.tile([RES.lat, RES.lon], (result.coordinate.shape[0], 1))

    # 표적 좌표 Resolution 적용
    result.target_position[:, -1] += 900
    trg_position = result.target_position / np.tile([RES.lat, RES.lon, RES.alt], (result.target_position.shape[0], 1))

    # 페이로드 생성
    payload = struct.pack(
        PAYLOAD_TO_FMT["attack_selection_result"],
        result.avs_id,
        result.system_group_id,
        result.smp_group_id,
        result.smp_mode,
        *[true_round(coord) for coord in coordinate.flat],
        result.turning_radius,
        result.turning_direction,
        *[true_round(trg_pos) for trg_pos in trg_position.flat],
        result.target_id,
        result.attack_priority,
        result.bd_assignment_flag,
    )
    payload += struct.pack(PAYLOAD_TO_FMT["heartbeat"], heartbeat)

    # 헤더 생성
    header = struct.pack(SYS_HEADER_FMT, 0x00, SystemMessageId.RECONN_SELECTION_RESULT, len(payload))
    checksum = true_round(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")

    # 모의장치/실장비: 융합 모듈(지상체/지상체 모의)로 임무계획 결과 전송
    sock.sendto(header + payload + checksum, (args.addr_fm_sys_ip, args.addr_mps_port))

    # 임무계획 분석 도구로 임무계획 결과 전송
    # TODO end_output_to_gui(payload=payload, src=src, address=(args.addr_gui_ip, args.addr_gui_port))
    logger.info(
        f"타격선별 결과: "
        f"비행체 ID: A{result.avs_id} / "
        f"표적 ID: T{result.target_id} / "
        f"타격 순서: {result.attack_priority} / "
        f"BD 할당여부: {bool(result.bd_assignment_flag)}"
    )
