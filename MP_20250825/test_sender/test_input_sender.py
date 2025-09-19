import pathlib
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from create_data import choice_indices, create_loc_infos, create_positions, create_trg
from transformers.hf_argparser import HfArgumentParser

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from mission_planning.configs.config import RES
from utils.crc import compute_crc16
from utils.logger import logger
from utils.timer import get_timestamp

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@dataclass
class SenderArgs:
    ip: str = "192.168.10.202"
    port: int = 10000
    mission: str = "reconn"
    num_avs: int = 15
    num_trg: int = 3


heartbeats = {
    "replan_req": 0,
    "cmd": 0,
    "scenario": 0,
    "info": 0,
    "gcs_trg": 0,
    "fus_outs": 0,
}


def send_req(s_args: SenderArgs) -> None:
    # 재계획 요청 정의
    req = 0x00
    # header 정의
    payload = struct.pack("> HH", req, heartbeats["replan_req"])
    header = struct.pack(">BHH", 0x00, 0x7B03, len(payload))
    checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
    while True:
        # 데이터 전송
        sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
        # 10Hz 주기
        time.sleep(0.1)
        # 하트비트 업데이트
        heartbeats["replan_req"] = (heartbeats["replan_req"] + 1) % 65535


def send_cmd(s_args: SenderArgs) -> None:
    # 운용자 입력 정의
    replan_request_flag = 0
    while True:
        # 비행체 수만큼 반복
        for avs in range(1, s_args.num_avs + 1):
            payload = struct.pack(
                "> H H H",
                avs,
                replan_request_flag,
                heartbeats["cmd"],
            )
            # header 정의
            header = struct.pack(">BHH", 0x00, 0x7B04, len(payload))
            checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
            # 데이터 전송
            sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
        # 10Hz 주기
        time.sleep(0.1)
        # 하트비트 업데이트
        heartbeats["cmd"] = (heartbeats["cmd"] + 1) % 65535


def send_scenario(s_args: SenderArgs) -> None:
    # 시나리오 정의
    boundary = np.array(
        [
            int(38.045218 / RES.lat),
            int(126.955085 / RES.lon),
            int(38.045218 / RES.lat),
            int(127.044915 / RES.lon),
            int(37.954782 / RES.lat),
            int(127.044915 / RES.lon),
            int(37.954782 / RES.lat),
            int(126.955085 / RES.lon),
        ]
    )
    area = np.array(
        [
            int(38.045218 / RES.lat),
            int(126.955085 / RES.lon),
            int(38.045218 / RES.lat),
            int(127.044915 / RES.lon),
            int(37.954782 / RES.lat),
            int(127.044915 / RES.lon),
            int(37.954782 / RES.lat),
            int(126.955085 / RES.lon),
            int(37.954782 / RES.lat),
            int(126.955085 / RES.lon),
            int(37.954782 / RES.lat),
            int(126.955085 / RES.lon),
        ]
    )
    while True:
        # 바디 정의
        payload = struct.pack("> iiiiiiii iiiiiiiiiiii H", *boundary, *area, heartbeats["scenario"])
        # 헤더 정의
        header = struct.pack(">BHH", 0x00, 0x7B05, len(payload))
        checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
        # 데이터 전송
        sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
        # 10Hz 주기
        time.sleep(0.1)
        # 하트비트 업데이트
        heartbeats["scenario"] = (heartbeats["scenario"] + 1) % 65535


def send_infos(s_args: SenderArgs) -> None:
    # 페이로드 정의
    is_arrival = 1
    group_id = 0
    # 비행체 상태 정의
    positions = create_positions(s_args.num_avs)
    modes = np.full(s_args.num_avs, 0)
    if s_args.mission == "reconn":
        modes = np.full(s_args.num_avs, 1)
    elif s_args.mission == "attack":
        modes[indices - 1] = 2
    r, p, y = int(0.0), int(0.0), int(0.0)
    xx, yy, zz = int(0.0), int(0.0), int(0.0)
    min_speed, max_speed = int(20.0), int(30.0)
    sensing_range = 1000
    fov_vert, fov_horz = int(0.0), int(0.0)
    duration = 1000
    return_pos = np.array([0.0, 0.0, 900.0]) / np.tile([RES.lat, RES.lon, RES.alt], (1,))
    while True:
        for avs_id in range(1, s_args.num_avs + 1):
            pos = positions[avs_id - 1] / np.tile([RES.lat, RES.lon, RES.alt], (1,))
            payload = struct.pack(
                ">HHHHiiHhhHHHHhhIHHQiiHH",
                avs_id,
                group_id,
                is_arrival,
                modes[avs_id - 1],
                *[int(p) for p in pos],
                r,
                p,
                y,
                xx,
                yy,
                zz,
                min_speed,
                max_speed,
                sensing_range,
                fov_vert,
                fov_horz,
                duration,
                *[int(rp) for rp in return_pos],
                heartbeats["info"],
            )
            # header 정의
            header = struct.pack(">BHH", 0x00, 0x7B06, len(payload))
            # 체크섬 정의
            checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
            # 데이터 전송
            sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
            # 하트비트 업데이트
            heartbeats["info"] = (heartbeats["info"] + 1) % 65535
        # 10Hz 주기
        time.sleep(0.1)


def send_gcs_trg(s_args: SenderArgs) -> None:
    # 페이로드 정의
    wp_offset_lat = int(10.0 / RES.lat)
    wp_offset_lon = int(10.0 / RES.lon)
    while True:
        for trg_id in range(1, s_args.num_trg + 1):
            payload = struct.pack(
                "> H ii H",
                trg_id,
                wp_offset_lat,
                wp_offset_lon,
                heartbeats["gcs_trg"],
            )
            # 헤더 정의
            header = struct.pack(">BHH", 0x00, 0x7B07, len(payload))
            # 체크섬 정의
            checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
            # 데이터 전송
            sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
            # 하트비트 업데이트
            heartbeats["gcs_trg"] = (heartbeats["gcs_trg"] + 1) % 65535
        # 10Hz 주기
        time.sleep(0.1)


def send_fusion(s_args: SenderArgs) -> None:
    # 페이로드 정의
    positions = create_positions(s_args.num_trg)
    while True:
        if s_args.mission != "surv":
            payload = struct.pack("> B", s_args.num_trg)
            payload += struct.pack("> Q", get_timestamp())
            for trg in range(1, s_args.num_trg + 1):
                pos = positions[trg - 1] / np.tile([RES.lat, RES.lon, RES.alt], (1,))
                payload += struct.pack("> H iiH B B", trg, *[int(p) for p in pos], 0, 5)
                loc_infos = create_loc_infos(indices[trg - 1])
                payload += struct.pack(">" + "H H iiH" * 5, *[int(l) for l in loc_infos])
            payload += struct.pack("> H", heartbeats["fus_outs"])
            # 헤더 정의
            header = struct.pack("> BHH", 0x00, 0x7A01, len(payload))
            # 체크섬 정의
            checksum = int(compute_crc16(header + payload)).to_bytes(length=2, byteorder="big")
            # 데이터 전송
            sock.sendto(header + payload + checksum, (s_args.ip, 6 + s_args.port))
            # 하트비트 업데이트
            heartbeats["fus_outs"] = (heartbeats["fus_outs"] + 1) % 65535
        # 10Hz 주기
        time.sleep(0.1)


def main(s_args: SenderArgs):
    # 스레드 생성
    req_thread = threading.Thread(target=send_req, args=(s_args,), daemon=True)
    cmd_thread = threading.Thread(target=send_cmd, args=(s_args,), daemon=True)
    scenario_thread = threading.Thread(target=send_scenario, args=(s_args,), daemon=True)
    info_thread = threading.Thread(target=send_infos, args=(s_args,), daemon=True)
    gcs_trg_thread = threading.Thread(target=send_gcs_trg, args=(s_args,), daemon=True)
    fus_thread = threading.Thread(target=send_fusion, args=(s_args,), daemon=True)

    # 스레드 시작
    req_thread.start()
    cmd_thread.start()
    scenario_thread.start()
    gcs_trg_thread.start()
    global indices
    indices = choice_indices(s_args.num_avs, s_args.num_trg)
    info_thread.start()
    fus_thread.start()
    req_thread.join()
    cmd_thread.join()
    gcs_trg_thread.join()
    info_thread.join()
    fus_thread.join()


if __name__ == "__main__":
    s_args = HfArgumentParser(SenderArgs).parse_args_into_dataclasses()[0]
    logger.info(s_args)
    main(s_args)
