import binascii
import pickle
import struct
import time
import zlib
from typing import Tuple

import numpy as np

from commu.sim.receiver.data.es2mps import SimulationObjInfo
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import AvsInfoForRl, TargetInfoForRl
from config.gui_config import SYS_TO_GUI_MSG_NO
from config.resolution_config import RES
from config.sim_msg_config import SIM_HEADER_FMT, SimulationStatus
from config.sys_msg_config import CBIT_INTERVAL
from utils.logger import logger
from utils.operation import Operation
from utils.server import sock
from utils.timer import get_timestamp


def send_sim_status_to_gui(operation: Operation, src: int, scenario_path: str, address: Tuple[str, int]):
    while True:
        if (
            operation.status in [SimulationStatus.START_INIT.value, SimulationStatus.EXIT_SIM.value]
            and operation.is_started == True
        ):
            header = struct.pack(SIM_HEADER_FMT, 0xFE, 0x00, 0x00, src, 0x00, 0x04, get_timestamp())
            encoded_scenario_path = scenario_path.encode()
            payload = struct.pack("> H H", operation.status, len(encoded_scenario_path))
            payload += encoded_scenario_path
            checksum = zlib.crc32(encoded_scenario_path).to_bytes(length=4, byteorder="big")
            # 데이터 전송
            sock.sendto(header + payload + checksum, address)

            if operation.status == SimulationStatus.EXIT_SIM.value:
                break

            # 1초 주기 전송
            time.sleep(CBIT_INTERVAL)


def send_sim_obj_info(data: SimulationObjInfo, show_offline: bool, address: Tuple[str, int]):
    """20250609

    Args:
        data (SimulationObjInfo): _description_
        show_offline (bool): _description_
        address (Tuple[str, int]): _description_
    """
    # 데이터 포맷 정의
    plf_fmt = "> h iiH Hhh HH Ii I iiH hhhhhhhh"
    trg_fmt = "> H B iiH"
    # 페이로드 길이 정의
    plf_len = struct.calcsize(plf_fmt)
    trg_len = struct.calcsize(trg_fmt)
    # 모의 객체 정보 출력
    if show_offline:
        logger.info(data)
    # 비행체 상태 정보
    for (
        id,
        lat,
        lon,
        alt,
        yaw,
        pitch,
        roll,
        fov_horz,
        fov_vert,
        rel_azim_ang,
        rel_elev_ang,
        slant_range,
        fr_lat,
        fr_lon,
        fr_elev,
        fp_lat_1,
        fp_lon_1,
        fp_lat_2,
        fp_lon_2,
        fp_lat_3,
        fp_lon_3,
        fp_lat_4,
        fp_lon_4,
    ) in zip(
        data.avs_ids,
        data.plf_lats,
        data.plf_lons,
        data.plf_alts,
        data.plf_heading,
        data.plf_pitchs,
        data.plf_rolls,
        data.fov_horzs,
        data.fov_verts,
        data.rel_azim_angs,
        data.rel_elev_angs,
        data.slant_ranges,
        data.fr_center_lats,
        data.fr_center_lons,
        data.fr_center_elevs,
        data.fp_lats_1,
        data.fp_lons_1,
        data.fp_lats_2,
        data.fp_lons_2,
        data.fp_lats_3,
        data.fp_lons_3,
        data.fp_lats_4,
        data.fp_lons_4,
    ):
        #
        if id != -1:
            header = struct.pack(
                SIM_HEADER_FMT,
                0xFE,
                plf_len,
                0x00,
                0x40,
                0x80,
                0x01,
                data.timestamp,
            )
            payload = struct.pack(">B", 0x00)
            payload += struct.pack(
                plf_fmt,
                id,
                int(lat / RES.lat),
                int(lon / RES.lon),
                int((alt + 900) / RES.alt),
                int(yaw / RES.heading),
                int(pitch / RES.pitch),
                int(roll / RES.roll),
                int(fov_horz / RES.fov_horz),
                int(fov_vert / RES.fov_horz),
                int(rel_azim_ang / RES.gimbal),
                int(rel_elev_ang / RES.gimbal),
                int(slant_range),
                int(fr_lat / RES.lat),
                int(fr_lon / RES.lon),
                int((fr_elev + 900) / RES.alt),
                int(fp_lat_1 / RES.corner),
                int(fp_lon_1 / RES.corner),
                int(fp_lat_2 / RES.corner),
                int(fp_lon_2 / RES.corner),
                int(fp_lat_3 / RES.corner),
                int(fp_lon_3 / RES.corner),
                int(fp_lat_4 / RES.corner),
                int(fp_lon_4 / RES.corner),
            )
            checksum = binascii.crc32(header + payload).to_bytes(length=4, byteorder="big")
            sock.sendto(header + payload + checksum, address)

    for id, type, lat, lon, alt in zip(
        data.trg_ids,
        data.trg_types,
        data.trg_lats,
        data.trg_lons,
        data.trg_alts,
    ):
        if id != -1:
            # 표적 정보
            header = struct.pack(
                ">B H B BBB Q",
                0xFE,
                trg_len,
                0x00,
                0x40,
                0x80,
                0x01,
                data.timestamp,
            )
            payload = struct.pack(">B", 0x01)
            payload += struct.pack(
                trg_fmt,
                id,
                type,
                int(lat / RES.lat),
                int(lon / RES.lon),
                int((alt + 900) / RES.alt),
            )
            checksum = binascii.crc32(header + payload).to_bytes(length=4, byteorder="big")
            sock.sendto(header + payload + checksum, address)


def send_exit(address: Tuple[str, int]) -> None:
    header = struct.pack(SIM_HEADER_FMT, 0xFE, 0x01, 0x00, 0x40, 0x01, 0x02, get_timestamp())
    payload = struct.pack(">B", 0x01)
    checksum = struct.pack(">I", binascii.crc32(payload))
    # 데이터 전송
    sock.sendto(header + payload + checksum, address)
    # 에피소드 종료 메시지
    logger.info("Episode terminated.")


def send_rl_state(inputs: MissionPlanInput, addr: Tuple[str, int]) -> None:
    data = {
        "state": {
            "scenario": np.random.randint(low=0, high=40000, size=(1, 1000)),
            "agent": {
                id: AvsInfoForRl(
                    pos=(
                        inputs.avs_info_dict[id].position
                        if id in inputs.avs_info_dict.keys()
                        else np.array([0.0, 0.0, 0.0])
                    ),
                    attitude=(
                        inputs.avs_info_dict[id].attitude
                        if id in inputs.avs_info_dict.keys()
                        else np.array([0.0, 0.0, 0.0])
                    ),
                    duration=(inputs.avs_info_dict[id].soc if id in inputs.avs_info_dict.keys() else 0.0),
                )
                for id in range(50)
            },
            "target": {
                id: TargetInfoForRl(
                    pos=(
                        inputs.trg_fus_res_dict[id].position
                        if id in inputs.trg_fus_res_dict.keys()
                        else np.array([0.0, 0.0, 0.0])
                    ),
                    value=0.0,
                )
                for id in range(20)
            },
        }
    }
    serialized_data = pickle.dumps(data)
    sock.sendto(serialized_data, addr)
    logger.info(f"Send RL state - IP: {addr}")


class GuiHeader:

    def __init__(self, addr: Tuple[str, int], header_fmt: str = SIM_HEADER_FMT) -> None:
        self.header_fmt = header_fmt
        self.addr = addr

    def _get_header(self, src: int, msg_id: int) -> bytes:
        return struct.pack(self.header_fmt, 0xFE, 0x00, 0x00, src, 0x00, SYS_TO_GUI_MSG_NO[msg_id], get_timestamp())

    def send(self, src: int, msg_id: int, payload: bytes) -> None:
        header = self._get_header(src=src, msg_id=msg_id)
        checksum = zlib.crc32(header + payload).to_bytes(length=4, byteorder="big")
        sock.sendto(header + payload + checksum, self.addr)
