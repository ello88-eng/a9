import struct
from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np

from commu.sys.receiver.data.fs2mps import TargetFusionResult, TargetLocalInfo
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
from config.mp_config import NUM_COLS, NUM_ROWS
from config.resolution_config import CRC_16_TABLE, RES
from config.sys_msg_config import PAYLOAD_TO_FMT, SYS_HEADER_FMT, SystemMessageId
from utils.coordinates import sort_coordinates_as_clockwise


class SystemMessageHeaderParser:

    def __init__(self, header_fmt: str = SYS_HEADER_FMT):
        self.header_fmt = header_fmt
        self.header_size = struct.calcsize(self.header_fmt)

    def _compute_crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            pos = ((crc >> 8) ^ byte) & 0xFF
            crc = ((crc << 8) ^ CRC_16_TABLE[pos]) & 0xFFFF

        return crc

    def validate(self, data: bytes) -> bool:
        payload = data[:-2]
        checksum = data[-2:]
        if checksum == int(self._compute_crc16(payload)).to_bytes(length=2, byteorder="big"):
            return True
        else:
            return False

    def parse(self, data: bytes) -> Tuple[str, bytes]:
        _, msg_id, _ = struct.unpack(self.header_fmt, data[: self.header_size])
        payload = data[self.header_size : -4]

        return msg_id, payload


class ReplanRequestParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["replan_request"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> ReplanRequest:
        avs_id, need_request = struct.unpack(self.fmt, payload)

        return {avs_id: ReplanRequest(avs_id=avs_id, need_request=need_request)}


class MissionPlanCommandParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["mission_plan_cmd"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> Dict[int, MissionPlanCommand]:
        reconn_request, target_id = struct.unpack(self.fmt, payload)

        return {target_id: MissionPlanCommand(reconn_request=reconn_request, target_id=target_id)}


class BoundaryParser:

    def __init__(self, fmts: List[str] = PAYLOAD_TO_FMT["boundary"]):
        self.fmt = fmts

    def parse(self, payload: bytes) -> Boundary:
        vertices = struct.unpack(self.fmt, payload[: struct.calcsize(self.fmt)])
        vertices = np.array(vertices).reshape(4, 2) * np.tile([RES.lat, RES.lon], (4, 1))

        return Boundary(np.array(sort_coordinates_as_clockwise(vertices)))


class AvsInfoParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["avs_info"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> Dict[int, AvsInfo]:
        (
            avs_id,
            system_group_id,
            smp_group_id,
            is_avs_arrival,
            is_system_group_arrival,
            system_mode,
            flight_control_mode,
            implement_mode,
            lat,
            lon,
            alt,
            roll,
            pitch,
            yaw,
            sensing_range,
            fov_vert,
            fov_horz,
            soc,
            voltage,
            hop_count,
        ) = struct.unpack(self.fmt, payload)
        avs_info = AvsInfo(
            avs_id=avs_id,
            system_group_id=system_group_id,
            smp_group_id=smp_group_id,
            is_avs_arrival=is_avs_arrival,
            is_system_group_arrival=is_system_group_arrival,
            system_mode=system_mode,
            flight_control_mode=flight_control_mode,
            implement_mode=implement_mode,
            position=np.array([lat * RES.lat, lon * RES.lon, alt * RES.alt - 900]),
            attitude=np.array([roll * RES.roll, pitch * RES.pitch, yaw * RES.heading]),
            sensing_range=sensing_range,
            fov_vert=fov_vert * RES.fov_vert,
            fov_horz=fov_horz * RES.fov_horz,
            soc=soc,
            voltage=voltage * RES.voltage,
            hop_count=hop_count,
        )

        return {avs_id: avs_info}


class MissionPlanTargetParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["mission_plan_target"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> Dict[int, MissionPlanTarget]:
        trg_id, wp_offset_lat, wp_offset_lon = struct.unpack(self.fmt, payload)
        gcs_trg = MissionPlanTarget(
            target_id=trg_id, waypoint_offset_lat=wp_offset_lat * RES.lat, waypoint_offset_lon=wp_offset_lon * RES.lon
        )

        return {trg_id: gcs_trg}


class CircularAreaParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["circular_area"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> CircularMissionArea:
        center_lat, center_lon, radius = struct.unpack(self.fmt, payload)

        return CircularMissionArea(center_lat=center_lat * RES.lat, center_lon=center_lon * RES.lon, radius=radius)


class PolygonAreaParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["polygon_area"]):
        self.corner_count_fmt = fmt["corner_count"]
        self.vertex_fmt = fmt["vertex"]

    def parse(self, payload: bytes) -> PolygonMissionArea:
        corner_count = struct.unpack(self.corner_count_fmt, payload[: struct.calcsize(self.corner_count_fmt)])[0]
        vertices = struct.unpack(
            "> " + corner_count * self.vertex_fmt,
            payload[
                struct.calcsize(self.corner_count_fmt) : struct.calcsize(self.corner_count_fmt)
                + struct.calcsize("> " + corner_count * self.vertex_fmt)
            ],
        )
        vertices = np.array(vertices, dtype=float).reshape(corner_count, -1)
        vertices *= np.tile([RES.lat, RES.lon], (vertices.shape[0], 1))

        return PolygonMissionArea(vertices)


class MissionInitialInfoParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["mission_initial_info"]) -> None:
        self.fmt = fmt

    def parse(self, payload: bytes) -> MissionInitialInfo:
        (
            surv_alt,
            reconn_alt,
            min_voltage,
            speed,
            arrival_param,
            surv_path_gen_method,
            mp_file_path,
            loiter_radius,
            loiter_direction,
            is_target_fusion_result_simulation,
            surv_turn_type,
        ) = struct.unpack(self.fmt, payload)

        return MissionInitialInfo(
            surv_alt=surv_alt * RES.init_alt,
            reconn_alt=reconn_alt * RES.init_alt,
            min_voltage=min_voltage * RES.voltage,
            speed=speed * RES.speed,
            arrival_param=arrival_param,
            surv_path_gen_method=surv_path_gen_method,
            mp_file_path=(
                "".join(mp_file_path.decode()).strip("\x00").split("\\")[-1] if type(mp_file_path) == bytes else ""
            ),
            loiter_radius=loiter_radius,
            loiter_direction=loiter_direction,
            is_simulation=is_target_fusion_result_simulation,
            surv_turn_type=surv_turn_type,
        )


class MissionStateParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["mission_state"]) -> None:
        self.fmt = fmt

    def parse(self, payload: bytes) -> MissionState:
        state = struct.unpack(self.fmt, payload)[0]

        return MissionState(state)


class TargetFusionResultParser:

    def __init__(self, fmts: Dict[str, str] = PAYLOAD_TO_FMT["target_fusion_result"]) -> None:
        self.glb_trg_cnt_fmt = fmts["global_target_count"]
        self.ts_fmt = fmts["timestamp"]
        self.is_sim = fmts["is_simulation"]
        self.glb_trg_info_fmt = fmts["global_target_info"]
        self.loc_trg_cnt_fmt = fmts["local_target_count"]
        self.loc_trg_info_fmt = fmts["local_target_info"]

    def parse(self, payload: bytes) -> Dict[int, TargetFusionResult]:
        glb_trg_cnt_len = struct.calcsize(self.glb_trg_cnt_fmt)
        ts_len = struct.calcsize(self.ts_fmt)
        is_sim_len = struct.calcsize(self.is_sim)
        glb_trg_info_len = struct.calcsize(self.glb_trg_info_fmt)
        loc_trg_cnt_len = struct.calcsize(self.loc_trg_cnt_fmt)
        loc_trg_info_len = struct.calcsize(self.loc_trg_info_fmt)

        trg_fus_res: List[TargetFusionResult] = []

        cursor = 0
        glb_trg_cnt = struct.unpack(self.glb_trg_cnt_fmt, payload[cursor : cursor + glb_trg_cnt_len])[0]
        cursor += glb_trg_cnt_len

        _ = struct.unpack(self.ts_fmt, payload[cursor : cursor + ts_len])[0]
        cursor += ts_len
        _ = struct.unpack(self.is_sim, payload[cursor : cursor + is_sim_len])[0]
        cursor += is_sim_len

        for _ in range(glb_trg_cnt):
            glb_trg_info = struct.unpack(self.glb_trg_info_fmt, payload[cursor : cursor + glb_trg_info_len])
            cursor += glb_trg_info_len
            glb_fus_res = TargetFusionResult(
                target_id=glb_trg_info[0],
                position=np.array(
                    [glb_trg_info[1] * RES.lat, glb_trg_info[2] * RES.lon, glb_trg_info[3] * RES.alt - 900]
                ),
                target_class=glb_trg_info[4],
                target_state=glb_trg_info[5],
                local_info_list=[],
            )

            loc_trg_cnt = struct.unpack(self.loc_trg_cnt_fmt, payload[cursor : cursor + loc_trg_cnt_len])[0]
            cursor += loc_trg_cnt_len

            for _ in range(loc_trg_cnt):
                loc_trg_info = struct.unpack(self.loc_trg_info_fmt, payload[cursor : cursor + loc_trg_info_len])
                cursor += loc_trg_info_len
                loc_fus_res = TargetLocalInfo(
                    avs_id=loc_trg_info[0],
                    target_local_id=loc_trg_info[1],
                    local_position=np.array(
                        [loc_trg_info[2] * RES.lat, loc_trg_info[3] * RES.lon, loc_trg_info[4] * RES.alt - 900]
                    ),
                    target_local_class=loc_trg_info[5],
                    class_probability=loc_trg_info[6],
                )
                glb_fus_res.local_info_list.append(loc_fus_res)

            trg_fus_res.append(glb_fus_res)

        return {res.target_id: res for res in trg_fus_res}


class GridCoverageDataParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["grid_coverage_data"]):
        self.fmt = fmt
        self.dtype = np.uint8

    def _convert_to_binaries(self, num_list: tuple) -> np.ndarray:
        bit_list = []
        for num in num_list:
            bit_list.append([int(bit) for bit in format(num, "08b")])

        return np.array(bit_list)

    def parse(self, payload: bytes) -> np.ndarray:
        unpacked_pl = struct.unpack(self.fmt, payload)
        grid_coverage_data = self._convert_to_binaries(unpacked_pl).reshape(NUM_ROWS, NUM_COLS).astype(self.dtype)

        return grid_coverage_data


class SoftwareVersionRequestParser:

    def __init__(self, fmt: str = PAYLOAD_TO_FMT["sw_version_req"]):
        self.fmt = fmt

    def parse(self, payload: bytes) -> int:
        req = struct.unpack(self.fmt, payload)[0]
        req = req & 0x01

        return req


replan_req_parser = ReplanRequestParser()
mp_cmd_parser = MissionPlanCommandParser()
boundary_parser = BoundaryParser()
avs_info_parser = AvsInfoParser()
mp_trg_parser = MissionPlanTargetParser()
sw_version_req_parser = SoftwareVersionRequestParser()
circular_area_parser = CircularAreaParser()
polygon_area_parser = PolygonAreaParser()
mp_init_info_parser = MissionInitialInfoParser()
mp_state_parser = MissionStateParser()
trg_fus_res_parser = TargetFusionResultParser()
grid_coverage_data_parser = GridCoverageDataParser()
MSG_ID_TO_SYS_PARSER = {
    SystemMessageId.REPLAN_REQUEST.value: lambda p: replan_req_parser.parse(p),
    SystemMessageId.MISSION_PLAN_CMD.value: lambda p: mp_cmd_parser.parse(p),
    SystemMessageId.BOUNDARY.value: lambda p: boundary_parser.parse(p),
    SystemMessageId.AVS_INFO.value: lambda p: avs_info_parser.parse(p),
    SystemMessageId.MISSION_PLAN_TARGET.value: lambda p: mp_trg_parser.parse(p),
    SystemMessageId.SW_VERSION_REQ.value: lambda p: sw_version_req_parser.parse(p),
    SystemMessageId.CIRCULAR_AREA.value: lambda p: circular_area_parser.parse(p),
    SystemMessageId.POLYGON_AREA.value: lambda p: polygon_area_parser.parse(p),
    SystemMessageId.MISSION_INITIAL_INFO.value: lambda p: mp_init_info_parser.parse(p),
    SystemMessageId.MISSION_STATE.value: lambda p: mp_state_parser.parse(p),
    SystemMessageId.TARGET_FUSION_RESULT.value: lambda p: trg_fus_res_parser.parse(p),
    SystemMessageId.GRID_COVERAGE_DATA.value: lambda p: grid_coverage_data_parser.parse(p),
}
