import binascii
import struct
from typing import Tuple

import numpy as np

from commu.sim.receiver.data.es2mps import SimulationObjInfo
from commu.sim.receiver.data.ts2mps import SimulationInitCommand
from config.resolution_config import RES
from config.sim_msg_config import SIM_HEADER_FMT, SimulationMessageId


class SimHeaderParser:

    def __init__(self, header_fmt: str = SIM_HEADER_FMT):
        self.header_fmt = header_fmt
        self.header_size = struct.calcsize(self.header_fmt)
        self.msgs = {}

    def validate(self, data: bytes) -> bool:
        payload = data[:-4]
        checksum = data[-4:]
        if checksum == binascii.crc32(payload):
            return True
        else:
            return False

    def parse(self, data: bytes) -> Tuple[str, bytes]:
        _, _, _, src, dst, msg_n, timestamp = struct.unpack(self.header_fmt, data[: self.header_size])
        msg_id = f"0x{src:02x}" + hex(dst)[2:] + f"{msg_n:02x}"
        payload = data[self.header_size : -4]
        _ = struct.unpack("I", data[-4:])

        return msg_id, timestamp, payload


class InitParser:

    def __init__(self, fmt: str = ">B BB 255s 255s BB") -> None:
        self.fmt = fmt

    def parse(self, timestamp: int, payload: bytes) -> SimulationInitCommand:
        field = struct.unpack(self.fmt, payload)
        num_sims = field[0]
        base_ports = np.array(field[1:3]) * RES.port
        scenario_paths = field[3:5]
        speeds = np.array(np.array(field[5:]) * RES.speed, dtype=int)

        return SimulationInitCommand(
            num_sims=num_sims,
            base_ports=base_ports,
            scenario_paths=["".join(path.decode()).strip("\x00") for path in scenario_paths if type(path) == bytes],
            speeds=speeds,
        )


class CtrlParser:

    def __init__(self, fmt: str = ">B") -> None:
        self.fmt = fmt

    def parse(self, timestamp: int, payload: bytes) -> int:
        return struct.unpack(self.fmt, payload)[0]


class ObjInfoParser:

    def __init__(
        self,
        plt_fmt: str = "> 50h 50i50i50H 50h50h50h 50i50i50i 50H50h50h",
        sensor_fmt: str = "> 50H50H 50I50i 50I",
        error_fmt: str = "> 50i50i50H 50h50h50h 50H50h50h 50H50H50I50i50I",
        fp_fmt: str = "> 50i50i50H 50h50h50h50h50h50h50h50h",
        trg_fmt: str = "> 100H 100B 100i100i100H 100f100f100f 100h100h100H",
    ) -> None:
        self.plt_fmt = plt_fmt
        self.sensor_fmt = sensor_fmt
        self.error_fmt = error_fmt
        self.fp_fmt = fp_fmt
        self.trg_fmt = trg_fmt

    def parse(self, timestamp: int, payload: bytes) -> SimulationObjInfo:
        #
        plf_len = struct.calcsize(self.plt_fmt)
        sensor_len = struct.calcsize(self.sensor_fmt)
        error_len = struct.calcsize(self.error_fmt)
        fp_len = struct.calcsize(self.fp_fmt)
        trg_len = struct.calcsize(self.trg_fmt)
        #
        i = 0
        plt_payload = struct.unpack(self.plt_fmt, payload[:plf_len])
        i += plf_len
        sensor_payload = struct.unpack(self.sensor_fmt, payload[i : i + sensor_len])
        i += sensor_len
        error_payload = struct.unpack(self.error_fmt, payload[i : i + error_len])
        i += error_len
        fp_payload = struct.unpack(self.fp_fmt, payload[i : i + fp_len])
        i += fp_len
        trg_payload = struct.unpack(self.trg_fmt, payload[i : i + trg_len])
        #
        sim_obj_info = SimulationObjInfo(
            timestamp=timestamp,
            avs_ids=np.array(plt_payload[:50]),
            plf_lats=np.array(plt_payload[50:100]) * RES.lat,
            plf_lons=np.array(plt_payload[100:150]) * RES.lon,
            plf_alts=np.array(plt_payload[150:200]) * RES.alt - 900,
            plf_vel_n=np.array(plt_payload[200:250]) * RES.vel_n,
            plf_vel_e=np.array(plt_payload[250:300]) * RES.vel_e,
            plf_vel_d=np.array(plt_payload[300:350]) * RES.vel_d,
            plf_heading=np.array(plt_payload[500:550]) * RES.heading,
            plf_pitchs=np.array(plt_payload[550:600]) * RES.pitch,
            plf_rolls=np.array(plt_payload[600:650]) * RES.roll,
            fov_horzs=np.array(sensor_payload[:50]) * RES.fov_horz,
            fov_verts=np.array(sensor_payload[50:100]) * RES.fov_horz,
            rel_azim_angs=np.array(sensor_payload[100:150]) * RES.gimbal,
            rel_elev_angs=np.array(sensor_payload[150:200]) * RES.gimbal,
            slant_ranges=np.array(sensor_payload[200:250]) * RES.slant,
            plf_lats_with_error=np.array(error_payload[:50]) * RES.lat,
            plf_lons_with_error=np.array(error_payload[50:100]) * RES.lon,
            plf_alts_with_error=np.array(error_payload[100:150]) * RES.alt,
            plf_vel_n_with_error=np.array(error_payload[150:200]) * RES.vel_n,
            plf_vel_e_with_error=np.array(error_payload[200:250]) * RES.vel_e,
            plf_vel_d_with_error=np.array(error_payload[250:300]) * RES.vel_d,
            plf_heading_with_error=np.array(error_payload[300:350]) * RES.heading,
            plf_pitchs_with_error=np.array(error_payload[350:400]) * RES.pitch,
            plf_rolls_with_error=np.array(error_payload[400:450]) * RES.roll,
            fov_horzs_with_error=np.array(error_payload[450:500]) * RES.fov_horz,
            fov_verts_with_error=np.array(error_payload[500:550]) * RES.fov_horz,
            rel_azim_angs_with_error=np.array(error_payload[550:600]) * RES.gimbal,
            rel_elev_angs_with_error=np.array(error_payload[600:650]) * RES.gimbal,
            slant_ranges_with_error=np.array(error_payload[650:700]) * RES.slant,
            fr_center_lats=np.array(fp_payload[:50]) * RES.lat,
            fr_center_lons=np.array(fp_payload[50:100]) * RES.lon,
            fr_center_elevs=np.array(fp_payload[100:150]) * RES.alt - 900,
            fp_lats_1=np.array(fp_payload[150:200]) * RES.corner,
            fp_lons_1=np.array(fp_payload[200:250]) * RES.corner,
            fp_lats_2=np.array(fp_payload[250:300]) * RES.corner,
            fp_lons_2=np.array(fp_payload[300:350]) * RES.corner,
            fp_lats_3=np.array(fp_payload[350:400]) * RES.corner,
            fp_lons_3=np.array(fp_payload[400:450]) * RES.corner,
            fp_lats_4=np.array(fp_payload[450:500]) * RES.corner,
            fp_lons_4=np.array(fp_payload[500:550]) * RES.corner,
            trg_ids=np.array(trg_payload[:100]),
            trg_types=np.array(trg_payload[100:200]),
            trg_lats=np.array(trg_payload[200:300]) * RES.lat,
            trg_lons=np.array(trg_payload[300:400]) * RES.lon,
            trg_alts=np.array(trg_payload[400:500]) * RES.alt - 900,
            trg_pos_x=np.array(trg_payload[500:600]) * 0.01,
            trg_pos_y=np.array(trg_payload[600:700]) * 0.01,
            trg_pos_z=np.array(trg_payload[700:800]) * 0.01,
            trg_roll=np.array(trg_payload[800:900]) * RES.roll,
            trg_pitch=np.array(trg_payload[900:1000]) * RES.pitch,
            trg_yaw=np.array(trg_payload[1000:1100]) * RES.heading,
        )
        return sim_obj_info


init_parser = InitParser()
ctrl_parser = CtrlParser()
obj_info_parser = ObjInfoParser()
MSG_ID_TO_SIM_PARSER = {
    SimulationMessageId.SIM_INIT_CMD.value: lambda t, p: init_parser.parse(t, p),
    SimulationMessageId.SIM_CTRL_CMD.value: lambda t, p: ctrl_parser.parse(t, p),
    SimulationMessageId.SIM_OBJ_INFO.value: lambda t, p: obj_info_parser.parse(t, p),
}
