import pathlib
import struct
import sys
import threading
from dataclasses import dataclass, field

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from typing import Dict, Tuple

import numpy as np
from mission_planning.configs.config import CRC_16_TABLE, RES
from utils.server import UdpServer


@dataclass
class MpOut:
    timestamp: int = -1
    id: int = -1
    approval_flag: int = -1
    attack_req: int = -1
    group_id: int = -1
    num_avs_per_group: int = -1
    mode: int = -1
    speed: int = -1
    num_wps: int = -1
    wps: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    turn_rad: int = -1
    turn_dir: int = -1
    trg_pt: np.ndarray[float] = field(default_factory=lambda: np.zeros((1,)))
    trg_id: int = -1
    attack_prio: int = -1
    is_bd: int = -1


class SysHeaderParser:

    def __init__(
        self,
        header_fmt: str = ">BBBBBBB",
        msg_id_len: int = 3,
        timestamp_len: int = 5,
    ):
        self.header_fmt = header_fmt
        self.header_size = struct.calcsize(self.header_fmt)
        self.msg_id_len = msg_id_len
        self.timestamp_len = timestamp_len

    def _compute_crc16(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            pos = (crc >> 8) ^ byte
            crc = (crc >> 8) ^ CRC_16_TABLE[pos]
        return crc

    def validate(self, data: bytes) -> bool:
        payload = data[:-2]
        checksum = data[-2:]
        if checksum == int(self._compute_crc16(payload)).to_bytes(length=2, byteorder="big"):
            return True
        else:
            return False

    def parse(self, data: bytes) -> Tuple[str, bytes]:
        avs_id, msg_id, msg_len = struct.unpack(self.header_fmt, data[: self.header_size])
        payload = data[self.header_size : -2]
        checksum = struct.unpack(">H", data[-2:])
        return msg_id, payload


class MpOutParser:

    def __init__(self, fmts: str = ["> Q B BB BB B H H", "> iiH", "> HH iiH B BB"]) -> None:
        self.fmts = fmts
        self.lens = [struct.calcsize(fmt) for fmt in fmts]

    def parse(self, payload: bytes) -> MpOut:
        (
            timestmap,
            avs,
            reconn_req,
            attack_req,
            gid,
            num_avs_per_group,
            mode,
            speed,
            num_wps,
        ) = struct.unpack(self.fmts[0], payload[: self.lens[0]])
        fmt = ">" + "iiH" * num_wps
        wps = np.array(struct.unpack(payload[self.lens[0] : self.lens[0] + struct.calcsize(fmt)])).reshape(
            num_wps, -1
        ) * np.tile([RES.lat, RES.lon, RES.alt], (num_wps, 1))
        (
            turn_rad,
            turn_dir,
            trg_pt_lat,
            trg_pt_lon,
            trg_pt_alt,
            trg_id,
            attack_prio,
            is_bd,
        ) = struct.unpack(
            self.fmts[2],
            payload[self.lens[0] + struct.calcsize(fmt) : self.lens[0] + struct.calcsize(fmt) + self.lens[2]],
        )
        trg_pt = np.array([[trg_pt_lat, trg_pt_lon, trg_pt_alt]]) * np.tile([RES.lat, RES.lon, RES.alt], (1, 3))
        return MpOut(
            timestamp=timestmap,
            id=avs,
            reconn_req=reconn_req,
            attack_req=attack_req,
            group_id=gid,
            num_avs_per_group=num_avs_per_group,
            mode=mode,
            speed=speed,
            num_wps=num_wps,
            # wps=wps,
            turn_rad=turn_rad,
            turn_dir=turn_dir,
            # trg_pt=trg_pt,
            trg_id=trg_id,
            attack_prio=attack_prio,
            is_bd=is_bd,
        )


mp_outs: Dict[int, MpOut] = {}
header_parser = SysHeaderParser()
parser = MpOutParser()


def rcv_callback(data: bytes, addr: str) -> None:
    _, payload = header_parser.parse(data)
    mp_out: MpOut = parser.parse(payload)
    mp_outs.update({mp_out.id: mp_out})
    print(mp_outs)


def main():
    rcv_server = UdpServer(address=("0.0.0.0", 10007), buffer_size=1024 * 50)
    rcv_server_thread = threading.Thread(target=rcv_server.start, args=(rcv_callback,), daemon=True)
    try:
        rcv_server_thread.start()
        rcv_server_thread.join()
    finally:
        rcv_server.stop()


if __name__ == "__main__":
    main()
