from dataclasses import dataclass
from typing import Optional


@dataclass
class MainArgs:
    # 모의 번호
    src: Optional[int] = 1
    # 모의 Base 포트
    base_port: Optional[int] = 10000
    # 시나리오 파일 경로
    scenario_path: Optional[str] = ""
    # 모의 배속
    speed: Optional[int] = 1
    # 연동 주소
    addr_ts_ip: Optional[str] = "192.168.10.101"
    addr_ts_port: Optional[int] = 10001
    addr_fm_sim_ip: Optional[str] = "192.168.10.16"
    addr_fm_sim_port: Optional[int] = 10009
    addr_fm_sys_ip: Optional[str] = "192.168.10.16"
    addr_fm_sys_port: Optional[int] = 10006
    addr_mps_ip: Optional[str] = "192.168.10.17"
    addr_mps_port: Optional[int] = 10007
    addr_gui_ip: Optional[str] = "192.168.10.17"
    addr_gui_port: Optional[int] = 19292
    addr_rl_ip: Optional[str] = "192.168.10.17"
    addr_rl_port: Optional[int] = 12929
    addr_dg_ip: Optional[str] = "192.168.10.205"
    addr_dg_port: Optional[int] = 10008
    # 강화학습 여부
    train: Optional[bool] = False
