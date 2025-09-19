import json
import os
import signal
import sys
import termios
import threading
import time
from copy import deepcopy
from typing import Dict

from transformers.hf_argparser import HfArgumentParser

from commu.sim.receiver.sim_receiver import SimHeaderParser
from commu.sim.sender.sim_sender import GuiHeader, send_sim_status_to_gui
from commu.sys.receiver.data.fs2mps import TargetFusionResult, TargetLocalInfo
from commu.sys.receiver.data.gcs2mps import AvsInfo, MissionPlanTarget, ReplanRequest
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.receiver.sys_receiver import MSG_ID_TO_SYS_PARSER, SystemMessageHeaderParser
from manager.manager import Manager
from mission_planning.configs.config import (
    ADDR_GZB,
    ALGO_TO_NAME_DICT,
    MISSION_LIST,
    MSG_TO_UPDATER_DICT,
    NUM_COLS,
    NUM_ROWS,
    REPLAN_INTERVAL,
    SHOW_INPUTS,
    SHOW_OUTPUT,
    STATE_PATH,
    STATUS,
    SYS_TO_GUI_MSG_NO,
    USE_MP_TRG,
    USE_PREV_MP_OUTPUT,
    MainArgs,
)
from planner.planner import Planner
from utils.logger import logger
from utils.operation import Operation, check_start
from utils.server import UdpServer
from utils.timer import get_timestamp

# 임무계획 입력 데이터 쓰레드 보호를 위한 쓰레딩 락
lock = threading.Lock()
# 모의 ICD 파서 클래스 초기화
sim_header_parser = SimHeaderParser()
# 체계 ICD 파서 클래스 초기화
sys_header_parser = SystemMessageHeaderParser()


def init(m_args: MainArgs) -> None:
    """20250611

    Args:
        m_args (MainArgs): _description_
    """
    global mp_input, manager, planner, gui_header, operation

    # 연동 주소 업데이트
    if check_start([m_args.scenario_path]):
        m_args.addr_gui_ip = "192.168.10.202"
    else:
        m_args.addr_gui_ip = "192.168.10.17"
    logger.info(f"GUI ADDR: {(m_args.addr_gui_ip, m_args.addr_gui_port)}")

    # GUI 헤더 생성 클래스 초기화
    gui_header = GuiHeader((m_args.addr_gui_ip, m_args.addr_gui_port))

    # 임무표적관리 클래스 초기화
    manager = Manager(
        mission_list=MISSION_LIST,
        num_rows=NUM_ROWS,
        num_columns=NUM_COLS,
        replan_algo=ALGO_TO_NAME_DICT["Replan"],
        surv_algo=ALGO_TO_NAME_DICT["Surv"],
        reconn_algo=ALGO_TO_NAME_DICT["Reconn"],
        attack_algo=ALGO_TO_NAME_DICT["Attack"],
        scenario_fp=m_args.scenario_path,
        use_mp_trg=USE_MP_TRG,
        use_prev_mp_result=USE_PREV_MP_OUTPUT,
        src_id=m_args.src,
    )
    # 임무계획 입력 데이터클래스 초기화
    mp_input = MissionPlanInput()

    # 임무계획 알고리즘 클래스 초기화
    planner = Planner(m_args)

    # 임무시작 플래그
    operation = Operation()


def create_input_from_json(state: Dict[str, dict]) -> MissionPlanInput:
    """20250609

    Args:
        state (Dict[str, dict]): _description_

    Returns:
        MissionPlanInput: _description_
    """
    input = MissionPlanInput(
        timestamp=state["timestamp"],
        replan_req=state["replan_req"],
        mp_cmd_dict={int(i): ReplanRequest(**cmd) for i, cmd in state["cmds"].items()},
        num_avs=state["num_avs"],
        num_tgr=state["num_obj"],
        boundary=state["boundary"],
        area=state["area"],
        avs_info_dict={int(i): AvsInfo(**info) for i, info in state["avs_infos"].items()},
        mp_trg_dict={int(i): MissionPlanTarget(**gcs_trg) for i, gcs_trg in state["gcs_trg_infos"].items()},
    )
    fus_outs = {}
    loc_infos = []
    for i, fus_out in state["fus_outs"].items():
        for loc_info in fus_out["loc_infos"]:
            loc_infos.append(TargetLocalInfo(**loc_info))
        fus_outs.update(
            {
                int(i): TargetFusionResult(
                    timestamp=fus_out["timestamp"],
                    target_id=fus_out["id"],
                    position=fus_out["pos"],
                    target_class=fus_out["trg_cls"],
                    local_info_list=loc_infos,
                )
            }
        )
        loc_infos = []
    input.trg_fus_res_dict = fus_outs
    input.w_coverage = state["grid_data"]
    return input


def sys_rcv_callback(data: bytes, addr: str, m_args: MainArgs) -> None:
    """20250611

    Args:
        data (bytes): _description_
        addr (str): _description_
        m_args (MainArgs): _description_
    """
    global mp_input
    # if sys_header_parser.validate(data):
    msg_id, payload = sys_header_parser.parse(data)
    parse_fn = MSG_ID_TO_SYS_PARSER.get(msg_id)
    if payload is not None and parse_fn is not None:
        # 임무 시작 플래그
        if operation.is_started == False and msg_id != 0x7C05:
            operation.is_started = True
        # 임무상태 변경
        if operation.status == STATUS["InitCompleted"] and msg_id != 0x7C05:
            operation.status = STATUS["StartSim"]
        value = parse_fn(payload)
        updater = getattr(mp_input, MSG_TO_UPDATER_DICT[msg_id])
        if msg_id in SYS_TO_GUI_MSG_NO.keys():
            gui_header.send(src=m_args.src, msg_id=msg_id, payload=payload)
        with lock:
            updater(value)


def validate_input(mp_input: MissionPlanInput) -> bool:
    """20250611

    Args:
        mp_input (MissionPlanInput): _description_

    Returns:
        bool: _description_
    """
    val = 0
    if mp_input.boundary.shape == (4, 2):
        val += 1
    for avs, info in mp_input.avs_info_dict.items():
        if (
            avs >= 0
            and info.avs_id >= 0
            and info.avs_system_mode >= 0
            and info.position.shape == (3,)
            and info.soc >= 0.0
        ):
            val += 1
    if mp_input.num_avs > 0 and mp_input.num_tgr >= 0:
        val += 1
    if val == 2 + mp_input.num_avs:
        return True
    return False


def execute_planner(m_args: MainArgs) -> None:
    """20250611

    Args:
        m_args (MainArgs): _description_
    """
    global mp_input

    # 임무계획 알고리즘 동작 여부 초기화
    while True:
        # 임무 시작 상태 & 임무계획 알고리즘이 동작 중이 아니면
        if operation.is_started == True and operation.is_planning == False:
            # 임무 재계획 판단 주기만큼 대기
            time.sleep(REPLAN_INTERVAL)
            with lock:
                # 임무계획 입력을 보고싶은 경우
                if SHOW_INPUTS:
                    logger.info(mp_input.boundary)
                    logger.info(mp_input.avs_info_dict.keys())
                    logger.info(mp_input.trg_fus_res_dict.keys())
                # 임무계획 입력이 잘 수신되었는지 확인
                if validate_input(mp_input):
                    # 임무계획 알고리즘 시작 선언
                    operation.is_planning = True
                    # 임무계획 입력이 완성되면 timestamp 추가
                    mp_input.timestamp = get_timestamp()
                    # 입무계획 입력을 임무표적관리에 업데이트
                    manager.update_input(deepcopy(mp_input))
                    # JSON 파일로 호스트에 저장
                    manager.save_input_as_json()
                    # 임무계획 알고리즘 실행
                    planner.plan(manager=manager, show_output=SHOW_OUTPUT)
                # 저장된 데이터 기반 임무계획 재생
                if STATE_PATH != "":
                    with open(STATE_PATH, "r") as f:
                        loaded = json.load(f)
                    mp_input = create_input_from_json(loaded["state"])
                    manager.update_input(deepcopy(mp_input))
                    manager.save_input_as_json()
                    planner.plan(manager=manager, show_output=SHOW_OUTPUT)
                # 임무계획 입력 초기화
                mp_input.reset_periodically()
                # 임무계획 알고리즘이 완료되었으면 데이터 수신가능 상태로 복귀
                operation.is_planning = False


def main(m_args: MainArgs):
    fd = sys.stdin.fileno()
    # 현재 터미널 속성 저장
    orig_attrs = termios.tcgetattr(fd)
    new_attrs = list(orig_attrs)

    # ECHOCTL 꺼서 ^C 에코 방지
    if hasattr(termios, "ECHOCTL"):
        new_attrs[3] = new_attrs[3] & ~termios.ECHOCTL  # lflags 에서 ECHOCTL 비트 끔
        termios.tcsetattr(fd, termios.TCSANOW, new_attrs)

    # Ctrl+C 시그널 핸들러를 main 안에서 정의하여 m_args, operation 클로저로 잡아두기
    def handle_sigint_in_main(signum, frame):
        # 상태를 ExitSim 으로 변경
        operation.status = STATUS["ExitSim"]
        operation.is_started = True
        logger.info("Exiting...")
        # 3) GUI로 한번 더 상태 전송
        for _ in range(3):
            send_sim_status_to_gui(
                op=operation,
                src=m_args.src,
                scenario_path=m_args.scenario_path,
                address=(m_args.addr_gui_ip, m_args.addr_gui_port),
            )
            time.sleep(1)
        sys.exit(0)

    # 5) SIGINT를 우리가 정의한 핸들러로 연결
    signal.signal(signal.SIGINT, handle_sigint_in_main)
    init(m_args)

    sys_server = UdpServer(address=("0.0.0.0", ADDR_GZB["Planner"][1]), buffer_size=1024 * 1024, args=m_args)
    sys_server_thread = threading.Thread(target=sys_server.start, args=(sys_rcv_callback,), daemon=True)
    gui_status_thread = threading.Thread(
        target=send_sim_status_to_gui,
        args=(
            operation,
            m_args.src,
            m_args.scenario_path,
            (m_args.addr_gui_ip, m_args.addr_gui_port),
        ),
        daemon=True,
    )
    try:
        sys_server_thread.start()
        gui_status_thread.start()
        execute_planner(m_args)
        sys_server_thread.join()
    finally:
        sys_server.stop()


if __name__ == "__main__":
    os.system("cls" if os.name in ["nt", "dos"] else "clear")
    parser = HfArgumentParser(MainArgs)
    m_args = parser.parse_args_into_dataclasses()
    main(m_args[0])
