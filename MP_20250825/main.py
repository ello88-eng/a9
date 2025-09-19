import json
import os
import signal
import sys
import termios
import threading
import time
from copy import deepcopy
from threading import Thread
from typing import Dict

from transformers.hf_argparser import HfArgumentParser

from commu.sim.receiver.sim_receiver import SimHeaderParser
from commu.sim.sender.sim_sender import GuiHeader, send_rl_state, send_sim_status_to_gui
from commu.sys.receiver.data.fs2mps import TargetFusionResult, TargetLocalInfo
from commu.sys.receiver.data.gcs2mps import AvsInfo, MissionPlanTarget, ReplanRequest
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.receiver.sys_receiver import (
    MSG_ID_TO_SYS_PARSER,
    SystemMessageHeaderParser,
)
from commu.sys.sender.sys_sender import send_smp_status, send_sw_version
from config.gui_config import SYS_TO_GUI_MSG_NO
from config.main_config import MainArgs
from config.mp_config import (
    ALGO_TO_NAME_DICT,
    GEN_SURV_PATH,
    MISSION_LIST,
    MISSION_MODE_TO_NAME_DICT,
    NUM_COLS,
    NUM_ROWS,
    REPLAN_INTERVAL,
    SHOW_INPUTS,
    SHOW_OUTPUT,
    STATE_FP,
    SYS_STATUS_TRANSITION,
    TRAIN_INTERVAL,
    USE_MP_TRG,
    USE_PREV_MP_OUTPUT,
    SystemMissionState,
)
from config.sim_msg_config import SimulationStatus
from config.sw_config import SW_VERSION
from config.sys_msg_config import UPDATER_FN_DICT, SystemMessageId
from manager.manager import Manager
from manager.mp_file_manager import MissionPlanFileManager
from planner.planner import Planner
from utils.logger import logger
from utils.operation import Operation, check_start
from utils.server import UdpServer

# 임무계획 입력 데이터 쓰레드 보호를 위한 쓰레딩 락
lock = threading.Lock()
# 모의 ICD 파서 클래스 초기화
sim_msg_header_parser = SimHeaderParser()
# 체계 ICD 파서 클래스 초기화
sys_msg_header_parser = SystemMessageHeaderParser()

# SW 버전 요청을 위한 하트비트 초기화
heartbeat = 1

#
accum_trg_fus_res = {}


def init(args: MainArgs) -> None:
    """20250611

    Args:
        args (MainArgs): _description_
    """
    global mp_input, manager, planner, gui_header, op

    # 연동 주소 업데이트
    if check_start([args.scenario_path]):
        args.addr_mps_ip = "192.168.10.202"
    else:
        args.addr_mps_ip = "192.168.10.17"
    args.addr_mps_port = args.addr_mps_port
    if check_start([args.scenario_path]):
        args.addr_gui_ip = "192.168.10.202"
    else:
        args.addr_gui_ip = "192.168.10.17"
    if check_start([args.scenario_path]):
        args.addr_rl_ip = "192.168.10.202"
    else:
        args.addr_rl_ip = "192.168.10.17"

    logger.info(f"TS ADDR: {(args.addr_ts_ip, args.addr_ts_port)}")
    logger.info(f"FM SIM ADDR: {(args.addr_fm_sim_ip, args.addr_fm_sim_port)}")
    logger.info(f"FM SYS ADDR: {(args.addr_fm_sys_ip, args.addr_fm_sys_port)}")
    logger.info(f"MPS ADDR: {(args.addr_mps_ip, args.addr_mps_port)}")
    logger.info(f"GUI ADDR: {(args.addr_gui_ip, args.addr_gui_port)}")
    logger.info(f"RL ADDR: {(args.addr_rl_ip, args.addr_rl_port)}")
    logger.info(f"DG ADDR: {(args.addr_dg_ip, args.addr_dg_port)}")

    # GUI 헤더 생성 클래스 초기화
    gui_header = GuiHeader((args.addr_gui_ip, args.addr_gui_port))

    # 임무표적관리 클래스 초기화
    manager = Manager(
        mission_list=MISSION_LIST,
        num_rows=NUM_ROWS,
        num_columns=NUM_COLS,
        replan_algo=ALGO_TO_NAME_DICT["Replan"],
        surv_algo=ALGO_TO_NAME_DICT["Surv"],
        reconn_algo=ALGO_TO_NAME_DICT["Reconn"],
        attack_algo=ALGO_TO_NAME_DICT["Attack"],
        scenario_fp=args.scenario_path,
        use_mp_trg=USE_MP_TRG,
        use_prev_mp_result=USE_PREV_MP_OUTPUT,
        src_id=args.src,
    )
    # 임무계획 입력 데이터클래스 초기화
    mp_input = MissionPlanInput()

    # 임무계획 알고리즘 클래스 초기화
    planner = Planner(args)

    # 임무시작 플래그
    op = Operation()


def create_input_from_json(state: Dict[str, dict]) -> MissionPlanInput:
    """수정

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
    input.grid_coverage = state["grid_data"]
    return input


def sys_receiver_callback(data: bytes, addr: str, args: MainArgs) -> None:
    """20250611

    Args:
        data (bytes): _description_
        addr (str): _description_
        args (MainArgs): _description_
    """
    global mp_input, heartbeat

    # if sys_msg_header_parser.validate(data):
    msg_id, payload = sys_msg_header_parser.parse(data)
    parser_fn = MSG_ID_TO_SYS_PARSER.get(msg_id)

    # SW 버전 요청에 대한 응답
    if msg_id == SystemMessageId.SW_VERSION_REQ:
        send_sw_version(address=(args.addr_fm_sys_ip, args.addr_mps_port), heartbeat=heartbeat)
        heartbeat = (heartbeat + 1) % 65535
        return

    # 메시지를 정상적으로 수신하면
    if payload is not None and parser_fn is not None:

        # GUI에 공유하는 메시지이면
        if msg_id in SYS_TO_GUI_MSG_NO.keys():
            gui_header.send(src=args.src, msg_id=msg_id, payload=payload)

        # 수신 메시지 처리
        value = parser_fn(payload)
        updater_fn = getattr(mp_input, UPDATER_FN_DICT[msg_id])
        with lock:
            updater_fn(value)

        # 초기 정보가 온 경우
        if msg_id == SystemMessageId.MISSION_INITIAL_INFO.value:
            # 사전 감시패턴 생성이 필요한 경우
            if GEN_SURV_PATH and mp_input.mission_init_info.surv_path_gen_method == 1:
                mp_file_manager = MissionPlanFileManager(args=args, mission_init_info=mp_input.mission_init_info)
                mp_file_manager.generate_surv_path(mp_input)


def validate_input(mp_input: MissionPlanInput) -> bool:
    """20250611

    Args:
        mp_input (MissionPlanInput): _description_

    Returns:
        bool: _description_
    """
    val = 0
    if mp_input.boundary.vertices.shape == (4, 2):
        val += 1
    for avs, info in mp_input.avs_info_dict.items():
        if avs >= 0 and info.avs_id >= 0 and info.system_mode >= 0 and info.position.shape == (3,) and info.soc >= 0.0:
            val += 1
    if mp_input.num_avs > 0 and mp_input.num_tgr >= 0:
        val += 1
    if val == 2 + mp_input.num_avs:
        return True
    return False


def execute_planner(args: MainArgs) -> None:
    """20250611

    Args:
        args (MainArgs): SW 환경설정 인자 데이터클래스
    """
    global mp_input

    while True:
        # 임무상태 변경
        if mp_input.mission_state.state in [SystemMissionState.START.value, SystemMissionState.FINISH.value]:
            op.is_started = mp_input.mission_state.state == 1

            # 임무가 시작되면 초기화
            if op.is_started and op.is_start_flag_changed:
                mp_input.reset_periodically()
                # 임무표적관리 클래스 초기화
                manager = Manager(
                    mission_list=MISSION_LIST,
                    num_rows=NUM_ROWS,
                    num_columns=NUM_COLS,
                    replan_algo=ALGO_TO_NAME_DICT["Replan"],
                    surv_algo=ALGO_TO_NAME_DICT["Surv"],
                    reconn_algo=ALGO_TO_NAME_DICT["Reconn"],
                    attack_algo=ALGO_TO_NAME_DICT["Attack"],
                    scenario_fp=args.scenario_path,
                    use_mp_trg=USE_MP_TRG,
                    use_prev_mp_result=USE_PREV_MP_OUTPUT,
                    src_id=args.src,
                )

                # 임무계획 알고리즘 클래스 초기화
                planner = Planner(args)

                op.reset_start_flag()

            op.status = SYS_STATUS_TRANSITION.get((op.status, mp_input.mission_state.state))

        # 학습 모드인 경우
        if args.train:
            with lock:
                if validate_input(mp_input):
                    time.sleep(TRAIN_INTERVAL)
                    send_rl_state(inputs=mp_input, addr=(args.addr_rl_ip, args.addr_rl_port))

        else:
            # 임무 시작 상태 & 임무계획 알고리즘이 동작 중이 아니면
            if op.is_started == True and op.is_planning == False:
                # 임무 재계획 판단 주기만큼 대기
                time.sleep(REPLAN_INTERVAL)

                # ! 임시
                if op.is_started == True:
                    logger.info(f"[임시] 임무 상태: 임무 중")
                if op.is_started == False:
                    logger.info(f"[임시] 임무 상태: 임무 중 아님")
                # ! 임시

                with lock:
                    # 임무계획 입력을 보고싶은 경우
                    if SHOW_INPUTS:
                        logger.info(mp_input.boundary)
                        logger.info(mp_input.avs_info_dict.keys())
                        logger.info(mp_input.trg_fus_res_dict.keys())

                    # ! 임시
                    # 임시: 자동임무 모드 출력
                    avs_to_implement_mode_dict: Dict[int, str] = {}
                    for avs_id, avs_info in mp_input.avs_info_dict.items():
                        avs_to_implement_mode_dict[avs_id] = MISSION_MODE_TO_NAME_DICT.get(avs_info.implement_mode)
                    logger.info(f"[임시] AVS별 자동임무 모드: {avs_to_implement_mode_dict}")
                    # 임시: 표적융합정보 출력
                    logger.info(f"[임시] 실시간 표적융합결과 (전역 ID): {mp_input.trg_fus_res_dict.keys()}")
                    accum_trg_fus_res.update(mp_input.trg_fus_res_dict)
                    logger.info(f"[임시] 누적 표적융합결과 (전역 ID): {accum_trg_fus_res.keys()}")
                    # ! 임시

                    # 임무계획 입력이 잘 수신되었는지 확인
                    if validate_input(mp_input):
                        # 임무계획 알고리즘 시작 선언
                        op.is_planning = True

                        # 입무계획 입력을 임무표적관리에 업데이트
                        manager.update_input(deepcopy(mp_input))
                        # JSON 파일로 호스트에 저장
                        manager.save_input_as_json()
                        # 임무계획 알고리즘 실행
                        planner.plan(manager=manager, show_output=SHOW_OUTPUT)

                    # 저장된 데이터 기반 임무계획 재생
                    if STATE_FP != "":
                        with open(STATE_FP, "r") as f:
                            loaded = json.load(f)
                        mp_input = create_input_from_json(loaded["state"])
                        manager.update_input(deepcopy(mp_input))
                        manager.save_input_as_json()
                        planner.plan(manager=manager, show_output=SHOW_OUTPUT)

                    # 임무계획 입력 초기화
                    mp_input.reset_periodically()

                    # 임무계획 알고리즘이 완료되었으면 데이터 수신가능 상태로 복귀
                    op.is_planning = False


def show_config():
    logger.info(f"임무계획 SW 버전: {'v' + '.'.join(map(str, SW_VERSION))}")
    logger.info(f"임무 종류: {MISSION_LIST}")
    logger.info(f"감시패턴 생성 여부: {GEN_SURV_PATH}")
    logger.info(f"임무별 임무계획 알고리즘: {ALGO_TO_NAME_DICT}")
    logger.info(f"임무 재계획 판단 간격 [초]: {REPLAN_INTERVAL}")


def main(args: MainArgs):
    fd = sys.stdin.fileno()
    # 현재 터미널 속성 저장
    orig_attrs = termios.tcgetattr(fd)
    new_attrs = list(orig_attrs)

    # ECHOCTL 꺼서 ^C 에코 방지
    if hasattr(termios, "ECHOCTL"):
        new_attrs[3] = new_attrs[3] & ~termios.ECHOCTL  # lflags 에서 ECHOCTL 비트 끔
        termios.tcsetattr(fd, termios.TCSANOW, new_attrs)

    # Ctrl+C 시그널 핸들러를 main 안에서 정의하여 args, operation 클로저로 잡아두기
    def handle_sigint_in_main(signum, frame):
        # 상태를 ExitSim 으로 변경
        op.status = SimulationStatus.EXIT_SIM.value
        op.is_started = True
        logger.info("Exiting...")
        # 3) GUI로 한번 더 상태 전송
        for _ in range(3):
            send_sim_status_to_gui(
                operation=op,
                src=args.src,
                scenario_path=args.scenario_path,
                address=(args.addr_gui_ip, args.addr_gui_port),
            )
            time.sleep(1)
        sys.exit(0)

    # 5) SIGINT를 우리가 정의한 핸들러로 연결
    signal.signal(signal.SIGINT, handle_sigint_in_main)

    # SW 초기화
    init(args)

    fs_sys_receiver = UdpServer(address=("0.0.0.0", args.addr_fm_sys_port), buffer_size=2**20, args=args)
    fs_sys_receiver_thread = Thread(target=fs_sys_receiver.start, args=(sys_receiver_callback, False), daemon=True)
    smp_status_sender_thread = Thread(
        target=send_smp_status, args=((args.addr_fm_sys_ip, args.addr_mps_port),), daemon=True
    )
    gui_status_sender_thread = Thread(
        target=send_sim_status_to_gui,
        args=(op, args.src, args.scenario_path, (args.addr_gui_ip, args.addr_gui_port)),
        daemon=True,
    )
    try:
        fs_sys_receiver_thread.start()
        smp_status_sender_thread.start()
        gui_status_sender_thread.start()
        execute_planner(args)
        fs_sys_receiver_thread.join()
    finally:
        fs_sys_receiver.stop()


if __name__ == "__main__":
    os.system("clear")
    show_config()
    parser = HfArgumentParser(MainArgs)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
