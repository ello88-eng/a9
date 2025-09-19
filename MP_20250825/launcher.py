import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers.hf_argparser import HfArgumentParser

from commu.sim.receiver.data.ts2mps import SimulationInitCommand
from commu.sim.receiver.sim_receiver import InitParser, SimHeaderParser
from mission_planning.configs.config import STATUS, MainArgs
from utils.logger import logger
from utils.operation import check_start, read_sim_info
from utils.server import UdpServer


@dataclass
class LauncherArgs:
    ip: Optional[str] = "0.0.0.0"
    port: Optional[int] = 45000
    # trains: Optional[List[bool]] = field(default_factory=lambda: [False, False])
    # ddps: Optional[List[bool]] = field(default_factory=lambda: [False, False])
    train: Optional[bool] = False
    ddp: Optional[bool] = False


sim_header_parser = SimHeaderParser()
sim_init_parser = InitParser()


def create_tmux_session(session_name: str, window_name: str) -> None:
    subprocess.call(["tmux", "new-session", "-d", "-s", session_name, "-n", window_name])


def create_tmux_window(session_name: str, window_name: str) -> None:
    subprocess.call(["tmux", "new-window", "-t", session_name, "-n", window_name])


def send_command_to_tmux(session_name: str, command: str, target_window: int) -> None:
    subprocess.call(
        [
            "tmux",
            "send-keys",
            "-t",
            f"{session_name}:{target_window}",
            command,
            "C-m",
        ]
    )


def kill_tmux_session(session_name: str) -> None:
    subprocess.call(["tmux", "kill-session", "-t", session_name])


def start_tmux_session(
    session_name: str,
    msg: SimulationInitCommand,
    args: LauncherArgs,
    fm_addr_list: List[Tuple[str, int]],
) -> None:
    main_cmd = "python /workspace/mission_planning/main.py"
    trainer_cmds = {
        "no_ddp": "python trainer/train.py",
        "ddp": "python trainer/train_ddp.py",
    }
    if "TMUX" in os.environ:
        logger.info("Warning: Already inside a tmux session. Please run this script outside of tmux.")
        return
    # 메인 tmux 세션 생성
    create_tmux_session(session_name=session_name, window_name="win_0")
    for i in range(msg.num_sims):
        window_name = f"win_{i}"
        if i > 0:
            # 첫 번째 창 이후의 새 창 생성
            create_tmux_window(session_name=session_name, window_name=window_name)
        # 각 창에 명령어 실행
        send_command_to_tmux(
            session_name=session_name,
            command=f"{main_cmd} --src {i + 1} --base_port {msg.base_ports[i]} --scenario_path {msg.scenario_paths[i]} --speed {msg.speeds[i]} --addr_fm_sys_ip {json.dumps(fm_addr_list[i])} --train {args.train}",
            target_window=i,
        )
    # trainer.py 실행
    # for train in args.trains:
    if args.train:
        trainer_window_name = f"win_{i + 1}"
        create_tmux_window(session_name=session_name, window_name=trainer_window_name)
        send_command_to_tmux(
            session_name=session_name,
            command=f"{trainer_cmds['ddp'] if args.ddp == True else trainer_cmds['no_ddp']} --num_envs {msg.num_sims}",
            target_window=i + 1,
        )
    if (msg.num_sims == 1 and args.train) or (msg.num_sims == 2 and not args.train):
        # 모든 창을 한눈에 볼 수 있도록 레이아웃 재조정
        subprocess.call(["tmux", "select-window", "-t", f"{session_name}:0"])
        subprocess.call(["tmux", "split-window", "-h"])
        subprocess.call(["tmux", "select-pane", "-t", f"{session_name}:0"])
        subprocess.call(["tmux", "swap-pane", "-s", f"{session_name}:1"])
        subprocess.call(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"])
        subprocess.call(["tmux", "kill-window", "-t", f"{session_name}:1"])
    if msg.num_sims == 2 and args.train:
        subprocess.call(["tmux", "select-window", "-t", f"{session_name}:0"])
        subprocess.call(["tmux", "split-window", "-v"])
        subprocess.call(["tmux", "select-pane", "-t", f"{session_name}:0"])
        subprocess.call(["tmux", "swap-pane", "-s", f"{session_name}:1"])
        subprocess.call(["tmux", "split-window", "-h"])
        subprocess.call(["tmux", "select-pane", "-t", f"{session_name}:0"])
        subprocess.call(["tmux", "swap-pane", "-s", f"{session_name}:2"])
        subprocess.call(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"])
        subprocess.call(["tmux", "kill-window", "-t", f"{session_name}:1"])
        subprocess.call(["tmux", "kill-window", "-t", f"{session_name}:2"])
    # tmux 세션 연결
    subprocess.call(["tmux", "attach-session", "-t", session_name])


def set_address(scenario_path_list: list):
    fm_addr_list = [MainArgs().addr_fm_sys_ip, MainArgs().addr_fm_sys_ip]
    fm_type_list = read_sim_info(scenario_path_list=scenario_path_list, module="FM")
    for i, fm_type in enumerate(fm_type_list):
        if fm_type == "P":
            fm_addr_list[i] = "192.168.10.16"
        elif fm_type == "S":
            fm_addr_list[i] = "192.168.10.202"
    return fm_addr_list


def udp_rcv_callback(data: bytes, addr: str, m_args=None) -> None:
    global STATUS, sim_header_parser, sim_init_parser, args
    _, timestamp, payload = sim_header_parser.parse(data)
    msg = sim_init_parser.parse(timestamp, payload)
    if check_start(msg.scenario_paths):
        fm_addr_list = set_address(msg.scenario_paths)
        start_tmux_session(session_name="MPS", msg=msg, args=args, fm_addr_list=fm_addr_list)


def launch(args: LauncherArgs) -> None:
    server = UdpServer(address=(args.ip, args.port), buffer_size=2**20, args=MainArgs())
    server_thread = threading.Thread(target=server.start, args=(udp_rcv_callback,), daemon=True)
    try:
        server_thread.start()
        server_thread.join()
    finally:
        server.stop()


if __name__ == "__main__":
    os.system("cls" if os.name in ["nt", "dos"] else "clear")
    args = HfArgumentParser(LauncherArgs).parse_args_into_dataclasses()[0]
    logger.info(args)
    launch(args)
