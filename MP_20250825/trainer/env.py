import pickle
import socket
import threading
import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from infos import AvsInfoForRl, TrgInfoForRl
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.logger import logger


class DataReceiver:
    def __init__(self, udp_ip="0.0.0.0", udp_port=None, timeout=1000):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.timeout = timeout  # 타임아웃 설정

        # UDP 소켓 설정
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.udp_port is None:
            self.sock.bind((self.udp_ip, 0))
            self.udp_port = self.sock.getsockname()[1]
        else:
            self.sock.bind((self.udp_ip, self.udp_port))

        self.state = {
            "scenario": np.zeros(40000),
            "agent": {id: AvsInfoForRl() for id in range(1, 51)},
            "target": {id: TrgInfoForRl() for id in range(1, 21)},
        }
        # 데이터 접근을 위한 락
        self.lock = threading.Lock()
        # 데이터를 수신했는지 여부를 나타내는 이벤트
        self.data_received_event = threading.Event()
        self.last_received_time = time.time()  # 마지막 데이터 수신 시간

    def start_receiving(self):
        receive_thread = threading.Thread(target=self.receive_and_update)
        receive_thread.daemon = True
        receive_thread.start()

    def receive_and_update(self):
        while True:
            try:
                self.sock.settimeout(self.timeout)
                data, addr = self.sock.recvfrom(40960)
                received_data = pickle.loads(data)
                with self.lock:
                    scenario_changes = received_data["state"]["scenario"][0]
                    for change in scenario_changes:
                        self.state["scenario"][change] = 1
                    self.state["agent"] = received_data["state"]["agent"]
                    self.state["target"] = received_data["state"]["target"]
                    self.last_received_time = time.time()
                    # 데이터 수신 이벤트 설정
                    self.data_received_event.set()
            except socket.timeout:
                logger.info("Timeout: Receiving data is stopped.")
                break
            except Exception as e:
                logger.info(f"Error receiving data: {e}")
                break

    def wait_for_data(self):
        # 데이터 수신을 기다림
        self.data_received_event.wait()

    def get_data(self):
        with self.lock:
            state_copy = self.state.copy()
        return {
            "state": state_copy,
        }

    def has_data_timed_out(self):
        return time.time() - self.last_received_time > self.timeout


class UnrealGymEnv(gym.Env):
    def __init__(self, udp_port=None, render_mode=None):
        self.render_mode = render_mode
        self.udp_port = udp_port
        self.observation_space = spaces.Dict(
            {
                "scenario": spaces.Box(low=0, high=1, shape=(40000,), dtype=np.float32),
                "agent": spaces.Box(low=0, high=4, shape=(7,), dtype=np.float32),
                "target": spaces.Box(low=0, high=4, shape=(80,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-100.0, high=100.0, shape=(2,), dtype=np.float32
        )
        self.receiver = DataReceiver(udp_port=udp_port)
        self.receiver.start_receiving()
        self.receiver.wait_for_data()  # 데이터 수신을 기다림

    def _get_obs(self):
        observation = self.receiver.get_data()
        return observation

    def _get_info(self):
        return {"debugg": ""}

    def step(self, action):
        if self.receiver.has_data_timed_out():
            raise RuntimeError("Receiving data is stopped.")

        observation = self._get_obs()
        terminated = False  # 환경 종료 조건 정의 필요
        reward = 1 if terminated else 0
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self):
        if self.receiver.has_data_timed_out():
            raise RuntimeError("Receiving data is stopped.")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        frame = "external Env"
        return frame
