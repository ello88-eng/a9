import threading
import xml.etree.ElementTree as ET

from config.mp_config import SystemMissionState
from config.sim_msg_config import SimulationStatus


class Operation:
    def __init__(
        self, status: int = SimulationStatus.INIT_COMPLETED.value, is_started: bool = False, is_planning: bool = False
    ) -> None:
        self._status = status
        self._is_started = is_started
        self._is_planning = is_planning
        self._is_start_flag_changed = False
        self._lock = threading.Lock()

    @property
    def status(self) -> int:
        with self._lock:
            return self._status

    @status.setter
    def status(self, new_status: int) -> None:
        with self._lock:
            self._status = new_status

    @property
    def is_start_flag_changed(self) -> bool:
        with self._lock:
            return self._is_start_flag_changed

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._is_started

    @is_started.setter
    def is_started(self, is_started: bool) -> None:
        with self._lock:
            if self._is_started != is_started and is_started == SystemMissionState.START.value:
                self._is_start_flag_changed = True
            self._is_started = is_started

    def reset_start_flag(self):
        self._is_start_flag_changed = False

    @property
    def is_planning(self) -> bool:
        with self._lock:
            return self._is_planning

    @is_planning.setter
    def is_planning(self, is_planning: bool) -> None:
        with self._lock:
            self._is_planning = is_planning


def read_sim_info(scenario_path_list: list, module: str):
    module_type_list = []
    for scenario_path in scenario_path_list:
        if scenario_path != "":
            # XML 파일 읽기
            tree = ET.parse("/workspace/shared/" + scenario_path + ".xml")
            # 루트 엘리먼트 가져오기
            root = tree.getroot()
            # 엘리먼트 찾기
            simulation = root.find(".//simulation")
            module_type = simulation.find(module).text
            module_type_list.append(module_type)
    return module_type_list


def check_start(scenario_path_list: list):
    type_list = read_sim_info(scenario_path_list=scenario_path_list, module="MP")
    if "P" in type_list:
        return False
    else:
        return True
