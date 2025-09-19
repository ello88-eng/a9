from typing import Dict, List

from manager.manager import Manager
from utils.logger import logger


class ADDReplanAlgorithmRunner:
    def __init__(self, manager: Manager):
        self.manager = manager

    def select(self):
        """현재 임무상태 기반 비행체별 가능한 임무 도출"""

        if hasattr(self.manager, "prev_mp_input"):
            if len(self.manager.mp_input.avs_info_dict) < len(self.manager.prev_mp_input.avs_info_dict):
                mode_to_mission_dict = {
                    "이동": ["A", "R", "S"],
                    "감시": ["A", "R", "s"],
                    "정찰": ["A"],
                    "추적": ["A"],
                    "타격": ["S", "R"],
                }
            else:
                mode_to_mission_dict = {
                    "이동": ["A", "R", "S"],
                    "감시": ["A", "R", "s"],
                    "정찰": ["A"],
                    "추적": ["A"],
                    "타격": ["S", "R"],
                }
        else:
            mode_to_mission_dict = {
                "이동": ["A", "R", "S"],
                "감시": ["A", "R", "s"],
                "정찰": ["A"],
                "추적": ["A"],
                "타격": ["S", "R"],
            }
        avs_to_avail_task: Dict[int, List[str]] = {}
        for avs_id in self.manager.avs_to_implement_mode_dict.keys():
            avs_to_avail_task[avs_id] = mode_to_mission_dict.get(self.manager.avs_to_implement_mode_dict[avs_id], [])
        logger.info(f"비행체별 수행가능 임무: {avs_to_avail_task}")

        return avs_to_avail_task
