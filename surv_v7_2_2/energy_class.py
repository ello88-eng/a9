from state_helper_class import StateHelper
from typing import List, Tuple, Optional, Dict
import numpy as np

class Energy:
    def __init__(self):
        self.sh = StateHelper()

    def energy_open_coverage_from_state(self, nx:int, ny:int, obs_pos, state: Dict[str, np.ndarray]) -> float:
        """
        E = 1 - coverage (추후 패널티 확장 예정)
        """
        bboxes = self.sh.bboxes_from_state_ll(state)
        ratio, _ = self.sh.coverage_ratio_open_space(nx, ny, obs_pos, bboxes)
        return 1.0 - ratio

    def energy_open_vs_obstacle_from_state(
        self,
        nx:int, ny:int, obs_pos, state,
        lam: float = 1.0,
        penalty_norm: str = "covered",  # {"covered","grid"}
    ):
        """
        E = (1 - open_ratio) + lam * penalty
        - penalty_norm="covered": penalty = nonopen_on_cov  (권장)
        - penalty_norm="grid":    penalty = nonopen_ratio   (비오픈 전체 대비 비율)
        """
        bboxes = self.sh.bboxes_from_state_ll(state)
        cb = self.sh.coverage_breakdown(nx, ny, obs_pos, bboxes)

        if penalty_norm == "covered":
            penalty = cb["nonopen_on_cov"]
        elif penalty_norm == "grid":
            penalty = cb["nonopen_ratio"]
        else:
            raise ValueError("penalty_norm must be 'covered' or 'grid'")

        return (1.0 - cb["open_ratio"]) + lam * penalty

    def energy_open_obs_overlap_from_state(
        self,
        nx:int, ny:int, obs_pos, state,
        lam_obs: float = 0.3,     # 비오픈(장애물) 커버 패널티 가중치
        lam_ov:  float = 0.2,     # 겹침 패널티 가중치 (소프트)
        norm_obs: str = "covered",  # {"covered","grid"}: 비오픈 패널티 정규화 방식
        norm_ov:  str = "open"      # {"covered","open"}: 겹침 패널티 정규화 방식
    ):
        """
        E = (1 - open_ratio) + lam_obs * penalty_nonopen + lam_ov * penalty_overlap
        - open_ratio: 오픈 영역 커버율 (↑ good)
        - penalty_nonopen:
            * "covered" -> nonopen_on_cov (커버된 셀 중 비오픈 비율, 권장)
            * "grid"    -> nonopen_ratio  (그리드 전체 대비 비오픈 커버 비율)
        - penalty_overlap:
            * "open"    -> overlap_on_open (오픈 내에서 겹친 비율, 권장)
            * "covered" -> overlap_on_covered (커버 전체 중 겹침 비율)
        """
        bboxes = self.sh.bboxes_from_state_ll(state)

        # open / non-open
        cb = self.sh.coverage_breakdown(nx, ny, obs_pos, bboxes)
        open_ratio = cb["open_ratio"]
        if norm_obs == "covered":
            p_nonopen = cb["nonopen_on_cov"]
        elif norm_obs == "grid":
            p_nonopen = cb["nonopen_ratio"]
        else:
            raise ValueError("norm_obs must be 'covered' or 'grid'")

        # overlap
        ob = self.sh.overlap_breakdown(nx, ny, obs_pos, bboxes)
        if norm_ov == "open":
            p_overlap = ob["overlap_on_open"]
        elif norm_ov == "covered":
            p_overlap = ob["overlap_on_covered"]
        else:
            raise ValueError("norm_ov must be 'open' or 'covered'")

        return (1.0 - open_ratio) + lam_obs * p_nonopen + lam_ov * p_overlap

