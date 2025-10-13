import numpy as np
from typing import List, Tuple, Optional, Union, Dict

Number = Union[int, float]

class StateHelper:
    def __init__(self):
        pass

    # state_helpers.py
    def _obs_to_open_mask(self, nx:int, ny:int, obs_pos:List[int]) -> np.ndarray:
        open_mask = np.ones((ny, nx), dtype=bool)  # True=open, False=obstacle
        for idx in obs_pos:
            r, c = divmod(idx, nx)
            open_mask[r, c] = False
        return open_mask

    def _auto_min_spacing(self, open_mask: np.ndarray, k: int) -> int:
        """오픈셀 면적과 점 개수로부터 대략적인 격자 최소 간격 추정."""
        open_cells = int(open_mask.sum())
        if k <= 1 or open_cells <= 1:
            return 1
        # 한 점이 차지하는 평균 면적의 루트 ≈ 간격
        spacing = int(np.sqrt(open_cells / k))
        return max(2, spacing)

    def _farthest_point_sampling(self, open_mask: np.ndarray,
                                k: int,
                                min_spacing: Optional[int] = None,
                                seed: int = 7) -> List[Tuple[int,int]]:
        """오픈셀에서 farthest-point 방식으로 k개 샘플링 (격자 간격 제한 지원)."""
        ys, xs = np.where(open_mask)
        N = len(ys)
        if N == 0 or k <= 0:
            return []
        rng = np.random.default_rng(seed)

        if min_spacing is None:
            min_spacing = self._auto_min_spacing(open_mask, k)

        pts = np.stack([ys, xs], axis=1)  # (N,2)
        # 시작점: 랜덤
        i0 = int(rng.integers(0, N))
        anchors = [tuple(pts[i0])]
        # 각 후보에 대해 "현재 선택된 앵커까지의 최소 제곱거리" 유지
        d2 = np.sum((pts - pts[i0])**2, axis=1)

        # helper: 간격 제약 검사
        def _ok_spacing(p, anchors_list, ms=min_spacing):
            py, px = p
            for (ay, ax) in anchors_list:
                if (py - ay)**2 + (px - ax)**2 < ms**2:
                    return False
            return True

        # 루프: 가장 먼 점을 고르되, 간격 불만족이면 후보 제외 → 반복
        while len(anchors) < k:
            j = int(np.argmax(d2))
            if d2[j] < 0:  # 더 이상 유효 후보 없음
                break
            cand = tuple(pts[j])
            if _ok_spacing(cand, anchors):
                anchors.append(cand)
                # 거리 업데이트(최소 거리 유지)
                d2 = np.minimum(d2, np.sum((pts - pts[j])**2, axis=1))
            else:
                # 간격 불만족: 해당 후보 영구 제외
                d2[j] = -1.0

            # 후보가 모두 소진되면 간격을 조금씩 완화
            if (d2 > 0).sum() == 0 and len(anchors) < k:
                min_spacing = max(1, min_spacing - 1)
                # 간격 완화 후 거리배열 초기화/갱신
                d2[:] = np.inf
                for (ay, ax) in anchors:
                    d2 = np.minimum(d2, np.sum((pts - np.array([ay, ax]))**2, axis=1))
                # 이미 간격 위반으로 제외했던 것들도 재검토되도록 음수는 0으로 리셋
                d2[d2 < 0] = 0.0

        return anchors[:k]

    def sample_anchors_in_open_space(self, nx:int, ny:int, obs_pos:List[int],
                                    num_avs:int,
                                    min_spacing: Optional[int] = None,
                                    seed:int = 7) -> List[Tuple[int,int]]:
        """
        오픈 공간에서 num_avs개의 anchor를 골고루 분산시켜 반환.
        - min_spacing: 격자 단위 최소 간격(없으면 자동 추정)
        - 반환 형식: [(row, col), ...]
        """
        open_mask = self._obs_to_open_mask(nx, ny, obs_pos)
        anchors = self._farthest_point_sampling(open_mask, num_avs, min_spacing=min_spacing, seed=seed)
        return anchors

    Number = Union[int, float]

    def _broadcast_param(self, x: Union[Number, List[Number]], k: int) -> List[Number]:
        """스칼라 -> 길이 k 리스트 브로드캐스트."""
        if isinstance(x, (int, float)):
            return [x] * k
        assert len(x) == k, f"Length mismatch: expected {k}, got {len(x)}"
        return list(x)

    def _rect_cover_mask(self, nx:int, ny:int, rect:Dict[str, float]) -> np.ndarray:
        """
        단일 회전 사각형에 의해 커버되는 셀(센터포인트 기준) 마스크 반환 (True=커버).
        rect: {'x0','y0','w','h','theta_deg'}, x0/y0는 좌측하단 기준점.
        """
        x0, y0, w, h, th = rect["x0"], rect["y0"], rect["w"], rect["h"], rect["theta_deg"]
        # 셀 중심 좌표 (x: 0.5..nx-0.5, y: 0.5..ny-0.5)
        xs = np.arange(nx) + 0.5
        ys = np.arange(ny) + 0.5
        X, Y = np.meshgrid(xs, ys)  # shape (ny, nx)

        # 앵커 기준 좌표로 이동 후, -theta 회전(직교 정렬로 투영)
        rad = np.deg2rad(th)
        ct, st = np.cos(rad), np.sin(rad)
        Xp = X - x0
        Yp = Y - y0
        # inverse rotate by theta: [x'; y'] = R(-theta) [Xp; Yp]
        Xr =  ct * Xp + st * Yp
        Yr = -st * Xp + ct * Yp

        inside = (Xr >= 0.0) & (Xr <= w) & (Yr >= 0.0) & (Yr <= h)
        return inside

    def coverage_ratio_open_space(self,
        nx:int, ny:int, obs_pos:List[int], bboxes:List[Dict[str, float]]
    ) -> Tuple[float, np.ndarray]:
        """
        박스 합집합이 오픈 스페이스에서 커버하는 비율(0~1) 반환.
        - 커버 판정은 셀 중심점이 박스 내부에 있는지로 계산.
        - 장애물 셀은 분모에서 제외.
        반환: (ratio, covered_union_mask)  # covered_union_mask는 전체 그리드 기준
        """
        open_mask = self._obs_to_open_mask(nx, ny, obs_pos)
        if open_mask.sum() == 0:
            return 0.0, np.zeros_like(open_mask, dtype=bool)

        union_mask = np.zeros_like(open_mask, dtype=bool)
        for rect in bboxes:
            union_mask |= self._rect_cover_mask(nx, ny, rect)

        covered_open = (union_mask & open_mask).sum()
        ratio = covered_open / float(open_mask.sum())
        return ratio, union_mask

    def coverage_breakdown(self, nx:int, ny:int, obs_pos, bboxes):
        """
        오픈/비오픈 커버 면적과 비율을 한 번에 계산.
        반환 dict:
        - open_ratio:      (오픈 영역 중 커버된 비율)
        - nonopen_ratio:   (비오픈(장애물/경계 밖 아님) 중 커버된 비율; 'grid' 정규화)
        - nonopen_on_cov:  (커버된 셀 중 비오픈 비율; 'covered' 정규화)
        - areas: dict(covered_open, covered_nonopen, total_open, total_nonopen, total_covered)
        - union_mask: 커버 union (시각화에 사용 가능)
        """
        open_mask = self._obs_to_open_mask(nx, ny, obs_pos)
        ratio_open, union_mask = self.coverage_ratio_open_space(nx, ny, obs_pos, bboxes)  # 이미 구현됨

        covered = union_mask.sum()
        total_open = int(open_mask.sum())
        total_nonopen = nx*ny - total_open

        covered_open = int((union_mask & open_mask).sum())
        covered_nonopen = int((union_mask & (~open_mask)).sum())

        # 정규화 방식 2종: grid 전체 기준 vs. "커버된 셀" 기준
        nonopen_ratio_grid = (covered_nonopen / total_nonopen) if total_nonopen > 0 else 0.0
        nonopen_ratio_on_cov = (covered_nonopen / covered) if covered > 0 else 0.0

        return {
            "open_ratio": ratio_open,
            "nonopen_ratio": nonopen_ratio_grid,
            "nonopen_on_cov": nonopen_ratio_on_cov,
            "areas": {
                "covered_open": covered_open,
                "covered_nonopen": covered_nonopen,
                "total_open": total_open,
                "total_nonopen": total_nonopen,
                "total_covered": covered
            },
            "union_mask": union_mask
        }

    def overlap_breakdown(self, nx:int, ny:int, obs_pos, bboxes):
        """
        겹침(둘 이상 박스가 덮은 셀) 비율을 계산.
        반환:
        - overlap_on_covered:  겹침셀 / (커버된 전체셀)         ← 커버 중 낭비 비율
        - overlap_on_open:     (겹침∧오픈) / (커버∧오픈)        ← 오픈 영역 내 겹침 비율
        - areas: dict(covered, overlap, covered_open, overlap_open)
        - union_mask: 커버 합집합 (시각화 용)
        """
        open_mask = self._obs_to_open_mask(nx, ny, obs_pos)
        ny_, nx_ = ny, nx
        if not bboxes:
            return {
                "overlap_on_covered": 0.0, "overlap_on_open": 0.0,
                "areas": {"covered": 0, "overlap": 0, "covered_open": 0, "overlap_open": 0},
                "union_mask": np.zeros((ny_, nx_), dtype=bool),
            }

        counts = np.zeros((ny_, nx_), dtype=np.uint16)
        for rect in bboxes:
            counts += self._rect_cover_mask(nx, ny, rect).astype(np.uint16)

        union_mask   = counts >= 1
        overlap_mask = counts >= 2

        covered         = int(union_mask.sum())
        overlap         = int(overlap_mask.sum())
        covered_open    = int((union_mask   & open_mask).sum())
        overlap_open    = int((overlap_mask & open_mask).sum())

        overlap_on_cov  = (overlap / covered)       if covered      > 0 else 0.0
        overlap_on_open = (overlap_open / covered_open) if covered_open > 0 else 0.0

        return {
            "overlap_on_covered": overlap_on_cov,
            "overlap_on_open": overlap_on_open,
            "areas": {
                "covered": covered,
                "overlap": overlap,
                "covered_open": covered_open,
                "overlap_open": overlap_open,
            },
            "union_mask": union_mask,
        }

    def build_bboxes_from_anchors_ll(
        self,
        anchors: List[Tuple[int,int]],
        widths: Union[Number, List[Number]],
        heights: Union[Number, List[Number]],
        thetas_deg: Union[Number, List[Number]]
    ) -> List[Dict[str, float]]:
        """
        앵커(행,열)를 박스의 '좌측하단(lower-left)' 코너로 놓고 회전 사각형 정의.
        좌표계: x=열(col), y=행(row), y는 아래에서 위로 증가(plt.imshow(origin='lower')와 일치)
        - anchor = (r, c) -> (x0, y0) = (c, r)
        - width: +x 방향, height: +y 방향
        - theta_deg: CCW(+), anchor(좌측하단) 기준 회전
        반환: [{'x0','y0','w','h','theta_deg'}, ...]
        """
        k = len(anchors)
        W = self._broadcast_param(widths, k)
        H = self._broadcast_param(heights, k)
        T = self._broadcast_param(thetas_deg, k)
        bboxes = []
        for (r, c), w, h, t in zip(anchors, W, H, T):
            bboxes.append({"x0": float(c), "y0": float(r), "w": float(w), "h": float(h), "theta_deg": float(t)})
        return bboxes

    def make_state_ll(
        self,
        anchors: List[Tuple[int,int]],
        width: float,
        height: float,
        thetas_deg: Union[float, List[float], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        LL-anchor 기준 state:
        - xy: (N,2) with [row(y), col(x)] for each anchor
        - w, h: 모든 박스 공통 스칼라 (float)
        - theta: (N,) 각도(deg), 박스별 회전
        """
        A = np.array([[r, c] for (r, c) in anchors], dtype=float)        # (N,2) = [y, x]
        if isinstance(thetas_deg, (int, float)):
            theta = np.full(len(anchors), float(thetas_deg), dtype=float)
        else:
            theta = np.asarray(thetas_deg, dtype=float)
            assert theta.shape[0] == len(anchors), "len(thetas_deg) must equal num anchors"
        return {"xy": A, "w": float(width), "h": float(height), "theta": theta}

    def bboxes_from_state_ll(self, state: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """
        state(LL) -> 회전 사각형 리스트(dict)
        dict: {'x0','y0','w','h','theta_deg'}
        """
        xy = state["xy"]        # (N,2) [y, x]
        w  = float(state["w"])
        h  = float(state["h"])
        th = state["theta"]     # (N,)
        bboxes = []
        for i in range(xy.shape[0]):
            y0, x0 = xy[i, 0], xy[i, 1]       # LL-anchor: (row -> y0, col -> x0)
            bboxes.append({"x0": float(x0), "y0": float(y0), "w": w, "h": h, "theta_deg": float(th[i])})
        return bboxes