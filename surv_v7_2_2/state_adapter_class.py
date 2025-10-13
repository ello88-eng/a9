# ===== PATCH A: state-adapter + SDF/유틸 (LL-anchor) =====
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

class StateAdapter:
    def __init__(self):
        pass

    # --- 0) STATE ADAPTER (LL = left-lower corner) ---
    def get_box_from_state_ll_centered(self, state: Dict, i: int) -> Tuple[float,float,float,float,float]:
        """
        state:
        - state["xy"][i] = [row(y), col(x)]  (LL anchor)
        - state["w"], state["h"] = float (공통)
        - state["theta"][i] = degree (per box)
        return: (ax, ay, w, h, theta_rad)
        ax = x = col, ay = y = row  (배열좌표)
        """
        ay = float(state["xy"][i, 0])  # row
        ax = float(state["xy"][i, 1])  # col
        w  = float(state["w"])
        h  = float(state["h"])
        th = float(np.deg2rad(state["theta"][i]))  # 내부 계산은 rad
        return ax, ay, w, h, th

    def set_box_to_state_ll_centered(self, state: Dict, i: int, box: Tuple[float,float,float,float,float]) -> Dict:
        """
        이동 전용 세터: xy만 갱신. (회전/리사이즈 단계에서 w/h/theta 갱신 가능)
        box: (ax, ay, w, h, theta_rad)
        """
        ax, ay, w, h, th = box
        state["xy"][i, 0] = float(ay)  # row
        state["xy"][i, 1] = float(ax)  # col
        return state

    # --- 1) Geometry ---
    def rotmat(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    def sample_bbox_perimeter_ll(self, ax, ay, w, h, theta, samples_per_side=60) -> np.ndarray:
        """
        LL(anchor) 기준 회전 사각형 테두리 샘플 (배열좌표, (x=col, y=row))
        """
        xs = np.linspace(0, w, samples_per_side, endpoint=False)
        ys = np.linspace(0, h, samples_per_side, endpoint=False)
        bottom = np.stack([xs, np.zeros_like(xs)], axis=1)
        right  = np.stack([np.full_like(ys, w), ys], axis=1)
        top    = np.stack([xs[::-1], np.full_like(xs, h)], axis=1)
        left   = np.stack([np.zeros_like(ys), ys[::-1]], axis=1)
        local  = np.vstack([bottom, right, top, left]).astype(np.float32)
        R = self.rotmat(theta)
        world = (local @ R.T) + np.array([ax, ay], dtype=np.float32)
        return world  # (4N,2) with (x,y)

    # --- 2) SDF / Gradient / Bilinear ---
    def signed_distance(self, open_mask: np.ndarray) -> np.ndarray:
        m = (open_mask > 0).astype(np.uint8)  # 1=open, 0=obstacle
        inside  = edt(m)
        outside = edt(1 - m)
        return (inside - outside).astype(np.float32)  # open:+, obstacle:-

    def sdf_gradient(self, sdf: np.ndarray):
        gy, gx = np.gradient(sdf)  # numpy는 (y,x) 순
        return gx.astype(np.float32), gy.astype(np.float32)

    def bilinear_sample(self, arr: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        x = np.clip(pts_xy[:,0], 0, W-1)
        y = np.clip(pts_xy[:,1], 0, H-1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.clip(x0+1, 0, W-1);  y1 = np.clip(y0+1, 0, H-1)
        dx = x - x0;                 dy = y - y0
        v00 = arr[y0, x0]; v10 = arr[y0, x1]
        v01 = arr[y1, x0]; v11 = arr[y1, x1]
        v0 = v00*(1-dx) + v10*dx
        v1 = v01*(1-dx) + v11*dx
        return v0*(1-dy) + v1*dy

    # --- 3) Push dir & Energy ---
    def push_direction_from_sdf_ll(self, sdf: np.ndarray, ax, ay, w, h, theta,
                                tau: float = 2.0, samples_per_side: int = 60) -> np.ndarray:
        pts = self.sample_bbox_perimeter_ll(ax, ay, w, h, theta, samples_per_side)
        gx, gy = self.sdf_gradient(sdf)
        d   = self.bilinear_sample(sdf, pts)
        gxx = self.bilinear_sample(gx,  pts)
        gyy = self.bilinear_sample(gy,  pts)

        wgt = np.zeros_like(d)
        inside = d < 0
        wgt[inside] = 1.0 + (-d[inside])        # 침범 깊을수록↑
        near = (d >= 0) & (d < tau)
        wgt[near] = (tau - d[near]) / tau       # 경계 근처↑
        if not np.any(wgt > 0):
            return np.zeros(2, dtype=np.float32)

        vx = np.sum(wgt * gxx) / np.sum(wgt)
        vy = np.sum(wgt * gyy) / np.sum(wgt)
        return np.array([vx, vy], dtype=np.float32)   # (x, y) 배열좌표

    def soft_intrusion_penalty_ll(self, sdf: np.ndarray, ax, ay, w, h, theta,
                                tau: float = 2.0, samples_per_side: int = 60) -> float:
        pts = self.sample_bbox_perimeter_ll(ax, ay, w, h, theta, samples_per_side)
        d = self.bilinear_sample(sdf, pts)
        return float(np.mean(np.log1p(np.exp(-(d/tau)))))  # softplus(-d/tau)

    def clamp_box_ll(self, ax, ay, w, h, nx, ny):
        """
        간단한 LL 앵커 클램프(화면 밖으로 나가지 않게).
        회전 모서리 돌출은 에너지로 제어(필요 시 강화 가능).
        """
        ax = float(np.clip(ax, 0, nx-1))
        ay = float(np.clip(ay, 0, ny-1))
        return ax, ay, w, h

 ##############
    # ---------------------------
    # (A) Overlap 근사 유틸(LL)
    # ---------------------------
    def _rotmat(self, theta: float):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s],[s, c]], dtype=np.float32)

    def _points_in_ll_box_mask(self, pts_xy: np.ndarray, box_ll) -> np.ndarray:
        """
        pts_xy: (N,2) with (x=col, y=row)
        box_ll: (ax, ay, w, h, th) where th(rad)
        LL anchor 기준 로컬좌표로 보냈을 때 0<=x<=w, 0<=y<=h 인지 테스트.
        """
        ax, ay, w, h, th = box_ll
        R = self._rotmat(th)
        Rt = R.T
        local = (pts_xy - np.array([ax, ay], dtype=np.float32)) @ Rt
        return (local[:,0] >= 0) & (local[:,0] <= w) & (local[:,1] >= 0) & (local[:,1] <= h)

    def _overlap_penalty_for_ll(self, box_i, boxes_ll, idx_i: int, samples: int = 200) -> float:
        """
        간단한 overlap 근사:
        - 박스 i 내부를 균등 샘플
        - 다른 박스에도 포함되는 샘플 비율을 페널티로 사용
        """
        ax, ay, w, h, th = box_i
        if w < 2 or h < 2:
            return 0.0

        # i 박스 내부에서 균등 샘플 생성 (로컬 → 월드)
        xs = np.random.rand(samples) * w
        ys = np.random.rand(samples) * h
        pts_local = np.stack([xs, ys], axis=1).astype(np.float32)
        R = self._rotmat(th)
        pts = (pts_local @ R.T) + np.array([ax, ay], dtype=np.float32)

        ov = 0
        for j, bj in enumerate(boxes_ll):
            if j == idx_i: 
                continue
            ov += np.count_nonzero(self._points_in_ll_box_mask(pts, bj))
        return float(ov) / float(samples)

    # ---------------------------
    # (B) Local Energy with LL
    # ---------------------------
    def _local_energy_for_box(self, state, i, sdf, lam_ov: float = 0.0):
        """
        로컬 에너지 = 침범(필수) + lam_ov * overlap(옵션, 전역 근사)
        """
        # 현재 박스(LL)
        box_i = self.get_box_from_state_ll_centered(state, i)
        e = self.soft_intrusion_penalty_ll(sdf, *box_i, tau=2.0, samples_per_side=60)

        if lam_ov > 0.0:
            # 전체 박스를 LL 포맷으로 빌드 (현재 state 구조에 맞춤)
            N = state["xy"].shape[0]
            boxes_ll = [ self.get_box_from_state_ll_centered(state, k) for k in range(N) ]
            e += lam_ov * self._overlap_penalty_for_ll(box_i, boxes_ll, i, samples=200)

        return e

    def _backtracking_accept(self, currE, cand_box_ll, state, i, sdf, lam_ov,
                            c: float = 1e-3, step: float = 1.0, dir_norm: float = 1.0):
        """
        Armijo-like sufficient decrease 체크
        cand_box_ll: (ax, ay, w, h, th) with th(rad)
        """
        # 침범
        newE = self.soft_intrusion_penalty_ll(sdf, *cand_box_ll, tau=2.0, samples_per_side=60)

        # 겹침(옵션)
        if lam_ov > 0.0:
            N = state["xy"].shape[0]
            boxes_ll = [ self.get_box_from_state_ll_centered(state, k) for k in range(N) ]
            newE += lam_ov * self._overlap_penalty_for_ll(cand_box_ll, boxes_ll, i, samples=200)

        return (newE <= currE - c * step * dir_norm), newE