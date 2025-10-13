import numpy as np
from state_adapter_class import StateAdapter

class Neighbor:
    def __init__(self):
        self.sa = StateAdapter()

    def _anchor_is_open(self, open_mask, y, x):
        ny, nx = open_mask.shape
        iy, ix = int(round(y)), int(round(x))
        if iy < 0 or iy >= ny or ix < 0 or ix >= nx:
            return False
        return open_mask[iy, ix]

    def _try_move_anchor_once_sdf(self, state, i, nx, ny, open_mask, sdf,
                              init_step: float = 6.0, beta: float = 0.5, lam_ov: float = 0.0,
                              coord_delta: float = 3.0, armijo_c: float = 1e-3, max_bt: int = 10):
        """
        LL 이동: SDF-based push + 좌표탐색(±δ) 합성 → Armijo 백트래킹으로 단조 감소 수용
        (기존 LL=center 버전을 LL-anchor 버전으로 완전히 교체)
        """
        # 현재 박스(LL)
        ax, ay, w, h, th = self.sa.get_box_from_state_ll_centered(state, i)
        currE = self.sa._local_energy_for_box(state, i, sdf, lam_ov)

        # push 벡터 (경계/침범 보정)
        v_push = self.sa.push_direction_from_sdf_ll(sdf, ax, ay, w, h, th, tau=2.0, samples_per_side=60)

        # 좌표탐색(±δ) in (x=col, y=row)
        dirs = [np.array([+coord_delta,0],np.float32), np.array([-coord_delta,0],np.float32),
                np.array([0,+coord_delta],np.float32), np.array([0,-coord_delta],np.float32)]
        best_dir = np.zeros(2, np.float32); bestE = currE
        for d in dirs:
            ax2, ay2, w2, h2 = self.sa.clamp_box_ll(ax + d[0], ay + d[1], w, h, nx, ny)
            cand = (ax2, ay2, w2, h2, th)
            ok_probe, newE = self.sa._backtracking_accept(currE, cand, state, i, sdf, lam_ov,
                                                c=0.0, step=1.0, dir_norm=np.linalg.norm(d))
            if newE < bestE:
                bestE, best_dir = newE, d

        # 방향 합성
        v = v_push
        if np.linalg.norm(best_dir) > 0:
            v = 0.7*v_push + 0.3*(best_dir/np.linalg.norm(best_dir)) * coord_delta

        if np.allclose(v, 0):
            return False, {"why": "zero_push"}

        # 백트래킹
        vnorm = float(np.linalg.norm(v))
        step  = init_step
        tried = 0
        while step > 0.5 and tried < max_bt:
            ax2, ay2, w2, h2 = self.sa.clamp_box_ll(ax + step*v[0], ay + step*v[1], w, h, nx, ny)
            cand = (ax2, ay2, w2, h2, th)
            ok, newE = self.sa._backtracking_accept(currE, cand, state, i, sdf, lam_ov,
                                            c=armijo_c, step=step, dir_norm=vnorm)
            if ok:
                self.sa.set_box_to_state_ll_centered(state, i, cand)
                return True, {"gain": currE - newE, "step": step}
            step *= beta; tried += 1

        return False, {"why":"no_armijo"}

    def _try_move_anchor_once(self, state, i, dy, dx, nx, ny, open_mask, nxny_clip=True):
        """
        state: {"xy": (N,2)[y,x], "w": float, "h": float, "theta": (N,)}
        i번째 앵커를 (dy,dx)만큼 움직인 후보 state 반환. (불가하면 None)
        """
        s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
        y, x = s["xy"][i]
        y2, x2 = y + dy, x + dx
        if nxny_clip:
            y2 = float(np.clip(y2, 0, ny-1))
            x2 = float(np.clip(x2, 0, nx-1))
        if not self._anchor_is_open(open_mask, y2, x2):
            return None
        s["xy"][i, 0] = y2
        s["xy"][i, 1] = x2
        return s

    def _get_min_wh(self, bounds):
        # MinOnlyBounds(wmin,hmin) 또는 (w,h) / dict(min_w,min_h) 호환
        if hasattr(bounds, "wmin") and hasattr(bounds, "hmin"):
            return float(bounds.wmin), float(bounds.hmin)
        if hasattr(bounds, "min_w") and hasattr(bounds, "min_h"):
            return float(bounds.min_w), float(bounds.min_h)
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            return float(bounds[0]), float(bounds[1])
        if isinstance(bounds, dict) and "min_w" in bounds and "min_h" in bounds:
            return float(bounds["min_w"]), float(bounds["min_h"])
        return 5.0, 5.0

    def _try_resize_global_once(self, state, dw, dh, bounds, only_grow=True, max_w=None, max_h=None):
        """
        state의 전역 width/height를 (dw, dh)만큼 변경한 후보 반환 (개선 여지 없으면 None).
        - only_grow=True 이면 증가만 허용 (감소 금지)
        - bounds: 최소 크기 보장 (w>=wmin, h>=hmin)
        - max_w/max_h 주면 상한도 보장
        """
        s = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()}
        wmin, hmin = self._get_min_wh(bounds)

        new_w = s["w"] + float(dw)
        new_h = s["h"] + float(dh)

        if only_grow:
            new_w = max(new_w, s["w"])
            new_h = max(new_h, s["h"])

        new_w = max(new_w, wmin)
        new_h = max(new_h, hmin)

        if max_w is not None:
            new_w = min(new_w, float(max_w))
        if max_h is not None:
            new_h = min(new_h, float(max_h))

        # 변화가 없다면 후보 없음
        if abs(new_w - s["w"]) < 1e-9 and abs(new_h - s["h"]) < 1e-9:
            return None

        s["w"] = float(new_w)
        s["h"] = float(new_h)
        return s

    def _try_rotate_once(self, state, i, dtheta_deg, wrap="360"):
        """
        i번째 박스의 회전을 dtheta_deg 만큼 변경한 후보 state 반환.
        wrap: "360" → [0,360)로 정규화 / "180" → [-90,90]로 정규화 / None → 무제한
        """
        s = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()}
        s["theta"][i] = float(s["theta"][i] + float(dtheta_deg))

        if wrap == "360":
            # 0~360 범위로
            s["theta"][i] = s["theta"][i] % 360.0
        elif wrap == "180":
            # -90~90 범위로 (boustrophedon 등 180주기 문제에 유용)
            th = ((s["theta"][i] + 90.0) % 180.0) - 90.0
            s["theta"][i] = th
        # wrap=None 이면 그대로 둠

        return s