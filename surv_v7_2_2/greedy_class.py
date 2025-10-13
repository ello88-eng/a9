
import time
from state_helper_class import StateHelper
from energy_class import Energy
from neighbor_class import Neighbor
import numpy as np

class GreedyRefine:
    def __init__(self):
        self.sh = StateHelper()
        self.en = Energy()
        self.nb = Neighbor()

    def greedy_refine(self, nx, ny, obs_pos, state_init, bounds,
                    lam_obs, lam_ov,
                    num_avs,
                    rot_step=5.0, size_step=1.0,
                    max_w=None, max_h=None,
                    patience=50, greedy_time_limit=3.0,
                    full_rot_every=5, rotate_load_cap=10, rotate_sample_frac=0.5):
        """
        기존 그리디 탐색을 짧게 돌려 초기 해를 만드는 함수.
        반환: (state_g, E_g, elapsed_s)
        """
        
        # 필요: _obs_to_open_mask / energy_open_obs_overlap_from_state / _try_* 함수들
        open_mask = self.sh._obs_to_open_mask(nx, ny, obs_pos)
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]

        # 상태/에너지 초기화
        state = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state_init.items()}
        E = self.en.energy_open_obs_overlap_from_state(
            nx, ny, obs_pos, state,
            lam_obs=lam_obs, lam_ov=lam_ov,
            norm_obs="covered", norm_ov="open"
        )

        max_iter = 2000
        no_improve = 0
        start_t = time.time()

        for it in range(max_iter):
            improved = False

            # Translation (셔플 + 그리디)
            order = np.random.permutation(num_avs)
            moved = []
            for idx in order:
                best_state = None
                best_E = E
                for dy, dx in dirs:
                    cand = self.nb._try_move_anchor_once(state, idx, dy, dx, nx, ny, open_mask)
                    if cand is None:
                        continue
                    Ec = self.en.energy_open_obs_overlap_from_state(
                        nx, ny, obs_pos, cand,
                        lam_obs=lam_obs, lam_ov=lam_ov,
                        norm_obs="covered", norm_ov="open"
                    )
                    if Ec < best_E - 1e-9:
                        best_E = Ec
                        best_state = cand
                if best_state is not None:
                    state = best_state
                    E = best_E
                    moved.append(int(idx))
                    improved = True

            # --- 회전 탐색: (현재 코드 동작과 동일) '이동 루프'의 마지막 idx만 회전 시도 ---
            # --- Rotation: 이번 iteration에서 "안 움직인" 앵커만 회전 시도 ---
            all_idx   = set(range(num_avs))
            to_rotate = list(all_idx - set(moved))

            # (A) 주기적 전체 회전 스윕
            if full_rot_every and ((it + 1) % full_rot_every == 0):
                to_rotate = list(range(num_avs))

            # (B) 아무도 없으면 기존 fallback 유지(이동 루프의 마지막 idx만 회전)
            if not to_rotate:
                to_rotate = [int(order[-1])]

            # (C) 부하 제한: 샘플링으로 대상 수 축소 (cap과 frac을 함께 적용)
            L = len(to_rotate)
            if L > 1 and (rotate_load_cap is not None or rotate_sample_frac < 1.0):
                k_frac = int(np.ceil(L * rotate_sample_frac)) if rotate_sample_frac < 1.0 else L
                k_cap  = rotate_load_cap if rotate_load_cap is not None else L
                k = max(1, min(L, k_frac, k_cap))
                if k < L:
                    to_rotate = list(np.random.choice(to_rotate, size=k, replace=False))

            # (선택) 회전 대상 순서도 섞어서 편향 줄이기
            to_rotate = list(np.random.permutation(to_rotate))

            for idx_rot in to_rotate:
                local_best_state = None
                local_best_E = E
                for dth in (rot_step, -rot_step):
                    cand = self.nb._try_rotate_once(state, idx_rot, dth, wrap="360")
                    Ec = self.en.energy_open_obs_overlap_from_state(
                        nx, ny, obs_pos, cand,
                        lam_obs=lam_obs, lam_ov=lam_ov,
                        norm_obs="covered", norm_ov="open"
                    )
                    if Ec < local_best_E - 1e-9:
                        local_best_E = Ec
                        local_best_state = cand
                if local_best_state is not None:
                    state = local_best_state
                    E = local_best_E
                    improved = True

            # 전역 사이즈 증가 (세 후보 중 최선 1회)
            best_state_size = None
            best_E_size = E
            for dw, dh in [(size_step, 0.0), (0.0, size_step), (size_step, size_step)]:
                cand = self.nb._try_resize_global_once(state, dw, dh, bounds,
                                            only_grow=True, max_w=max_w, max_h=max_h)
                if cand is None:
                    continue
                Ec = self.en.energy_open_obs_overlap_from_state(
                    nx, ny, obs_pos, cand,
                    lam_obs=lam_obs, lam_ov=lam_ov,
                    norm_obs="covered", norm_ov="open"
                )
                if Ec < best_E_size - 1e-9:
                    best_E_size = Ec
                    best_state_size = cand
            if best_state_size is not None:
                state = best_state_size
                E = best_E_size
                improved = True

            # 중단 조건
            no_improve = 0 if improved else (no_improve + 1)
            if no_improve >= patience:
                break
            if time.time() - start_t > greedy_time_limit:
                break

        elapsed_s = time.time() - start_t
        return state, E, elapsed_s
