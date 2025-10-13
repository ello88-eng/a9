from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional
from neighbor_class import Neighbor
from state_helper_class import StateHelper
from energy_class import Energy
import time

# === Simulated Annealing utilities (vanilla) ===
@dataclass
class SAConfig:
    # temperature
    T0: float = 1.0
    Tmin: float = 1e-3
    alpha: float = 0.98  # geometric cooling

    # neighbor probs
    p_move: float = 0.6
    p_rot:  float = 0.3
    p_size: float = 0.1

    # step maxima (will shrink with T)
    step_xy_max: int = 2
    step_th_max_deg: float = 10.0
    step_sz_max: int = 2

    # step decay exponents (T/T0)^gamma
    gamma_xy: float = 1.0
    gamma_th: float = 1.0
    gamma_sz: float = 1.0

    # allow shrink of w/h when T/T0 < ratio
    shrink_enable_ratio: float = 0.5

    # retry when neighbor invalid
    neighbor_retry: int = 8

    # SA loop sizes (기본 예산)
    outer_rounds: int = 30
    inner_loops:  int = 200

    # stall/reheat
    stall_rounds: int = 5
    reheats_max:  int = 1
    reheat_factor: float = 1.5

    seed: int = 7

class Anneal:
    def __init__(self):
        self.nb = Neighbor()
        self.sh = StateHelper()
        self.en = Energy()

    def cool_geometric(self, T: float, alpha: float) -> float:
        return max(T * alpha, 1e-12)

    def metropolis_accept(self, dE: float, T: float, rng: np.random.Generator) -> bool:
        if dE <= 0.0:
            return True
        if T <= 0.0:
            return False
        return rng.random() < np.exp(-dE / T)

    def _copy_state(self, state):
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()}

    def propose_neighbor(self, state, nx, ny, open_mask, bounds, sdf,
                        max_w=None, max_h=None, lam_ov=0.0,
                        T: float = 1.0, T0: float = 1.0, 
                        cfg: SAConfig = None,
                        rng: np.random.Generator = None) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Return (cand_state, meta) or (None, None).
        - move: 임의 앵커 1개 (±Δy, ±Δx) * step_xy (open cell 강제)
        - rotate: 임의 앵커 1개 ±Δθ
        - size: (dw,0)/(0,dh)/(dw,dh) 중 하나, 낮은 T에서 shrink 허용
        """
        assert cfg is not None
        rng = rng or np.random.default_rng(cfg.seed)

        ratio = max(1e-9, min(1.0, T / T0))
        step_xy = max(1, int(round(cfg.step_xy_max * (ratio**cfg.gamma_xy))))
        step_th = max(2.0, float(cfg.step_th_max_deg * (ratio**cfg.gamma_th)))
        step_sz = max(1, int(round(cfg.step_sz_max * (ratio**cfg.gamma_sz))))

        probs = np.array([cfg.p_move, cfg.p_rot, cfg.p_size], dtype=float)
        probs = probs / probs.sum()
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        for _ in range(cfg.neighbor_retry):
            op = rng.choice([0,1,2], p=probs)  # 0:move, 1:rotate, 2:size

            if op == 0:  # move
                cand = self._copy_state(state)
                i = int(rng.integers(0, cand["xy"].shape[0]))
                dy, dx = dirs[int(rng.integers(0, len(dirs)))]
                ok = True
                for _step in range(step_xy):
                    nxt = self.nb._try_move_anchor_once(cand, i, dy, dx, nx, ny, open_mask, nxny_clip=True)
                    # 루프 안에서 각 박스 i에 대해:
                    ok_move, meta_m = self.nb._try_move_anchor_once_sdf(
                        state, i, nx, ny, open_mask, sdf,
                        init_step=6.0, beta=0.5, lam_ov=lam_ov  # lam_ov는 네 구성에 맞게
                    )
                    if ok_move:
                        # state 가 수정되므로, cand 업데이트
                        cand = state

                    
                    if nxt is None:
                        ok = False
                        break
                    cand = nxt
                if ok:
                    return cand, {"op":"move", "idx":i, "dy":dy*step_xy, "dx":dx*step_xy}

            elif op == 1:  # rotate
                cand = self._copy_state(state)
                i = int(rng.integers(0, cand["xy"].shape[0]))
                dth = (+step_th) if rng.random() < 0.5 else (-step_th)
                cand = self.nb._try_rotate_once(cand, i, dth, wrap="360")
                return cand, {"op":"rotate", "idx":i, "dth":dth}

            else:  # size
                cand = self._copy_state(state)
                allow_shrink = (ratio < cfg.shrink_enable_ratio)
                # 감소 확률은 30% 정도로 보수적
                sign_w = (1 if (not allow_shrink or rng.random()<0.7) else -1)
                sign_h = (1 if (not allow_shrink or rng.random()<0.7) else -1)
                mode = rng.choice(["w","h","both"], p=[0.4, 0.4, 0.2])
                dw = sign_w * step_sz if mode in ("w","both") else 0
                dh = sign_h * step_sz if mode in ("h","both") else 0
                if dw == 0 and dh == 0:
                    continue
                nxt = self.nb._try_resize_global_once(
                    cand, dw, dh, bounds,
                    only_grow=not allow_shrink,
                    max_w=max_w, max_h=max_h
                )
                if nxt is not None:
                    cand = nxt
                    return cand, {"op":"size", "dw":dw, "dh":dh, "mode":mode, "shrink":allow_shrink}

        return None, None

    def anneal(self, nx:int, ny:int, obs_pos, state0, bounds, sdf, 
            lam_obs:float=0.3, lam_ov:float=0.5,
            cfg: SAConfig = None,
            time_limit: float = 10.0,
            max_w=None, max_h=None,
            rng_seed: int = 7):
        """
        Full SA loop (vanilla).
        Returns: best_state, best_E, stats(dict)
        """
        assert cfg is not None
        rng = np.random.default_rng(rng_seed)
        open_mask = self.sh._obs_to_open_mask(nx, ny, obs_pos)

        state = self._copy_state(state0)
        E = self.en.energy_open_obs_overlap_from_state(nx, ny, obs_pos, state,
                                            lam_obs=lam_obs, lam_ov=lam_ov,
                                            norm_obs="covered", norm_ov="open")
        best_state = self._copy_state(state)
        best_E = E

        T = cfg.T0
        start = time.time()
        timeup = False
        stall = 0
        reheats = 0

        # 카운트/로그
        stats = dict(
            proposed=0, accepted=0,
            move=0, move_acc=0,
            rotate=0, rotate_acc=0,
            size=0, size_acc=0,
            rounds=[]
        )

        for r in range(cfg.outer_rounds):
            round_prop = 0
            round_acc  = 0
            round_best_E = E

            # FIX: 라운드별 진단 카운터 초기화
            round_better = 0
            round_worse = 0
            round_worse_sum = 0.0

            for _ in range(cfg.inner_loops):
                if (time.time() - start) > time_limit:
                    timeup = True
                    break

                cand, meta = self.propose_neighbor(
                    state, nx, ny, open_mask, bounds, sdf, 
                    max_w=max_w, max_h=max_h, lam_ov=lam_ov,
                    T=T, T0=cfg.T0, cfg=cfg, rng=rng
                )
                if cand is None:
                    continue

                stats["proposed"] += 1
                round_prop += 1
                stats[meta["op"]] += 1

                Ec = self.en.energy_open_obs_overlap_from_state(nx, ny, obs_pos, cand,
                                                        lam_obs=lam_obs, lam_ov=lam_ov,
                                                        norm_obs="covered", norm_ov="open")
                dE = float(Ec - E)

                tau_w = 0.01   # 0.2~0.5 사이로 시작해서 로그 보며 튠
                T_eff = T * (tau_w if dE > 0.0 else 1.0) # 결과 악화됐을 때도 수용하는 비율 조정
                if self.metropolis_accept(dE, T_eff, rng):
                    if dE <= 0.0:
                        round_better += 1
                    else:
                        round_worse  += 1
                        round_worse_sum += dE
                    state, E = cand, Ec
                    stats["accepted"] += 1
                    round_acc += 1
                    stats[meta["op"]+"_acc"] += 1
                    

                    if E < best_E - 1e-12:
                        best_E = E
                        best_state = self._copy_state(state)

            cov = 1.0 - E
            acc_rate = (round_acc / round_prop) if round_prop > 0 else 0.0
            worse_ratio   = (round_worse / round_acc) if round_acc > 0 else 0.0
            avg_worse_dE  = (round_worse_sum / round_worse) if round_worse > 0 else 0.0
            print(
                f"[SA] round={r+1}/{cfg.outer_rounds}  "
                f"T={T:.4f}  score={cov:.4f}  "
                f"acc={round_acc}/{round_prop}({acc_rate:.2%})  "
                f"worse_acc={round_worse}/{round_acc}({worse_ratio:.2%})  "
                f"avg_worse_dE={avg_worse_dE:.4f}"
            )

            # 누적 stats 업데이트
            stats.setdefault("accepted_better", 0)
            stats.setdefault("accepted_worse", 0)
            stats.setdefault("sum_worse_dE", 0.0)
            stats["accepted_better"] += round_better
            stats["accepted_worse"]  += round_worse
            stats["sum_worse_dE"]    += round_worse_sum

            stats["rounds"].append(dict(
                T=T, coverage=cov,
                proposed=round_prop, accepted=round_acc, acc_rate=acc_rate,
                accepted_better=round_better, accepted_worse=round_worse,
                avg_worse_dE=avg_worse_dE
            ))

            # stall check (이 라운드 시작 대비 개선 여부)
            if E < round_best_E - 1e-12:
                stall = 0
            else:
                stall += 1

            # 냉각
            T = self.cool_geometric(T, cfg.alpha)

            # 조기 종료/재가열
            if T < cfg.Tmin:
                print("[SA] reached Tmin")
                break
            if timeup:
                print("[SA] time limit reached")
                break
            if stall >= cfg.stall_rounds and reheats < cfg.reheats_max:
                reheats += 1
                stall = 0
                T = min(cfg.T0, T * cfg.reheat_factor)
                print(f"[SA] reheating #{reheats}: T -> {T:.4f}")

        return best_state, best_E, stats