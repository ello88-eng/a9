from dataclasses import dataclass
from vis_class import RegionToolkit
from state_helper_class import StateHelper
from state_adapter_class import StateAdapter
from energy_class import Energy
from greedy_class import GreedyRefine
from anneal_class import Anneal, SAConfig
import numpy as np
import time

@dataclass
class MinOnlyBounds:
    wmin: float = 10.0; hmin: float = 10.0

# --- (Optional) quick preview: uncomment to see all four shapes ---
if __name__ == "__main__":

    rt = RegionToolkit()
    sh = StateHelper()
    en = Energy()
    sa = StateAdapter()
    gr = GreedyRefine()
    an = Anneal()

    # Make area
    shapes = [
        ("rectangle", dict(nx=100, ny=100, rect_w=80, rect_h=80)),
        ("l",         dict(nx=100, ny=100)),
        ("y",         dict(nx=100, ny=100)),
        ("random",    dict(nx=100, ny=100, steps=400, seeds=2, seed=13)),
    ]

    # Parameter setting
    # TODO : FOV, line space part are required
    # TODO : MinOnlyBounds should be calculated from the area size and altitude
    # TODO : 'thetas' varibale should be expaned according to the 'num_avs'
    # TODO : box_size could set to be same because to consider a fair share.
    # TODO : altitude seperation 
    num_avs = 4
    thetas = np.zeros(num_avs)  # 박스별 회전 벡터 (초기 0)

    # box rotation constraints
    rot_step = 5.0   # 한 번에 회전할 각도(deg)
    full_rot_every    = 5      # 매 5 iteration마다 전체 앵커 회전 스윕 (0 또는 None이면 비활성)
    rotate_load_cap   = 10     # 회전 평가 최대 개수 상한 (None이면 비활성)
    rotate_sample_frac = 0.5   # 회전 평가할 대상 비율(0<frac≤1). 1.0이면 비활성

    # box size constraints
    size_step = 1.0          # 한 번에 키울 픽셀(셀) 수 (예: 1)
    max_w, max_h = None, None  # 필요하면 상한 지정 (예: nx, ny 등)

    # energy functions parameter
    lam_obs = 0.34
    lam_ov  = 0.0

    altitude = 500  # altitude
    bounds   = MinOnlyBounds(10, 10)  # Bbox minimum size constraints
    target_shape = "rectangle"
    time_limit   = 10  # seconds
    use_sa = True  # ← SA 사용 스위치 (False면 기존 그리디 루프 사용)

    # 선택된 모양 실행
    for name, kw in shapes:
        if name != target_shape:
            continue

        # 1) 영역 생성 & 앵커 샘플링
        nx, ny, obs_pos = rt.generate_region(name, **kw)
        anchors = sh.sample_anchors_in_open_space(
            nx, ny, obs_pos, num_avs=num_avs, min_spacing=None, seed=7
        )

        # 2) 초기 state (LL-앵커, 최소 크기, 회전 0)
        state = sh.make_state_ll(
            anchors, width=bounds.wmin, height=bounds.hmin, thetas_deg=thetas
        )

        # 3) 초기 에너지 (open↑, non-open/overlap↓)
        # E = energy_open_coverage_from_state(nx, ny, obs_pos, state)
        # E = energy_open_vs_obstacle_from_state(nx, ny, obs_pos, state, lam=lam_obs, penalty_norm="covered")
        E = en.energy_open_obs_overlap_from_state(
            nx, ny, obs_pos, state,
            lam_obs=lam_obs, lam_ov=lam_ov,
            norm_obs="covered", norm_ov="open"
        )
        print(f"[init] score ={1.0 - E:.4f}")

        # temp test ------------------------------------
        # bbox size cap margin
        open_mask = sh._obs_to_open_mask(nx, ny, obs_pos)
        sdf = sa.signed_distance(open_mask)   # ← 매 라운드 전에 1회 (open_mask가 바뀌면 재계산)
        ys, xs = np.where(open_mask)
        open_w_cap = int(xs.max() - xs.min() + 1)
        open_h_cap = int(ys.max() - ys.min() + 1)

        cap_margin = 1.05  # 5% 여유
        max_w = min(nx, int(open_w_cap * cap_margin))
        max_h = min(ny, int(open_h_cap * cap_margin))
        
        
        if use_sa: # simulated annealing
            # 1) 그리디로 워밍업 (시간 예산의 일부만 사용: 예 30%)
            greedy_frac = 0.50
            greedy_tlim = max(1.0, time_limit * greedy_frac)

            # 필요 변수: num_avs, rot_step, size_step, full_rot_every, rotate_load_cap, rotate_sample_frac, max_w, max_h
            state_g, E_g, t_greedy = gr.greedy_refine(
                nx, ny, obs_pos, state, bounds,
                lam_obs=lam_obs, lam_ov=lam_ov,
                num_avs=num_avs,
                rot_step=rot_step, size_step=size_step,
                max_w=max_w, max_h=max_h,
                patience=50,
                greedy_time_limit=greedy_tlim,
                full_rot_every=full_rot_every,
                rotate_load_cap=rotate_load_cap,
                rotate_sample_frac=rotate_sample_frac
            )
            print(f"[warm-start: greedy] score={1.0 - E_g:.4f}, elapsed={t_greedy:.2f}s")

            # 2) 남은 시간으로 SA 수행
            sa_tlim = max(1.0, time_limit - t_greedy)
            
            cfg = SAConfig(
                T0=0.9, Tmin=1e-3, alpha=0.98,
                p_move=0.6, p_rot=0.3, p_size=0.05,
                step_xy_max=2, step_th_max_deg=10.0, step_sz_max=1,
                gamma_xy=1.0, gamma_th=1.0, gamma_sz=1.0,
                shrink_enable_ratio=0.9,
                neighbor_retry=8,
                outer_rounds=90, inner_loops=200,
                stall_rounds=5, reheats_max=1, reheat_factor=1.5,
                seed=7
            )

            sa_best, sa_best_E, sa_stats = an.anneal(
                nx, ny, obs_pos, state_g, bounds, sdf,
                lam_obs=lam_obs, lam_ov=lam_ov,
                cfg=cfg, time_limit=time_limit,
                max_w=max_w, max_h=max_h, rng_seed=7
            )

            # 카운트 요약
            print("[SA] totals:",
                f"proposed={sa_stats['proposed']}, accepted={sa_stats['accepted']};",
                f"move={sa_stats['move']}/{sa_stats['move_acc']},",
                f"rotate={sa_stats['rotate']}/{sa_stats['rotate_acc']},",
                f"size={sa_stats['size']}/{sa_stats['size_acc']}")

            # === 안전 가드(최종 선택) — 이 줄들로 기존 state,E 할당을 교체 ===
            if sa_best_E < E_g - 1e-12:   # 작은 엡실론으로 동률 처리 안정화
                state, E = sa_best, sa_best_E
                chosen = "SA"
            else:
                state, E = state_g, E_g
                chosen = "Greedy (warm-start)"

            print(f"[select] chosen={chosen}  score={1.0 - E:.4f}")
        else: # greedy search
            # 순수 그리디만 (원래 else 블록을 사용하고 싶다면, 위 greedy_refine를 호출한 뒤 state,E만 반영)
            state, E, _ = gr.greedy_refine(
                nx, ny, obs_pos, state, bounds,
                lam_obs=lam_obs, lam_ov=lam_ov,
                num_avs=num_avs,
                rot_step=rot_step, size_step=size_step,
                max_w=max_w, max_h=max_h,
                patience=50,
                greedy_time_limit=time_limit,
                full_rot_every=full_rot_every,
                rotate_load_cap=rotate_load_cap,
                rotate_sample_frac=rotate_sample_frac
                )

        # 6) 결과 시각화 & 지표 출력
        bboxes_best = sh.bboxes_from_state_ll(state)
        ratio, _ = sh.coverage_ratio_open_space(nx, ny, obs_pos, bboxes_best)
        cb = sh.coverage_breakdown(nx, ny, obs_pos, bboxes_best)
        ob = sh.overlap_breakdown(nx, ny, obs_pos, bboxes_best)
        print(f"open={cb['open_ratio']:.3f}, nonopen_on_cov={cb['nonopen_on_cov']:.3f}, overlap_on_open={ob['overlap_on_open']:.3f}")

        rt.plot_rotated_bboxes(nx, ny, obs_pos, bboxes_best, f"Anchor-only | cov={ratio:.3f}")
        break  # target_shape 하나만 실행
