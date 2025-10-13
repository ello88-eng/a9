import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import os

class RegionToolkit:
    """Grid 기반 영역 생성 및 시각화 유틸"""
    def __init__(self):
        pass

    def generate_region(self, shape: str,
                        **kwargs) -> Tuple[int, int, List[int]]:
        """
        Generate a grid region and return (nx, ny, obs_pos).
        Shapes:
        - "rectangle": central open rectangle; rest obstacles
            kwargs: rect_w (int), rect_h (int), margin_x (int), margin_y (int)
        - "l": L-shaped open area; rest obstacles
            kwargs: vx0,vx1,vy0,vy1 (vertical leg), hx0,hx1,hy0,hy1 (horizontal leg)
        - "y": Y-shaped open area; rest obstacles
            kwargs: cx, cy (center), stem_h, arm_len, arm_thick, stem_thick, slope (float)
        - "random": random organic open blob via drunkard walk (+ optional dilation)
            kwargs: steps (int), seeds (int), jitter_prob (float), seed (int),
                    thicken_iters (int), thicken_neighbors ({"4","8"})
        """
        shape = shape.lower()
        nx = kwargs.get("nx")
        ny = kwargs.get("ny")

        open_mask = np.zeros((ny, nx), dtype=bool)  # False = obstacle, True = open

        if shape == "rectangle":
            rect_w  = kwargs.get("rect_w", max(4, nx // 2))
            rect_h  = kwargs.get("rect_h", max(4, ny // 2))
            margin_x = kwargs.get("margin_x", (nx - rect_w) // 2)
            margin_y = kwargs.get("margin_y", (ny - rect_h) // 2)
            x0 = np.clip(margin_x, 0, nx-1); y0 = np.clip(margin_y, 0, ny-1)
            x1 = np.clip(x0 + rect_w - 1, 0, nx-1)
            y1 = np.clip(y0 + rect_h - 1, 0, ny-1)
            open_mask[y0:y1+1, x0:x1+1] = True

        elif shape == "l":
            # --- Bigger, ratio-based defaults (targeting ~70–85% open when nx=ny=100) ---
            # vertical leg thickness & span
            v_w   = kwargs.get("v_w",   max(20, int(round(0.24 * nx))))
            v_x0  = kwargs.get("v_x0",  int(round(0.20 * nx)))
            v_y0  = kwargs.get("v_y0",  int(round(0.10 * ny)))
            v_y1  = kwargs.get("v_y1",  int(round(0.90 * ny)))
            # horizontal leg thickness & span
            h_h   = kwargs.get("h_h",   max(20, int(round(0.24 * ny))))
            h_y1  = kwargs.get("h_y1",  int(round(0.70 * ny)))  # bottom of horizontal band
            h_y0  = kwargs.get("h_y0",  max(0, h_y1 - h_h))
            h_x0  = kwargs.get("h_x0",  v_x0 + v_w)
            h_x1  = kwargs.get("h_x1",  int(round(0.92 * nx)))

            vx0, vx1 = np.clip(sorted([v_x0, v_x0 + v_w - 1]), 0, nx-1)
            vy0, vy1 = np.clip(sorted([v_y0, v_y1]),            0, ny-1)
            hx0, hx1 = np.clip(sorted([h_x0, h_x1]),            0, nx-1)
            hy0, hy1 = np.clip(sorted([h_y0, h_y1]),            0, ny-1)

            open_mask[vy0:vy1+1, vx0:vx1+1] = True
            open_mask[hy0:hy1+1, hx0:hx1+1] = True

            # kwargs 호환(기존 사용하던 vx0~hy1 직접 지정도 지원)
            for k, v in dict(vx0=kwargs.get("vx0"), vx1=kwargs.get("vx1"),
                            vy0=kwargs.get("vy0"), vy1=kwargs.get("vy1"),
                            hx0=kwargs.get("hx0"), hx1=kwargs.get("hx1"),
                            hy0=kwargs.get("hy0"), hy1=kwargs.get("hy1")).items():
                if v is not None:
                    # 사용자가 지정하면 위 설정을 덮어쓰고 다시 채움
                    locals()[k] = int(np.clip(v, 0, nx-1 if k.endswith(('x0','x1')) else ny-1))
            # (사용자가 직접 좌표를 모두 넣었을 경우) 다시 그리기
            # (이미 True로 칠했더라도 재칠해도 무방)
            # open_mask[vy0:vy1+1, vx0:vx1+1] = True
            # open_mask[hy0:hy1+1, hx0:hx1+1] = True

        elif shape == "y":
            # --- Bigger Y: thicker stem & longer arms by ratios ---
            cx = kwargs.get("cx", nx//2)
            cy = kwargs.get("cy", ny//2)
            stem_h     = kwargs.get("stem_h",     max(8, int(round(0.38 * ny))))
            stem_thick = kwargs.get("stem_thick", max(2, int(round(0.30 * nx))))
            arm_len    = kwargs.get("arm_len",    max(6, int(round(0.48 * min(nx, ny)))))
            arm_thick  = kwargs.get("arm_thick",  max(2, int(round(0.30 * nx))))
            slope      = kwargs.get("slope", 0.6)

            yy, xx = np.mgrid[0:ny, 0:nx]
            # Vertical stem (upwards from center)
            stem = (np.abs(xx - cx) <= stem_thick//2) & (yy >= cy) & (yy <= min(ny-1, cy + stem_h))
            # Arms (up-left / up-right from cy)
            band = int(round(0.02 * ny))
            armL = (yy <= cy + band) & (yy >= max(0, cy - arm_len)) & (np.abs((yy - cy) + slope*(xx - cx)) <= arm_thick)
            armR = (yy <= cy + band) & (yy >= max(0, cy - arm_len)) & (np.abs((yy - cy) - slope*(xx - cx)) <= arm_thick)

            open_mask |= stem | armL | armR

        elif shape == "random":
            # --- Larger organic blob: more steps + optional dilation(thickening) ---
            steps        = kwargs.get("steps", int(round(0.65 * nx * ny)))
            seeds        = kwargs.get("seeds", max(3, int(round(min(nx, ny) / 35))))
            jitter_prob  = kwargs.get("jitter_prob", 0.18)
            seed         = kwargs.get("seed", 7)
            thicken_iters = kwargs.get("thicken_iters", max(1, int(round(min(nx, ny)/50))))  # e.g., 2 when 100x100
            thicken_neighbors = kwargs.get("thicken_neighbors", "8")  # "4" or "8"

            rng = np.random.default_rng(seed)

            # Start seeds near center (spread slightly)
            centers = np.clip(np.array([
                [ny//2 + rng.integers(-ny//8, ny//8), nx//2 + rng.integers(-nx//8, nx//8)]
                for _ in range(seeds)
            ]), [0,0], [ny-1, nx-1])

            for (sy, sx) in centers:
                y, x = int(sy), int(sx)
                open_mask[y, x] = True
                for _ in range(max(1, steps // max(1, seeds))):
                    dy, dx = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
                    if rng.random() < jitter_prob:
                        dy = rng.choice([-1, 1]); dx = rng.choice([-1, 1])
                    y = int(np.clip(y + dy, 0, ny-1))
                    x = int(np.clip(x + dx, 0, nx-1))
                    open_mask[y, x] = True
                    # occasional local thickening
                    if rng.random() < 0.44:
                        y0, y1 = max(0, y-1), min(ny-1, y+1)
                        x0, x1 = max(0, x-1), min(nx-1, x+1)
                        open_mask[y0:y1+1, x0:x1+1] = True

            # light morphological thickening without external libs
            def _dilate(mask: np.ndarray, neighbors: str = "8") -> np.ndarray:
                g = mask.copy()
                # 4-neighbors
                g[1:, :]  |= mask[:-1, :]
                g[:-1, :] |= mask[1:, :]
                g[:, 1:]  |= mask[:, :-1]
                g[:, :-1] |= mask[:, 1:]
                if neighbors == "8":
                    g[1:, 1:]   |= mask[:-1, :-1]
                    g[1:, :-1]  |= mask[:-1, 1:]
                    g[:-1, 1:]  |= mask[1:, :-1]
                    g[:-1, :-1] |= mask[1:, 1:]
                return g

            for _ in range(thicken_iters):
                open_mask = _dilate(open_mask, neighbors=thicken_neighbors)

        else:
            raise ValueError(f"Unknown shape: {shape}. Use one of: rectangle, l, y, random")

        # Convert to obs_pos (row-major indices for obstacles = where open_mask is False)
        obs_pos: List[int] = [int(r * nx + c) for r in range(ny) for c in range(nx) if not open_mask[r, c]]
        return nx, ny, obs_pos

   
    def plot_rotated_bboxes(self,
                            nx: int,
                            ny: int,
                            obs_pos: List[int],
                            bboxes: List[Dict[str, float]],
                            title: str = "",
                            save_path: str = "outputs/plot_rotated_bboxes.png"):
        """
        회전된 bounding box들을 시각화하고 파일로 저장한다.
        """
        # 출력 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        grid = np.ones((ny, nx), dtype=float)
        for idx in obs_pos:
            r, c = divmod(idx, nx)
            grid[r, c] = 0.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap='RdYlBu', origin='lower', vmin=0, vmax=1, alpha=0.85)

        for i, rect in enumerate(bboxes, 1):
            x0, y0, w, h, th = rect["x0"], rect["y0"], rect["w"], rect["h"], rect["theta_deg"]
            patch = Rectangle((x0, y0), w, h, angle=th, fill=False, linewidth=2)
            ax.add_patch(patch)
            # 중심 근사치 위치에 라벨
            ax.text(x0 + w * 0.5, y0 + h * 0.5, str(i), ha='center', va='center', fontweight='bold')

        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)

        # show 대신 savefig 사용
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"[INFO] Figure saved to: {save_path}")