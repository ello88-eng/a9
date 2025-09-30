
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Grid:
    W: int = 100
    H: int = 100
    dx: float = 1.0
    dy: float = 1.0

def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    return Path(poly, closed=True).contains_points(points)

def make_grid_points(grid: Grid) -> np.ndarray:
    xs = np.arange(grid.W) * grid.dx + 0.5 * grid.dx
    ys = np.arange(grid.H) * grid.dy + 0.5 * grid.dy
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

def uniform_points_in_polygon(poly: np.ndarray, grid: Grid) -> np.ndarray:
    pts = make_grid_points(grid)
    mask = points_in_polygon(pts, poly)
    return pts[mask]

def balanced_assignment(P: np.ndarray, centers: np.ndarray, quota: List[int]) -> np.ndarray:
    D = np.sum((P[:, None, :] - centers[None, :, :])**2, axis=2)  # (M,K)
    pref = np.argsort(D, axis=1)                                  # (M,K)
    order = np.argsort(D[np.arange(D.shape[0]), pref[:, 0]])      # process easy points first
    cap = np.array(quota, dtype=int).copy()
    idx = -np.ones(P.shape[0], dtype=int)
    for p in order:
        for c in pref[p]:
            if cap[c] > 0:
                idx[p] = c; cap[c] -= 1; break
    unassigned = np.where(idx < 0)[0]
    if len(unassigned) > 0:
        nn = np.argmin(D[unassigned], axis=1)
        idx[unassigned] = nn
    return idx

def project_to_polygon_if_needed(candidate: np.ndarray, poly: np.ndarray, pts_in_cell: np.ndarray) -> np.ndarray:
    inside = points_in_polygon(candidate[None, :], poly)[0]
    if inside: return candidate
    if len(pts_in_cell) == 0: return candidate
    d2 = np.sum((pts_in_cell - candidate[None, :])**2, axis=1)
    return pts_in_cell[np.argmin(d2)]

def equal_area_cvt_centers(poly: np.ndarray, K: int, grid: Grid, iters: int = 10, seed: int = 0) -> Tuple[np.ndarray, List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    P = uniform_points_in_polygon(poly, grid)
    if len(P) < K:
        raise ValueError("Not enough inside-grid points. Increase grid or reduce K.")
    centers = P[rng.choice(len(P), size=K, replace=False)]
    base = len(P) // K
    quota = [base] * K
    for i in range(len(P) - base * K):
        quota[i % K] += 1
    for _ in range(iters):
        idx = balanced_assignment(P, centers, quota)
        new_centers = centers.copy()
        for k in range(K):
            pts_k = P[idx == k]
            if len(pts_k) > 0:
                candi = pts_k.mean(axis=0)
                new_centers[k] = project_to_polygon_if_needed(candi, poly, pts_k)
            else:
                new_centers[k] = P[rng.integers(0, len(P))]
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers; break
        centers = new_centers
    idx = balanced_assignment(P, centers, quota)
    final_cells = [P[idx == k] for k in range(K)]
    return centers, final_cells

# --- tiny demo ---
def make_rectangle_polygon(x0,y0,x1,y1):
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], float)

def sample_concave_polygon():
    return np.array([[20,70],[80,70],[80,85],[55,85],[55,30],[45,30],[45,85],[20,85]], float)

def plot_poly_and_centers(poly, centers, title=""):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    closed = np.vstack([poly, poly[0]])
    ax.plot(closed[:,0], closed[:,1])
    ax.scatter(centers[:,0], centers[:,1], s=30)
    ax.set_aspect("equal","box")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    grid = Grid(100,100,1.0,1.0)
    rect = make_rectangle_polygon(20,20,80,80)
    centers_rect, _ = equal_area_cvt_centers(rect, K=3, grid=grid, iters=8, seed=2)
    plot_poly_and_centers(rect, centers_rect, "Equal-area CVT centers (rectangle)")
