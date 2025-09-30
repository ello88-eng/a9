
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

@dataclass
class Box:
    cx: float; cy: float; theta: float; w: float; h: float

@dataclass
class MinOnlyBounds:
    wmin: float = 10.0; hmin: float = 10.0

@dataclass
class Grid:
    W: int = 100; H: int = 100; dx: float = 1.0; dy: float = 1.0

def make_rectangle_polygon(x0,y0,x1,y1):
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], float)

def corners_of_box(b: Box):
    hw, hh = b.w*0.5, b.h*0.5
    local = np.array([[-hw,-hh],[ hw,-hh],[ hw, hh],[-hw, hh]])
    ct, st = np.cos(b.theta), np.sin(b.theta)
    R = np.array([[ct,-st],[st,ct]])
    return local @ R.T + np.array([b.cx,b.cy])

def draw(poly, boxes, title=""):
    fig = plt.figure(figsize=(6,6)); ax = plt.gca()
    closed = np.vstack([poly, poly[0]]); ax.plot(closed[:,0], closed[:,1])
    for b in boxes:
        cs = np.vstack([corners_of_box(b), corners_of_box(b)[0]])
        ax.plot(cs[:,0], cs[:,1])
    ax.set_aspect("equal","box"); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_title(title); plt.show()

def points_in_polygon(points, poly):
    return Path(poly, closed=True).contains_points(points)

def grid_points(grid: Grid):
    xs = np.arange(grid.W)*grid.dx + 0.5*grid.dx
    ys = np.arange(grid.H)*grid.dy + 0.5*grid.dy
    X,Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

def cvt_voronoi_seeds(poly, K, grid: Grid, iters=3, seed=0):
    pts = grid_points(grid)
    inside_mask = points_in_polygon(pts, poly)
    P = pts[inside_mask]
    rng = np.random.default_rng(seed)
    if len(P) < K: raise ValueError("Not enough inside points")
    seeds = P[rng.choice(len(P), size=K, replace=False)]
    for _ in range(iters):
        d = np.sum((P[:,None,:]-seeds[None,:,:])**2, axis=2)
        idx = np.argmin(d, axis=1)
        for k in range(K):
            sel = (idx==k)
            if np.any(sel): seeds[k] = P[sel].mean(axis=0)
            else: seeds[k] = P[rng.integers(0, len(P))]
    return P, inside_mask, seeds

def voronoi_cells(P, seeds):
    d = np.sum((P[:,None,:]-seeds[None,:,:])**2, axis=2)
    idx = np.argmin(d, axis=1)
    return [P[idx==k] for k in range(len(seeds))]

def pca_axes(points):
    c = points.mean(axis=0)
    X = points - c
    if len(points) < 2: return c, np.array([1.0,0.0]), np.array([0.0,1.0])
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]; v = Vt[1] if Vt.shape[0] > 1 else np.array([-u[1],u[0]])
    return c, u/np.linalg.norm(u), v/np.linalg.norm(v)

def obb_extents(points, c, u, v):
    pu = (points - c) @ u; pv = (points - c) @ v
    return pu.min(), pu.max(), pv.min(), pv.max()

def corners_in_cell(corners, seed, other_seeds):
    if other_seeds.size==0: return True
    own = np.sum((corners - seed[None,:])**2, axis=1)
    oth = np.min(np.sum((corners[:,None,:]-other_seeds[None,:,:])**2, axis=2), axis=1)
    return np.all(own <= oth + 1e-6)

def fit_obb_in_cell(cell_pts, seed, other_seeds, bounds: MinOnlyBounds, shrink=0.9, max_iter=200):
    c,u,v = pca_axes(cell_pts); v = np.array([-u[1],u[0]])
    umin,umax,vmin,vmax = obb_extents(cell_pts, c, u, v)
    w = max(bounds.wmin, (umax-umin)*shrink); h = max(bounds.hmin, (vmax-vmin)*shrink)
    theta = float(np.arctan2(u[1], u[0])); b = Box(c[0], c[1], theta, w, h)
    for _ in range(max_iter):
        cs = corners_of_box(b)
        if corners_in_cell(cs, seed, other_seeds): return b
        b.w = max(bounds.wmin, b.w*0.95); b.h = max(bounds.hmin, b.h*0.95)
    return b

def strip_boxes_in_cell(cell_pts, seed, other_seeds, bounds: MinOnlyBounds, N_strips, shrink=0.9, max_iter=200):
    if len(cell_pts)==0: return []
    c,u,v = pca_axes(cell_pts); v = np.array([-u[1],u[0]])
    umin,umax,vmin,vmax = obb_extents(cell_pts, c, u, v)
    span_u = (umax-umin)*shrink; span_v = (vmax-vmin)*shrink
    max_possible = int(max(1, np.floor(span_u / bounds.wmin)))
    N = max(1, min(N_strips, max_possible))
    w = span_u / N; h = max(bounds.hmin, span_v)
    theta = float(np.arctan2(u[1], u[0]))
    boxes = []
    for i in range(N):
        u_center = umin + (i + 0.5) * (span_u / N)
        ctr = c + u * u_center
        b = Box(ctr[0], ctr[1], theta, w, h)
        it=0
        while it<max_iter:
            cs = corners_of_box(b)
            if corners_in_cell(cs, seed, other_seeds): break
            b.w = max(bounds.wmin, b.w*0.95); b.h = max(bounds.hmin, b.h*0.95)
            it += 1
        boxes.append(b)
    return boxes

def allocate_strips(K_total, cell_pts_list, bounds: MinOnlyBounds, shrink=0.9):
    nonempty = [i for i,pts in enumerate(cell_pts_list) if len(pts)>0]
    if len(nonempty)==0: return [0]*len(cell_pts_list)
    if K_total < len(nonempty):
        sizes = np.array([len(cell_pts_list[i]) for i in nonempty], float)
        order = np.argsort(-sizes)
        chosen = set([nonempty[i] for i in order[:K_total]])
        return [1 if i in chosen else 0 for i in range(len(cell_pts_list))]
    areas = np.array([len(pts) for pts in cell_pts_list], float)
    s = areas.sum(); base = np.zeros(len(cell_pts_list), int)
    for i,pts in enumerate(cell_pts_list):
        base[i] = 0 if len(pts)==0 else max(1, int(np.floor(K_total*len(pts)/max(1,s))))
    diff = K_total - int(base.sum())
    if diff>0:
        order = np.argsort(-areas)
        for i in order:
            if diff==0: break
            if len(cell_pts_list[i])==0: continue
            base[i]+=1; diff-=1
    elif diff<0:
        order = np.argsort(areas)
        for i in order:
            if diff==0: break
            if base[i]>1: base[i]-=1; diff+=1
    capped = base.copy()
    for i,pts in enumerate(cell_pts_list):
        if len(pts)==0: capped[i]=0; continue
        c,u,v = pca_axes(pts); v = np.array([-u[1],u[0]])
        umin,umax,_,_ = obb_extents(pts, c, u, v)
        max_possible = int(max(1, np.floor((umax-umin)*shrink / bounds.wmin)))
        capped[i] = max(1, min(capped[i], max_possible))
    diff = K_total - int(capped.sum())
    if diff>0:
        order = np.argsort(-areas)
        for i in order:
            if diff==0: break
            c,u,v = pca_axes(cell_pts_list[i]); v = np.array([-u[1],u[0]])
            umin,umax,_,_ = obb_extents(cell_pts_list[i], c, u, v)
            max_possible = int(max(1, np.floor((umax-umin)*shrink / bounds.wmin)))
            if capped[i] < max_possible:
                capped[i]+=1; diff-=1
    elif diff<0:
        order = np.argsort(-areas)
        for i in order:
            if diff==0: break
            if capped[i]>1: capped[i]-=1; diff+=1
    return list(capped)

def voronoi_obb_per_cell(poly, K_cells, bounds: MinOnlyBounds, grid: Grid, cvt_iters=3, seed=0):
    P,_,seeds = cvt_voronoi_seeds(poly, K_cells, grid, iters=cvt_iters, seed=seed)
    cells = voronoi_cells(P, seeds)
    boxes = []
    for k,pts in enumerate(cells):
        other = np.delete(seeds, k, axis=0)
        if len(pts)==0: boxes.append(Box(seeds[k,0], seeds[k,1], 0.0, bounds.wmin, bounds.hmin))
        else: boxes.append(fit_obb_in_cell(pts, seeds[k], other, bounds, shrink=0.9))
    return boxes

def voronoi_strip_tiling(poly, K_total, K_cells, bounds: MinOnlyBounds, grid: Grid, cvt_iters=3, seed=0):
    P,_,seeds = cvt_voronoi_seeds(poly, K_cells, grid, iters=cvt_iters, seed=seed)
    cells = voronoi_cells(P, seeds)
    alloc = allocate_strips(K_total, cells, bounds, shrink=0.9)
    boxes_all = []
    for k,pts in enumerate(cells):
        N = alloc[k]
        if N==0: continue
        other = np.delete(seeds, k, axis=0)
        boxes_all.extend(strip_boxes_in_cell(pts, seeds[k], other, bounds, N, shrink=0.9))
    return boxes_all
