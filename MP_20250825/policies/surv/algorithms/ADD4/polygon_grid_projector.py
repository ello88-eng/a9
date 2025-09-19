# -*- coding: utf-8 -*-
import math
import random
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from config.mp_config import GRID_COVERAGE_THRESHOLD, USE_GRID_COVERAGE
from utils.coordinates import true_round


class PolygonGridProjector:
    """
    - 좌표 입력: [lat, lon]
    - 내부 연산: (x=lon, y=lat)로 변환해서 래스터라이즈
    - obs_pos: 폴리곤(or hull) 내부 픽셀의 flat indices (y*W + x)
    - in_pos projection: 장애물에서 min_clearance(맨해튼 거리) 이상 떨어진 자유공간으로 이동
    """

    def __init__(self, grid_size: int = 200):
        self.grid_size = true_round(grid_size)

    # ---------------- Convex hull (monotonic chain) on (x,y) = (lon,lat)
    @staticmethod
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _convex_hull_xy(self, points_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = sorted(set(map(tuple, points_xy)))
        if len(pts) <= 1:
            return pts
        lower = []
        for p in pts:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    import numpy as np

    def _downsample_grid_converage_data(
        self, grid: np.ndarray[int], factor: int = 4, threshold: float = GRID_COVERAGE_THRESHOLD
    ):
        h, w = grid.shape
        shrinked_grid: np.ndarray = grid.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))

        return (shrinked_grid >= threshold).astype(int)

    def get_obstacle_position_from_polygon_latlon(
        self,
        mission_area_latlon: List[List[float]],  # [[lat,lon], x4], 순서 무관
        mission_capable_area_latlon: List[List[float]],  # [[lat,lon], ...]
        grid_coverage: np.ndarray[np.uint8],
        use_hull: bool = True,
        stride: int = 1,
        max_count: Optional[int] = None,
    ) -> Tuple[List[int], List[Tuple[float, float]], Tuple[float, float, float, float]]:
        """
        return:
        - obs_pos: flat indices (list[int])  ← 이제 '폴리곤 외부'가 장애물
        - used_outline_xy: 사용된 외곽 (lon,lat) 좌표 리스트 (hull 또는 원본)
        - bbox: (min_lon, max_lon, min_lat, max_lat)
        """
        # ROI bbox
        lats = [p[0] for p in mission_area_latlon]
        lons = [p[1] for p in mission_area_latlon]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # polygon (x=lon, y=lat)
        poly_xy = [(p[1], p[0]) for p in mission_capable_area_latlon]
        if USE_GRID_COVERAGE:
            use_hull = True
        used_xy = self._convex_hull_xy(poly_xy) if use_hull else poly_xy
        if len(used_xy) < 3:
            return [], used_xy, (min_lon, max_lon, min_lat, max_lat)

        # to pixel (y-down)
        W = H = self.grid_size
        arr = np.asarray(used_xy, dtype=float)
        px = (arr[:, 0] - min_lon) / (max_lon - min_lon) * (W - 1)
        py = (1.0 - (arr[:, 1] - min_lat) / (max_lat - min_lat)) * (H - 1)
        poly_pixels = list(map(tuple, np.stack([px, py], axis=1)))

        # rasterize → mask (폴리곤 내부=1 = free space, 외부=0 = obstacle)
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).polygon(poly_pixels, fill=1, outline=1)
        mask = np.array(img, dtype=np.uint8)
        if USE_GRID_COVERAGE:
            shrinked_grid_coverage = self._downsample_grid_converage_data(grid_coverage)
            grid = 1 - shrinked_grid_coverage.astype(np.uint8)
            grid[(grid != 0) & (mask != 1)] = 0
            grid = np.array(grid, dtype=np.uint8)
            rotated_mask = np.rot90(grid, k=-1)
        else:
            rotated_mask = np.rot90(mask, k=-1)

        # ★ 변경점: 장애물은 '폴리곤 외부'이므로 mask == 0
        ys, xs = np.where(rotated_mask == 0)
        if stride > 1:
            ys = ys[::stride]
            xs = xs[::stride]
        flat = (ys * W + xs).astype(int)
        if max_count is not None and len(flat) > max_count:
            flat = flat[:max_count]

        return flat.tolist(), used_xy, (min_lon, max_lon, min_lat, max_lat)

    # ---------------- in_pos projection with min_clearance
    @staticmethod
    def _flat_to_yx(idx: int, W: int) -> Tuple[int, int]:
        y, x = divmod(true_round(idx), W)
        return y, x

    @staticmethod
    def _yx_to_flat(y: int, x: int, W: int) -> int:
        return true_round(y) * true_round(W) + true_round(x)

    @staticmethod
    def _neighbors4(y, x, H, W):
        if y > 0:
            yield y - 1, x
        if y < H - 1:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x < W - 1:
            yield y, x + 1

    @staticmethod
    def _bfs_dist_to_obstacles(obstacle_mask: np.ndarray) -> np.ndarray:
        """
        obstacle_mask: (H,W) 1=장애물, 0=빈공간
        return: 각 셀에서 가장 가까운 장애물까지의 맨해튼 거리 (셀 단위)
        """
        H, W = obstacle_mask.shape
        INF = 10**9
        dist = np.full((H, W), INF, dtype=np.int32)
        q = deque()
        ys, xs = np.where(obstacle_mask > 0)
        for y, x in zip(ys, xs):
            dist[y, x] = 0
            q.append((y, x))
        while q:
            y, x = q.popleft()
            d = dist[y, x] + 1
            for ny, nx in PolygonGridProjector._neighbors4(y, x, H, W):
                if d < dist[ny, nx]:
                    dist[ny, nx] = d
                    q.append((ny, nx))
        return dist

    @staticmethod
    def _nearest_allowed_cell(start_y: int, start_x: int, allowed_mask: np.ndarray) -> Tuple[int, int]:
        """
        start가 허용이 아니면, 4-이웃 BFS로 가장 가까운 allowed 셀을 찾음.
        """
        H, W = allowed_mask.shape
        if allowed_mask[start_y, start_x]:
            return start_y, start_x
        q = deque([(start_y, start_x)])
        seen = {(start_y, start_x)}
        while q:
            y, x = q.popleft()
            for ny, nx in PolygonGridProjector._neighbors4(y, x, H, W):
                if (ny, nx) in seen:
                    continue
                if allowed_mask[ny, nx]:
                    return ny, nx
                seen.add((ny, nx))
                q.append((ny, nx))

        return start_y, start_x  # 허용 셀이 전무한 극단적 케이스

    def project_in_pos_with_clearance(
        self, obs_pos: List[int], in_pos: List[int], min_clearance: int = 2
    ) -> List[int]:
        """obs_pos(장애물)로부터 min_clearance(맨해튼) 이상 떨어진 자유공간으로 각 in_pos를 투영하여 반환"""
        H = W = self.grid_size

        # 장애물 마스크
        obstacle = np.zeros((H, W), dtype=np.uint8)
        for idx in obs_pos:
            y, x = self._flat_to_yx(idx, W)
            obstacle[y, x] = 1

        # 거리장 + 허용 마스크
        dist = self._bfs_dist_to_obstacles(obstacle)
        allowed = dist >= 2

        # 투영
        adjusted_list = []
        adjusted_ny_nx_list = []
        cnt = 0
        for idx in in_pos:
            y, x = self._flat_to_yx(idx, W)
            ny, nx = self._nearest_allowed_cell(y, x, allowed)

            if cnt == 0:
                adjusted_ny_nx_list.append((ny, nx))
                cnt += 1
                continue

            for temp_ny, temp_nx in adjusted_ny_nx_list:
                if math.sqrt((temp_ny - ny) ** 2 + (temp_nx - nx) ** 2) >= min_clearance:
                    adjusted_ny_nx_list.append((ny, nx))
                    if len(adjusted_ny_nx_list) == len(in_pos):
                        break
                else:
                    while True:
                        random.seed(time.time())
                        if random.random() < 0.5:
                            new_nx = nx + random.randint(
                                int(math.sqrt(min_clearance)), int(math.sqrt(min_clearance)) + 3
                            )
                            new_ny = ny + random.randint(
                                int(math.sqrt(min_clearance)), int(math.sqrt(min_clearance)) + 3
                            )
                        else:
                            new_nx = nx - random.randint(
                                int(math.sqrt(min_clearance)), int(math.sqrt(min_clearance)) + 3
                            )
                            new_ny = ny - random.randint(
                                int(math.sqrt(min_clearance)), int(math.sqrt(min_clearance)) + 3
                            )

                        if allowed[new_ny, new_nx]:
                            adjusted_ny_nx_list.append((ny, nx))

                            if len(adjusted_ny_nx_list) == len(in_pos):
                                break

        for ny, nx in adjusted_ny_nx_list:
            adjusted_list.append(self._yx_to_flat(ny, nx, W))

        return adjusted_list
