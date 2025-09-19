import math
import random
import time
from copy import deepcopy
from enum import IntEnum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union, voronoi_diagram

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from utils.coordinates import (
    convert_avs_pos_lla_to_enu,
    convert_boundary_lla_to_enu,
    convert_waypoints_enu_to_lla,
    split_origin,
)
from utils.gridmap import FloatGrid, GridMap, rotate_matrix_2d
from utils.logger import logger
from utils.recognition import get_avail_avs


class RandomJumpPolygon:

    def __init__(
        self,
        polygon: Polygon,
        obstacles: List[Polygon],
        num_regions: int,
        iterations: int = 10000,
        delta: float = 0.1,
    ):
        self.polygon = polygon
        self.obstacles = unary_union(obstacles)  # Combine all obstacles into a single geometry
        self.num_regions = num_regions
        self.iterations = iterations
        self.delta = delta
        self.valid_area = self.polygon.difference(self.obstacles)  # Valid area is the polygon without obstacles
        self.points = self.initialize_points()

    def initialize_points(self) -> List[Point]:
        min_x, min_y, max_x, max_y = self.valid_area.bounds
        points = []
        while len(points) < self.num_regions:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            point = Point(x, y)
            if self.valid_area.contains(point):
                points.append(point)
        return points

    def random_jump(self) -> None:
        min_x, min_y, max_x, max_y = self.valid_area.bounds
        for iteration in range(self.iterations):
            new_points = []
            for point in self.points:
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                new_point = Point(x, y)
                if self.valid_area.contains(new_point):
                    new_points.append(new_point)
                else:
                    new_points.append(point)
            self.points = new_points
            if self.check_area_uniformity(iteration):
                break

    def check_area_uniformity(self, iteration: int) -> bool:
        points_coords = [(point.x, point.y) for point in self.points]
        multi_point = MultiPoint(points_coords)
        vor = voronoi_diagram(multi_point, envelope=self.polygon)

        areas = []
        if vor.geom_type == "GeometryCollection":
            for geom in vor.geoms:
                if self.valid_area.intersects(geom):
                    intersected_region = self.valid_area.intersection(geom)
                    if intersected_region.geom_type == "Polygon":
                        areas.append(intersected_region.area)
        elif vor.geom_type == "Polygon":
            if self.valid_area.intersects(vor):
                intersected_region = self.valid_area.intersection(vor)
                areas.append(intersected_region.area)

        if areas:
            mean_area = np.mean(areas)
            std_dev = np.std(areas)
            return std_dev < self.delta * mean_area
        return False

    def plot_regions(self) -> None:
        points_coords = [(point.x, point.y) for point in self.points]
        multi_point = MultiPoint(points_coords)
        vor = voronoi_diagram(multi_point, envelope=self.polygon)

        plt.figure()
        if vor.geom_type == "GeometryCollection":
            for geom in vor.geoms:
                if self.valid_area.intersects(geom):
                    intersected_region: Polygon = self.valid_area.intersection(geom)
                    if intersected_region.geom_type == "Polygon":
                        x, y = intersected_region.exterior.xy
                        plt.fill(x, y, alpha=0.4)
        elif vor.geom_type == "Polygon":
            if self.valid_area.intersects(vor):
                intersected_region = self.valid_area.intersection(vor)
                x, y = intersected_region.exterior.xy
                plt.fill(x, y, alpha=0.4)

        x, y = self.polygon.exterior.xy
        plt.plot(x, y, color="black")

        if self.obstacles.geom_type == "Polygon":
            ox, oy = self.obstacles.exterior.xy
            plt.plot(ox, oy, color="red")
        elif self.obstacles.geom_type == "MultiPolygon":
            for obstacle in self.obstacles.geoms:
                ox, oy = obstacle.exterior.xy
                plt.plot(ox, oy, color="red")

        plt.scatter(
            [point.x for point in self.points],
            [point.y for point in self.points],
            color="blue",
        )
        plt.show()

    def get_regions_coordinates(self) -> List[List[Tuple[float, float]]]:
        points_coords = [(point.x, point.y) for point in self.points]
        multi_point = MultiPoint(points_coords)
        vor = voronoi_diagram(multi_point, envelope=self.polygon)

        regions = []
        if vor.geom_type == "GeometryCollection":
            for geom in vor.geoms:
                if self.valid_area.intersects(geom):
                    intersected_region: Polygon = self.valid_area.intersection(geom)
                    if intersected_region.geom_type == "Polygon":
                        regions.append(list(intersected_region.exterior.coords))
        elif vor.geom_type == "Polygon":
            if self.valid_area.intersects(vor):
                intersected_region = self.valid_area.intersection(vor)
                regions.append(list(intersected_region.exterior.coords))

        return regions

    def run(self, show: bool = False):
        self.random_jump()
        if show:
            self.plot_regions()
        return self.get_regions_coordinates()


class SweepSearcher:
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1

    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1

    def __init__(self, moving_direction, sweep_direction, x_inds_goal_y, goal_y):
        self.moving_direction = moving_direction
        self.sweep_direction = sweep_direction
        self.turing_window = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_goal_y
        self.goal_y = goal_y

    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index

        # found safe grid
        if not self.check_occupied(n_x_index, n_y_index, grid_map):
            return n_x_index, n_y_index
        else:  # occupied
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(c_x_index, c_y_index, grid_map)
            if (next_c_x_index is None) and (next_c_y_index is None):
                # moving backward
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if self.check_occupied(next_c_x_index, next_c_y_index, grid_map, FloatGrid(1.0)):
                    # moved backward, but the grid is occupied by obstacle
                    return None, None
            else:
                # keep moving until end
                while not self.check_occupied(
                    next_c_x_index + self.moving_direction,
                    next_c_y_index,
                    grid_map,
                ):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()
            return next_c_x_index, next_c_y_index

    @staticmethod
    def check_occupied(c_x_index, c_y_index, grid_map, occupied_val=FloatGrid(0.5)):
        return grid_map.check_occupied_from_xy_index(c_x_index, c_y_index, occupied_val)

    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):

        for d_x_ind, d_y_ind in self.turing_window:

            next_x_ind = d_x_ind + c_x_index
            next_y_ind = d_y_ind + c_y_index

            # found safe grid
            if not self.check_occupied(next_x_ind, next_y_ind, grid_map):
                return next_x_ind, next_y_ind

        return None, None

    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not self.check_occupied(ix, self.goal_y, grid_map):
                return False

        # all lower grid is occupied
        return True

    def update_turning_window(self):
        # turning window definition
        # robot can move grid based on it.
        self.turing_window = [
            (self.moving_direction, 0.0),
            (self.moving_direction, self.sweep_direction),
            (0, self.sweep_direction),
            (-self.moving_direction, self.sweep_direction),
        ]

    def swap_moving_direction(self):
        self.moving_direction *= -1
        self.update_turning_window()

    def search_start_grid(self, grid_map):
        x_inds = []
        y_ind = 0
        if self.sweep_direction == self.SweepDirection.DOWN:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
        elif self.sweep_direction == self.SweepDirection.UP:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=False)

        if self.moving_direction == self.MovingDirection.RIGHT:
            return min(x_inds), y_ind
        elif self.moving_direction == self.MovingDirection.LEFT:
            return max(x_inds), y_ind

        raise ValueError("self.moving direction is invalid ")


def find_sweep_direction_and_start_position(ox, oy):
    # find sweep_direction
    max_dist = 0.0
    vec = [0.0, 0.0]
    sweep_start_pos = [0.0, 0.0]
    for i in range(len(ox) - 1):
        dx = ox[i + 1] - ox[i]
        dy = oy[i + 1] - oy[i]
        d = np.hypot(dx, dy)

        if d > max_dist:
            max_dist = d
            vec = [dx, dy]
            sweep_start_pos = [ox[i], oy[i]]

    return vec, sweep_start_pos


def convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position):
    tx = [ix - sweep_start_position[0] for ix in ox]
    ty = [iy - sweep_start_position[1] for iy in oy]
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    converted_xy = np.stack([tx, ty]).T @ rotate_matrix_2d(th)

    return converted_xy[:, 0], converted_xy[:, 1]


def convert_global_coordinate(x, y, sweep_vec, sweep_start_position):
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    converted_xy = np.stack([x, y]).T @ rotate_matrix_2d(-th)
    rx = [ix + sweep_start_position[0] for ix in converted_xy[:, 0]]
    ry = [iy + sweep_start_position[1] for iy in converted_xy[:, 1]]
    return rx, ry


def search_free_grid_index_at_edge_y(grid_map, from_upper=False):
    y_index = None
    x_indexes = []

    if from_upper:
        x_range = range(grid_map.height)[::-1]
        y_range = range(grid_map.width)[::-1]
    else:
        x_range = range(grid_map.height)
        y_range = range(grid_map.width)

    for iy in x_range:
        for ix in y_range:
            if not SweepSearcher.check_occupied(ix, iy, grid_map):
                y_index = iy
                x_indexes.append(ix)
        if y_index:
            break

    return x_indexes, y_index


def setup_grid_map(ox, oy, resolution, sweep_direction, offset_grid=10):
    width = 200
    height = 200
    center_x = (np.max(ox) + np.min(ox)) / 2.0
    center_y = (np.max(oy) + np.min(oy)) / 2.0
    grid_map = GridMap(width, height, resolution, center_x, center_y)
    grid_map.set_value_from_polygon(ox, oy, FloatGrid(1.0), inside=False)

    grid_map.expand_grid()

    x_inds_goal_y = []
    goal_y = 0
    if sweep_direction == SweepSearcher.SweepDirection.UP:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
    elif sweep_direction == SweepSearcher.SweepDirection.DOWN:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=False)

    return grid_map, x_inds_goal_y, goal_y


def sweep_path_search(
    sweep_searcher: SweepSearcher,
    grid_map: GridMap,
    grid_search_animation=False,
):
    # search start grid
    c_x_index, c_y_index = sweep_searcher.search_start_grid(grid_map)
    if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5)):
        logger.info("Cannot find start grid")
        return [], []

    x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)
    px, py = [x], [y]

    fig, ax = None, None
    if grid_search_animation:
        fig, ax = plt.subplots()
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

    while True:
        c_x_index, c_y_index = sweep_searcher.move_target_grid(c_x_index, c_y_index, grid_map)

        if sweep_searcher.is_search_done(grid_map) or (c_x_index is None or c_y_index is None):
            break

        x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)

        px.append(x)
        py.append(y)

        grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5))

        if grid_search_animation:
            grid_map.plot_grid_map(ax=ax)
            plt.pause(1.0)

    return px, py


def douglas_peucker(xs, ys, epsilon):
    # Find the point with the maximum distance
    def find_max_distance_index(xs, ys, start, end):
        max_dist = 0.0
        index = start
        for i in range(start + 1, end):
            dist = perpendicular_distance(xs, ys, start, end, i)
            if dist > max_dist:
                max_dist = dist
                index = i
        return index, max_dist

    # Calculate the perpendicular distance from a point to a line
    def perpendicular_distance(xs, ys, start, end, i):
        x1, y1 = xs[start], ys[start]
        x2, y2 = xs[end], ys[end]
        x0, y0 = xs[i], ys[i]
        return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Recursive simplification
    def simplify(xs, ys, start, end, epsilon):
        index, max_dist = find_max_distance_index(xs, ys, start, end)
        if max_dist > epsilon:
            results1 = simplify(xs, ys, start, index, epsilon)
            results2 = simplify(xs, ys, index, end, epsilon)
            return results1[:-1] + results2
        else:
            return [(xs[start], ys[start], 800.0), (xs[end], ys[end], 800.0)]

    # Start the simplification process
    simplified_points = simplify(xs, ys, 0, len(xs) - 1, epsilon)
    return [list(t) for t in zip(*simplified_points)]


def get_corner_waypoints(rx, ry, z_value):
    corner_points = [(rx[0], ry[0], z_value)]
    for i in range(1, len(rx) - 1):
        dx1, dy1 = rx[i] - rx[i - 1], ry[i] - ry[i - 1]
        dx2, dy2 = rx[i + 1] - rx[i], ry[i + 1] - ry[i]
        angle_1 = np.degrees(np.arctan2(dy1, dx1))
        angle_2 = np.degrees(np.arctan2(dy2, dx2))
        if np.abs(angle_2 - angle_1) >= 10:
            corner_points.append((rx[i], ry[i], z_value))
    corner_points.append((rx[-1], ry[-1], z_value))
    return corner_points


def get_surv_path(
    ox,
    oy,
    resolution,
    moving_direction=SweepSearcher.MovingDirection.RIGHT,
    sweeping_direction=SweepSearcher.SweepDirection.UP,
):
    sweep_vec, sweep_start_position = find_sweep_direction_and_start_position(ox, oy)
    rox, roy = convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position)
    grid_map, x_inds_goal_y, goal_y = setup_grid_map(rox, roy, resolution, sweeping_direction)
    sweep_searcher = SweepSearcher(moving_direction, sweeping_direction, x_inds_goal_y, goal_y)
    px, py = sweep_path_search(sweep_searcher, grid_map)
    rx, ry = convert_global_coordinate(px, py, sweep_vec, sweep_start_position)
    # dpx, dpy = douglas_peucker(xs=rx, ys=ry, epsilon=5.0)
    path = get_corner_waypoints(rx, ry, 800.0)

    return path


def show_surveillance_path(xs: List[float], ys: List[float], resolution: float) -> None:
    path = get_surv_path(xs, ys, resolution)
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    plt.plot(xs, ys, "-x")
    plt.plot(px, py)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def select(
    manager: Manager,
) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
    # 기준 좌표 얻기
    lat0, lon0, alt0 = split_origin(manager.mp_input.boundary.vertices)
    # 영역 좌표 변환
    top_l, top_r, bot_r, bot_l = convert_boundary_lla_to_enu(
        lat0=manager.lat0,
        lon0=manager.lon0,
        alt0=manager.alt0,
        boundary_as_lla=manager.mp_input.boundary,
    )
    # 비행체 좌표 변환
    _ = convert_avs_pos_lla_to_enu(infos=deepcopy(manager.mp_input.avs_info_dict), lat0=lat0, lon0=lon0, alt0=alt0)
    # 감시임무 가능한 비행체 반환
    avs_list, _ = get_avail_avs(
        avs_to_avail_task=manager.avs_to_available_task_dict,
        criteria=["S", "s"],
        avs_info_dict=manager.mp_input.avs_info_dict,
    )
    # 임무계획 결과 초기화
    mp_output_dict: Dict[int, SurveillanceSelectionResult] = {}
    start_time = time.time()
    # 영역을 비행체 개수만큼 분할
    rjp = RandomJumpPolygon(
        polygon=Polygon([top_l, top_r, bot_r, bot_l]),
        obstacles=None,
        num_regions=len(avs_list),
    )
    regions = rjp.run()
    region_xs = [list(zip(*region))[0] for region in regions]
    region_ys = [list(zip(*region))[1] for region in regions]
    # 영역 분할이 잘 맞으면
    if len(avs_list) == len(region_xs):
        # 분할된 영역 별 감시경로 생성
        for i, avs in enumerate(avs_list):
            wps = get_surv_path(region_xs[i], region_ys[i], resolution=50)
            #
            mp_output_dict[avs] = SurveillanceSelectionResult(
                avs_id=avs,
                approval_flag=0,
                sys_group_id=0,
                num_avs_per_group=0,
                smp_mode=1,
                speed=27,
                waypoint_count=len(wps),
                waypoints=convert_waypoints_enu_to_lla(wps, lat0, lon0, alt0),
                turn_radius=0,
                turn_dir=0,
                trg_pt=np.zeros((1, 3)),
                trg_id=0,
                attack_prio=0,
                is_bd=0,
            )
    # 영역 분할이 안 맞으면
    else:
        # 감시임무 가능 비행체 재확인
        if len(avs_list) > 0:
            # 영역을 비행체 개수만큼 분할
            rjp = RandomJumpPolygon(
                polygon=Polygon([top_l, top_r, bot_r, bot_l]),
                obstacles=None,
                num_regions=len(avs_list),
            )
            regions = rjp.run()
            region_xs = [list(zip(*region))[0] for region in regions]
            region_ys = [list(zip(*region))[1] for region in regions]
            # 각 분할된 영역 별 감시경로 생성
            for i, avs in enumerate(avs_list):
                wps = get_surv_path(region_xs[i], region_ys[i], resolution=50)
                # 각 비행체 별 감시경로 할당
                mp_output_dict[avs] = SurveillanceSelectionResult(
                    avs_id=avs,
                    approval_flag=0,
                    sys_group_id=0,
                    num_avs_per_group=0,
                    smp_mode=1,
                    speed=27,
                    waypoint_count=len(wps),
                    waypoints=convert_waypoints_enu_to_lla(wps, lat0, lon0, alt0),
                    turn_radius=0,
                    turn_dir=0,
                    trg_pt=np.zeros((1, 3)),
                    trg_id=0,
                    attack_prio=0,
                    is_bd=0,
                )
    return time.time() - start_time, mp_output_dict, deepcopy(manager.mp_input)


def main():
    # ox = [-2345.24, 2273.810, 2273.810, -2345.24, -2345.24]
    ox = [
        -914.8667315147579,
        -670.0871633695317,
        -1000.0,
        -1000.0,
        -800.0,
        -800.0,
        -371.78384324382074,
        -327.4303270396802,
        -555.004011459156,
        -2338.7709761411866,
        -967.6467801697285,
        -914.8667315147579,
    ]
    # oy = [392.857, 392.857, -4678.573, -4678.573, 392.857]
    oy = [
        392.857,
        0.0,
        0.0,
        -1000.0,
        -1000.0,
        -272.98002544158527,
        -478.75951543920525,
        -1909.7821649244138,
        -2195.3561972868633,
        -2743.3662814359304,
        392.857,
        392.857,
    ]
    resolution = 50
    show_surveillance_path(ox, oy, resolution)


if __name__ == "__main__":
    main()
