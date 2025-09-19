import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from osc import approximate_open_space_rectangle # Open space coordinator
from mibb import multi_inscribed_bbox_packing

class Darp:
    def __init__(self, nx, ny, notEqualPortions, initial_positions, portions, obs_pos, visualization,
                 max_iter, cc_variation, random_level, d_cells, importance):
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.max_iter = max_iter

        self.grid = np.zeros((ny, nx))
        for obs in obs_pos:
            row, col = obs // nx, obs % nx
            if 0 <= row < ny and 0 <= col < nx:
                self.grid[row, col] = -1

        self.assg_mat = np.full((ny, nx), -1, dtype=int)
        free_cells = np.argwhere(self.grid != -1)
        
        initial_coords = np.array([[pos // nx, pos % nx] for pos in initial_positions])
        
        for y, x in free_cells:
            distances = np.linalg.norm(initial_coords - np.array([y, x]), axis=1)
            self.assg_mat[y, x] = np.argmin(distances)

    def divide_regions(self):
        print("Using Voronoi partitioning for region division.")
        return True, 0


class MultiRobotPathPlanner_ADD7:
    def __init__(
        self,
        nx,
        ny,
        notEqualPortions,
        initial_positions,
        portions,
        obs_pos,
        visualization,
        map_roi,
        turn_radius,
        fov_len,
        MaxIter=80000,
        CCvariation=0.01,
        randomLevel=0.0001,
        dcells=2,
        importance=False,
    ):
        self.nx = nx
        self.ny = ny
        self.num_uavs = len(initial_positions)
        self.initial_positions = initial_positions
        self.obs_pos = obs_pos
        self.turn_radius = turn_radius
        self.fov_len = fov_len
        

        approximate_open_space_rectangle(nx, ny, obs_pos, method="pca")
        optimal_bboxes = multi_inscribed_bbox_packing(
            nx=30, ny=20, 
            obs_pos=[...],
            min_width=8, min_height=6,
            robot_count=4,
            time_limit_seconds=5
        )


        # # DARP 인스턴스 생성
        # self.darp_instance = Darp(
        #     nx, ny, notEqualPortions, initial_positions, portions, obs_pos,
        #     visualization, MaxIter, CCvariation, randomLevel, dcells, importance
        # )
        
        # DARP 실행
        # self.DARP_success, self.iterations = self.darp_instance.divide_regions()
        
        if not self.DARP_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            print("DARP Success.")
            self._generate_paths(map_roi)

    def _generate_paths(self, map_roi):
        """경로 생성 메인 로직"""
        l_bin = self.calculate_grid_size(map_roi, self.nx, self.ny)
        
        # 기존 결과 구조 초기화
        self.hull_grids = []
        self.shull_grids = []
        self.circ_paths = []
        self.best_case = {"paths": []}
        
        # 각 UAV별 경로 생성
        for value in range(self.darp_instance.num_uavs):
            mask = self.darp_instance.assg_mat == value
            coords = np.column_stack(np.where(mask))
            
            if coords.shape[0] < 3:
                self.circ_paths.append([])
                self.hull_grids.append(np.full(self.darp_instance.assg_mat.shape, -1))
                self.shull_grids.append(np.full(self.darp_instance.assg_mat.shape, -1))
                continue

            # ConvexHull 계산
            hull = ConvexHull(coords)
            hull_coords = coords[hull.vertices]
            
            # Hull grid 생성
            hull_grid = np.full(self.darp_instance.assg_mat.shape, -1)
            rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], self.darp_instance.assg_mat.shape)
            hull_grid[rr, cc] = value
            self.hull_grids.append(hull_grid)

            shull_grid = np.full(self.darp_instance.assg_mat.shape, -1)
            shull_grid[rr, cc] = value
            self.shull_grids.append(shull_grid)

            # Hull edges 계산
            hull_edges = []
            for simplex in hull.simplices:
                hull_edges.append((coords[simplex[0]], coords[simplex[1]]))
            
            # 경로 생성
            path = self._generate_intersection_path(coords, hull_coords, hull_edges, l_bin)
            self.circ_paths.append(path)
        
        # 결과 포맷팅
        self.best_case["paths"] = [[(int(p[0]), int(p[1])) for p in path] for path in self.circ_paths]

    def _generate_intersection_path(self, coords, hull_coords, hull_edges, l_bin):
        """Intersection 기반 경로 생성"""
        mean = np.mean(coords, axis=0)
        
        # 가로 방향 주축 설정 (긴 직선 경로)
        major_axis = np.array([1.0, 0.0])
        minor_axis = np.array([0.0, 1.0])
        
        # 스캔 라인 계산
        hull_points = coords[ConvexHull(coords).vertices] if len(coords) >= 3 else coords
        projected_points = hull_points @ minor_axis
        max_grid = np.max(projected_points) - np.min(projected_points) if len(projected_points) > 1 else 1
        
        overlap_margin = 0.2
        s_beta = (self.fov_len / l_bin) * (1 - overlap_margin) if l_bin > 0 else 1
        
        perpendicular_axis = np.array([-major_axis[1], major_axis[0]])
        line_count = int(max_grid / s_beta + 1) if s_beta > 0 else 1
        line_spacing = s_beta
        
        # 스캔 라인들 생성
        lines = []
        for i in range(-line_count // 2, line_count // 2 + 1):
            offset = perpendicular_axis * (i * line_spacing)
            point_on_line = mean + offset
            lines.append((point_on_line, major_axis))
        
        # Intersection 계산
        intersections = []
        for line_point, direction in lines:
            direction = direction / np.linalg.norm(direction)
            line_intersections = []
            
            for edge_start, edge_end in hull_edges:
                intersection = self.line_edge_intersection(line_point, direction, edge_start, edge_end)
                if intersection is not None:
                    line_intersections.append(intersection)
            
            if len(line_intersections) >= 2:
                line_intersections.sort(key=lambda p: np.dot(p - line_point, direction))
                intersections.append((line_intersections[0], line_intersections[-1]))
        
        # 지그재그 패턴 경로 생성
        path = []
        if len(intersections) > 0:
            mid_point = len(intersections) // 2
            max_lines = max(mid_point, len(intersections) - mid_point)
            
            for i in range(max_lines):
                # 1파트 라인
                if i < mid_point:
                    pair1 = intersections[i]
                    path.extend([pair1[0], pair1[1]])
                
                # 2파트 라인
                part2_idx = mid_point + i
                if part2_idx < len(intersections):
                    pair2 = intersections[part2_idx]
                    
                    if i >= mid_point:  # 1파트가 끝난 후 남은 라인들
                        path.extend([pair2[0], pair2[1]])  # 정방향
                    else:
                        path.extend([pair2[1], pair2[0]])  # 역방향
        
        return path

    def line_edge_intersection(self, line_point, line_direction, edge_start, edge_end):
        """직선과 선분의 교점 계산"""
        edge_vector = edge_end - edge_start
        matrix = np.array([line_direction, -edge_vector]).T
        
        if np.linalg.det(matrix) == 0:
            return None
            
        try:
            t, u = np.linalg.solve(matrix, edge_start - line_point)
            if 0 <= u <= 1:
                return line_point + t * line_direction
        except np.linalg.LinAlgError:
            return None
        return None

    def calculate_grid_size(self, map_roi, nx, ny):
        """Grid size 계산"""
        return 50

    def plot_visualization(self, grid_size, obs_pos, safe_in_pos, avs_pos_list):
        """경로 계획 결과 시각화"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.darp_instance.assg_mat, cmap='viridis', origin='lower', alpha=0.7)

        # 장애물 표시
        obs_coords = np.array([[o % grid_size, o // grid_size] for o in obs_pos])
        if len(obs_coords) > 0:
            plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c='red', s=10, label='Obstacles')

        # 초기 위치 표시
        initial_coords = np.array([[p % grid_size, p // grid_size] for p in safe_in_pos])
        plt.scatter(initial_coords[:, 0], initial_coords[:, 1], c='black', s=100, marker='X', label='Initial Positions')

        # 경로 표시
        colors = plt.cm.jet(np.linspace(0, 1, len(avs_pos_list)))
        for i, path in enumerate(self.best_case["paths"]):
            if not path: 
                continue
            path_arr = np.array(path)
            plt.plot(path_arr[:, 1], path_arr[:, 0], color=colors[i], linewidth=2, label=f'UAV {i+1} Path')
            plt.scatter(path_arr[:, 1], path_arr[:, 0], color=colors[i], s=25, zorder=5)
        
        plt.legend()
        plt.title("Multi-Robot Path Planning Visualization")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)
        plt.show()