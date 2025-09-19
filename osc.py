import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Literal
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

### =======================================================
def fit_rotated_rectangle_for_diagonal(open_coords):
    """
    다이아몬드/대각선 형태를 위한 회전된 사각형 생성
    """
    if len(open_coords) < 3:
        return fit_circumscribed_rectangle(open_coords)
    
    coords_array = np.array(open_coords)
    
    # 1. 먼저 대각선 패턴인지 확인
    def is_diagonal_pattern(coords):
        # ConvexHull로 외곽선 찾기
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        
        # 주요 방향들 계산
        directions = []
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            direction = p2 - p1
            if np.linalg.norm(direction) > 0:
                directions.append(direction / np.linalg.norm(direction))
        
        # 45도 근처 방향이 있는지 확인
        diagonal_angles = [np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
        for direction in directions:
            angle = np.arctan2(direction[1], direction[0])
            for diag_angle in diagonal_angles:
                if abs(angle - diag_angle) < np.pi/6:  # 30도 허용범위
                    return True, angle
        return False, 0
    
    is_diagonal, main_angle = is_diagonal_pattern(coords_array)
    
    if is_diagonal:
        # 대각선 패턴이면 회전된 사각형 생성
        return create_rotated_bbox(coords_array, main_angle)
    else:
        # 일반 패턴이면 PCA 사용
        return create_pca_aligned_bbox(coords_array)

def create_rotated_bbox(coords_array, rotation_angle):
    """주어진 각도로 회전된 bounding box 생성"""
    
    # 회전 행렬
    cos_a = np.cos(-rotation_angle)
    sin_a = np.sin(-rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # 좌표 회전
    rotated_coords = coords_array @ rotation_matrix.T
    
    # 회전된 좌표계에서 bounding box
    min_rot = rotated_coords.min(axis=0)
    max_rot = rotated_coords.max(axis=0)
    
    # 여유 공간 추가 (다이아몬드가 잘리지 않도록)
    margin = 0.1
    width = max_rot[0] - min_rot[0]
    height = max_rot[1] - min_rot[1]
    
    min_rot[0] -= width * margin
    max_rot[0] += width * margin
    min_rot[1] -= height * margin
    max_rot[1] += height * margin
    
    # 회전된 사각형의 네 모서리
    corners_rot = np.array([
        [min_rot[0], min_rot[1]],  # 좌하
        [max_rot[0], min_rot[1]],  # 우하  
        [max_rot[0], max_rot[1]],  # 우상
        [min_rot[0], max_rot[1]]   # 좌상
    ])
    
    # 원래 좌표계로 역회전
    inv_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    corners_orig = corners_rot @ inv_rotation.T
    
    # 축에 정렬된 bounding box (회전된 사각형을 포함하는)
    min_col = int(np.floor(corners_orig[:, 0].min()))
    max_col = int(np.ceil(corners_orig[:, 0].max()))
    min_row = int(np.floor(corners_orig[:, 1].min()))
    max_row = int(np.ceil(corners_orig[:, 1].max()))
    
    return (min_col, min_row, max_col, max_row)

def create_pca_aligned_bbox(coords_array):
    """PCA 주성분 방향으로 정렬된 bbox"""
    
    # PCA로 주성분 찾기
    pca = PCA(n_components=2)
    pca.fit(coords_array)
    
    # 첫 번째 주성분 방향
    main_direction = pca.components_[0]
    angle = np.arctan2(main_direction[1], main_direction[0])
    
    return create_rotated_bbox(coords_array, angle)
# ================================================================

def get_open_space_coordinates(nx: int, ny: int, obs_pos: List[int]) -> List[Tuple[int, int]]:
    """
    그리드에서 open space의 좌표들을 구합니다.
    
    Args:
        nx, ny: 그리드 크기
        obs_pos: 장애물 위치 리스트 (1차원 인덱스)
    
    Returns:
        List of (col, row) coordinates of open space
    """
    # 전체 셀 인덱스
    all_cells = set(range(nx * ny))
    
    # 장애물 제외
    open_cells = all_cells - set(obs_pos)
    
    # (col, row) 좌표로 변환
    open_coords = []
    for cell_idx in open_cells:
        row = cell_idx // nx
        col = cell_idx % nx
        open_coords.append((col, row))
    
    return open_coords


def fit_inscribed_rectangle(open_coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Open space에 내접하는 최대 사각형을 찾습니다.
    
    Args:
        open_coords: Open space 좌표 리스트 [(col, row), ...]
    
    Returns:
        (min_col, min_row, max_col, max_row) - 내접 사각형의 경계
    """
    if not open_coords:
        return (0, 0, 0, 0)
    
    # Open space를 그리드로 변환
    coords_array = np.array(open_coords)
    min_col, min_row = coords_array.min(axis=0)
    max_col, max_row = coords_array.max(axis=0)
    
    # 그리드 생성 (1: open, 0: obstacle/outside)
    grid_width = max_col - min_col + 1
    grid_height = max_row - min_row + 1
    grid = np.zeros((grid_height, grid_width), dtype=int)
    
    for col, row in open_coords:
        grid_col = col - min_col
        grid_row = row - min_row
        grid[grid_row, grid_col] = 1
    
    # 최대 직사각형 찾기 (히스토그램 기반 알고리즘)
    max_area = 0
    best_rect = (min_col, min_row, min_col, min_row)
    
    # 각 행에 대해 히스토그램 생성
    heights = np.zeros(grid_width)
    
    for row_idx in range(grid_height):
        # 히스토그램 업데이트
        for col_idx in range(grid_width):
            if grid[row_idx, col_idx] == 1:
                heights[col_idx] += 1
            else:
                heights[col_idx] = 0
        
        # 이 히스토그램에서 최대 직사각형 찾기
        area, left, right, height = largest_rectangle_in_histogram(heights)
        
        if area > max_area:
            max_area = area
            # 원래 좌표계로 변환
            rect_min_col = left + min_col
            rect_max_col = right + min_col
            rect_min_row = row_idx - height + 1 + min_row
            rect_max_row = row_idx + min_row
            best_rect = (rect_min_col, rect_min_row, rect_max_col, rect_max_row)
    
    return best_rect


def largest_rectangle_in_histogram(heights: np.ndarray) -> Tuple[int, int, int, int]:
    """
    히스토그램에서 최대 직사각형을 찾습니다.
    
    Returns:
        (area, left_idx, right_idx, height)
    """
    stack = []
    max_area = 0
    best_result = (0, 0, 0, 0)
    
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > max_area:
                max_area = area
                best_result = (area, idx, i - 1, height)
            start = idx
        stack.append((start, h))
    
    # 스택에 남은 요소들 처리
    for idx, height in stack:
        area = height * (len(heights) - idx)
        if area > max_area:
            max_area = area
            best_result = (area, idx, len(heights) - 1, height)
    
    return best_result


def fit_circumscribed_rectangle(open_coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Open space를 완전히 포함하는 최소 외접 사각형을 찾습니다.
    
    Args:
        open_coords: Open space 좌표 리스트 [(col, row), ...]
    
    Returns:
        (min_col, min_row, max_col, max_row) - 외접 사각형의 경계
    """
    if not open_coords:
        return (0, 0, 0, 0)
    
    coords_array = np.array(open_coords)
    min_col, min_row = coords_array.min(axis=0)
    max_col, max_row = coords_array.max(axis=0)
    
    return (min_col, min_row, max_col, max_row)


def fit_best_rectangle(open_coords: List[Tuple[int, int]], method: str = "pca") -> Tuple[int, int, int, int]:
    """
    Open space에 가장 잘 맞는 사각형을 찾습니다.
    
    Args:
        open_coords: Open space 좌표 리스트 [(col, row), ...]
        method: "pca", "convex_hull", "oriented_bbox" 중 하나
    
    Returns:
        (min_col, min_row, max_col, max_row) - 최적 사각형의 경계
    """
    if not open_coords:
        return (0, 0, 0, 0)
    
    coords_array = np.array(open_coords)
    
    if method == "pca":
        # PCA를 사용한 주성분 방향 기반 사각형
        if len(coords_array) < 2:
            return fit_circumscribed_rectangle(open_coords)
        
        # 중심점
        center = coords_array.mean(axis=0)
        
        # PCA 수행
        pca = PCA(n_components=2)
        pca.fit(coords_array)
        
        # 주성분 방향으로 투영
        transformed = pca.transform(coords_array)
        
        # 투영된 좌표의 경계
        min_proj = transformed.min(axis=0)
        max_proj = transformed.max(axis=0)
        
        # 원래 좌표계로 역변환
        corners_proj = np.array([
            [min_proj[0], min_proj[1]],
            [max_proj[0], min_proj[1]],
            [max_proj[0], max_proj[1]],
            [min_proj[0], max_proj[1]]
        ])
        
        corners_orig = pca.inverse_transform(corners_proj)
        
        # 축에 정렬된 bounding box로 근사
        min_col, min_row = corners_orig.min(axis=0)
        max_col, max_row = corners_orig.max(axis=0)
        
        return (int(min_col), int(min_row), int(max_col), int(max_row))
    
    elif method == "convex_hull":
        # Convex Hull 기반 사각형
        if len(coords_array) < 3:
            return fit_circumscribed_rectangle(open_coords)
        
        try:
            hull = ConvexHull(coords_array)
            hull_points = coords_array[hull.vertices]
            
            # Hull의 bounding box
            min_col, min_row = hull_points.min(axis=0)
            max_col, max_row = hull_points.max(axis=0)
            
            return (int(min_col), int(min_row), int(max_col), int(max_row))
        except:
            return fit_circumscribed_rectangle(open_coords)
    
    elif method == "oriented_bbox":
            # ConvexHull edge 방향 기반 단순 접근
        if len(coords_array) < 3:
            return fit_circumscribed_rectangle(open_coords)
        
        try:
            hull = ConvexHull(coords_array)
            hull_points = coords_array[hull.vertices]
            
            min_area = float('inf')
            best_rect = fit_circumscribed_rectangle(open_coords)
            
            # Hull의 각 edge 방향으로 회전하여 최소 bbox 찾기
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                
                # Edge 벡터
                edge = p2 - p1
                if np.linalg.norm(edge) == 0:
                    continue
                    
                # Edge 방향으로 정렬
                edge_angle = np.arctan2(edge[1], edge[0])
                
                # 회전 행렬 (edge가 x축이 되도록)
                cos_a = np.cos(-edge_angle)
                sin_a = np.sin(-edge_angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                # 모든 점들을 회전
                rotated_points = coords_array @ rotation_matrix.T
                
                # 회전된 좌표계에서 axis-aligned bbox
                min_rot = rotated_points.min(axis=0)
                max_rot = rotated_points.max(axis=0)
                
                area = (max_rot[0] - min_rot[0]) * (max_rot[1] - min_rot[1])
                
                if area < min_area:
                    min_area = area
                    
                    # 회전된 bbox의 4개 모서리
                    corners_rot = np.array([
                        [min_rot[0], min_rot[1]],
                        [max_rot[0], min_rot[1]],
                        [max_rot[0], max_rot[1]],
                        [min_rot[0], max_rot[1]]
                    ])
                    
                    # 원래 좌표계로 역회전
                    inv_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                    corners_orig = corners_rot @ inv_rotation.T
                    
                    # 회전된 사각형을 포함하는 axis-aligned bbox
                    min_col = int(np.floor(corners_orig[:, 0].min()))
                    max_col = int(np.ceil(corners_orig[:, 0].max()))
                    min_row = int(np.floor(corners_orig[:, 1].min()))
                    max_row = int(np.ceil(corners_orig[:, 1].max()))
                    
                    best_rect = (min_col, min_row, max_col, max_row)
            
            return best_rect
            
        except Exception as e:
            return fit_circumscribed_rectangle(open_coords)

    else:
        # 기본값: 외접 사각형
        return fit_circumscribed_rectangle(open_coords)


def approximate_open_space_rectangle(
    nx: int, 
    ny: int, 
    obs_pos: List[int], 
    method: Literal["inscribed", "circumscribed", "pca", "convex_hull", "oriented_bbox"] = "pca"
) -> Tuple[int, int, int, int]:
    """
    Open space를 사각형으로 근사화합니다.
    
    Args:
        nx, ny: 그리드 크기
        obs_pos: 장애물 위치 리스트 (1차원 인덱스)
        method: 근사화 방법
            - "inscribed": 내접 사각형 (open space 안에 완전히 들어가는 최대 사각형)
            - "circumscribed": 외접 사각형 (open space를 완전히 포함하는 최소 사각형)
            - "pca": PCA 기반 최적 사각형
            - "convex_hull": Convex Hull 기반 사각형
            - "oriented_bbox": 회전된 최소 bounding box
    
    Returns:
        (min_col, min_row, max_col, max_row) - 근사 사각형의 경계
    """
    # Open space 좌표 구하기
    open_coords = get_open_space_coordinates(nx, ny, obs_pos)
    
    if not open_coords:
        return (0, 0, 0, 0)
    
    # 선택된 방법에 따라 사각형 근사
    if method == "inscribed":
        return fit_inscribed_rectangle(open_coords)
    elif method == "circumscribed":
        return fit_circumscribed_rectangle(open_coords)
    else:
        return fit_best_rectangle(open_coords, method)


def visualize_rectangle_approximation(
    nx: int, 
    ny: int, 
    obs_pos: List[int], 
    initial_positions: List[int] = None,
    methods: List[str] = None
):
    """
    다양한 사각형 근사화 방법들을 시각화합니다.
    
    Args:
        nx, ny: 그리드 크기
        obs_pos: 장애물 위치 리스트
        initial_positions: UAV 초기 위치 리스트 (선택사항)
        methods: 시각화할 방법들 리스트
    """
    if methods is None:
        methods = ["inscribed", "circumscribed", "pca", "convex_hull", "oriented_bbox"]
    
    if initial_positions is None:
        initial_positions = []
    
    # Open space 좌표
    open_coords = get_open_space_coordinates(nx, ny, obs_pos)
    
    # 서브플롯 생성
    n_methods = len(methods)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # 색상 설정
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, method in enumerate(methods):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 기본 그리드 그리기
        grid = np.ones((ny, nx))  # 1: open space
        
        # 장애물 표시
        for obs_idx in obs_pos:
            obs_row = obs_idx // nx
            obs_col = obs_idx % nx
            if 0 <= obs_row < ny and 0 <= obs_col < nx:
                grid[obs_row, obs_col] = 0  # 0: obstacle
        
        ax.imshow(grid, cmap='RdYlBu', origin='lower', alpha=0.7)
        
        # 사각형 근사
        min_col, min_row, max_col, max_row = approximate_open_space_rectangle(
            nx, ny, obs_pos, method
        )
        
        # 사각형 그리기
        rect_width = max_col - min_col
        rect_height = max_row - min_row
        rect = patches.Rectangle((min_col-0.5, min_row-0.5), rect_width, rect_height,
                               linewidth=3, edgecolor=colors[idx % len(colors)], 
                               facecolor='none', label=f'{method.title()} Rectangle')
        ax.add_patch(rect)
        
        # UAV 위치 표시
        for i, uav_idx in enumerate(initial_positions):
            uav_row = uav_idx // nx
            uav_col = uav_idx % nx
            ax.plot(uav_col, uav_row, 'ko', markersize=8, markerfacecolor='yellow', 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(uav_col, uav_row, f'U{i+1}', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Open space 점들 표시 (샘플링)
        if len(open_coords) > 500:
            sample_coords = np.random.choice(len(open_coords), 500, replace=False)
            sample_open_coords = [open_coords[i] for i in sample_coords]
        else:
            sample_open_coords = open_coords
        
        if sample_open_coords:
            open_x, open_y = zip(*sample_open_coords)
            ax.scatter(open_x, open_y, c='lightblue', s=1, alpha=0.5, label='Open Space')
        
        ax.set_xlim(-1, nx)
        ax.set_ylim(-1, ny)
        ax.set_title(f'{method.title()} Rectangle\n({rect_width}×{rect_height}, Area={rect_width*rect_height})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 빈 서브플롯 숨기기
    for idx in range(n_methods, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 결과 요약
    print(f"\n=== Rectangle Approximation Results ===")
    print(f"Grid size: {nx}×{ny}")
    print(f"Open space cells: {len(open_coords)}")
    print(f"Obstacle cells: {len(obs_pos)}")
    print()
    
    for method in methods:
        min_col, min_row, max_col, max_row = approximate_open_space_rectangle(
            nx, ny, obs_pos, method
        )
        width = max_col - min_col
        height = max_row - min_row
        area = width * height
        coverage = area / len(open_coords) if open_coords else 0
        
        print(f"{method.title():15}: {width:2d}×{height:2d} (area={area:4d}, coverage={coverage:.2f})")


# 데모 함수
def demo_rectangle_approximation():
    """사각형 근사화 데모"""
    import random
    
    # 예시 환경 생성
    nx, ny = 40, 30
    
    # L자 형태의 장애물 생성
    obs_pos = []
    
    # 세로 막대
    for i in range(5, 25):
        for j in range(15, 20):
            obs_pos.append(i * nx + j)
    
    # 가로 막대
    for i in range(20, 25):
        for j in range(20, 35):
            obs_pos.append(i * nx + j)
    
    # 모서리에 몇 개 더
    corner_obs = [0, 1, 2, nx-1, nx-2, nx*(ny-1), nx*ny-1, nx*ny-2]
    obs_pos.extend(corner_obs)
    
    # UAV 위치
    initial_positions = [3*nx + 5, 8*nx + 35, 25*nx + 8]
    
    # 모든 방법으로 시각화
    visualize_rectangle_approximation(nx, ny, obs_pos, initial_positions)


if __name__ == "__main__":
    demo_rectangle_approximation()