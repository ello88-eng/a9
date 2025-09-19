import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from typing import List, Tuple, Dict
from itertools import combinations
# multiple inscribed bbox

def get_open_space_coordinates(nx: int, ny: int, obs_pos: List[int]) -> List[Tuple[int, int]]:
    """그리드에서 open space 좌표 추출"""
    all_cells = set(range(nx * ny))
    open_cells = all_cells - set(obs_pos)
    
    open_coords = []
    for cell_idx in open_cells:
        row = cell_idx // nx
        col = cell_idx % nx
        open_coords.append((col, row))
    
    return open_coords


def find_mixed_directions(open_coords, max_directions=8):
    """PCA 주성분 > Hull edge 순으로 방향 찾기"""
    coords_array = np.array(open_coords)
    directions = []
    
    # 1. PCA 주성분 방향 (최우선)
    if len(coords_array) >= 2:
        pca = PCA(n_components=2)
        pca.fit(coords_array)
        
        for component in pca.components_:
            angle = np.arctan2(component[1], component[0])
            directions.append(angle)
    
    # 2. Hull edge 방향들
    if len(coords_array) >= 3:
        try:
            hull = ConvexHull(coords_array)
            hull_points = coords_array[hull.vertices]
            
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                edge = p2 - p1
                if np.linalg.norm(edge) > 0:
                    angle = np.arctan2(edge[1], edge[0])
                    directions.append(angle)
        except:
            pass
    
    # 중복 제거 및 개수 제한
    directions = list(set([round(a, 3) for a in directions]))
    return directions[:max_directions]


class BBoxCandidate:
    """bbox 후보 클래스"""
    def __init__(self, min_col, min_row, max_col, max_row, angle=0):
        self.min_col = min_col
        self.min_row = min_row
        self.max_col = max_col
        self.max_row = max_row
        self.angle = angle
        self.width = max_col - min_col
        self.height = max_row - min_row
        self.area = self.width * self.height
        self.coverage = 0.0
        self.boundary_violation = 0.0
        self.score = 0.0


def generate_bbox_candidates_for_direction(open_coords, direction, min_width, min_height, 
                                         nx, ny, num_size_steps=5):
    """특정 방향에서 다양한 크기의 bbox 후보들 생성"""
    coords_array = np.array(open_coords)
    open_set = set(open_coords)
    candidates = []
    
    # 회전 변환
    cos_a = np.cos(-direction)
    sin_a = np.sin(-direction)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_coords = coords_array @ rotation_matrix.T
    min_rot = rotated_coords.min(axis=0)
    max_rot = rotated_coords.max(axis=0)
    
    # 회전된 공간에서 가능한 크기들
    # max_width = max_rot[0] - min_rot[0]
    # max_height = max_rot[1] - min_rot[1]
    max_width = 200
    max_height = 200

    # width_steps = np.linspace(min_width, max_width, num_size_steps)
    width_steps = np.linspace(max_width, min_width, num_size_steps)
    # height_steps = np.linspace(min_height, max_height, num_size_steps)
    height_steps = np.linspace(max_height, min_height, num_size_steps)
    
    # 다양한 크기와 위치에서 bbox 후보들 생성
    for width in width_steps:
        for height in height_steps:
            if width < min_width or height < min_height:
                continue
                
            # 가능한 위치들 탐색 (stride 사용으로 성능 최적화)
            stride = max(1, int(min(width, height) // 4))
            
            for start_x in range(int(min_rot[0]), int(max_rot[0] - width + 1), stride):
                for start_y in range(int(min_rot[1]), int(max_rot[1] - height + 1), stride):
                    
                    # 회전된 bbox 모서리들
                    corners_rot = np.array([
                        [start_x, start_y],
                        [start_x + width, start_y],
                        [start_x + width, start_y + height],
                        [start_x, start_y + height]
                    ])
                    
                    # 원래 좌표계로 역변환
                    inv_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                    corners_orig = corners_rot @ inv_rotation.T
                    
                    min_col = int(np.floor(corners_orig[:, 0].min()))
                    max_col = int(np.ceil(corners_orig[:, 0].max()))
                    min_row = int(np.floor(corners_orig[:, 1].min()))
                    max_row = int(np.ceil(corners_orig[:, 1].max()))
                    
                    # bbox 평가
                    bbox = BBoxCandidate(min_col, min_row, max_col, max_row, direction)
                    bbox.coverage, bbox.boundary_violation = evaluate_bbox(
                        bbox, open_set, nx, ny
                    )
                    
                    if bbox.coverage > 0:  # 유효한 bbox만 추가
                        candidates.append(bbox)
    
    return candidates


def evaluate_bbox(bbox, open_set, nx, ny, 
                 overlap_penalty=0.1, boundary_penalty=0.5):
    """bbox 평가: 커버리지와 경계 침범 계산"""
    
    # bbox 내부 픽셀들
    bbox_pixels = []
    for row in range(bbox.min_row, bbox.max_row + 1):
        for col in range(bbox.min_col, bbox.max_col + 1):
            bbox_pixels.append((col, row))
    
    # 커버리지 계산
    covered_pixels = 0
    boundary_violations = 0
    
    for pixel in bbox_pixels:
        if pixel in open_set:
            covered_pixels += 1
        elif (0 <= pixel[0] < nx and 0 <= pixel[1] < ny):
            # 그리드 내부 장애물
            boundary_violations += 1
        else:
            # 그리드 밖으로 나감
            boundary_violations += 1
    
    total_bbox_pixels = len(bbox_pixels)
    coverage = covered_pixels / total_bbox_pixels if total_bbox_pixels > 0 else 0
    boundary_violation = boundary_violations / total_bbox_pixels if total_bbox_pixels > 0 else 0
    
    # 점수 계산
    # score = coverage - boundary_penalty * boundary_violation
    score = coverage * 2.0 - boundary_penalty * boundary_violation - overlap_penalty * 0.05

    bbox.score = score
    
    return coverage, boundary_violation


def calculate_bbox_overlap(bbox1, bbox2):
    """두 bbox 간 겹침 비율 계산"""
    # 겹치는 영역 계산
    overlap_min_col = max(bbox1.min_col, bbox2.min_col)
    overlap_max_col = min(bbox1.max_col, bbox2.max_col)
    overlap_min_row = max(bbox1.min_row, bbox2.min_row)
    overlap_max_row = min(bbox1.max_row, bbox2.max_row)
    
    if overlap_min_col >= overlap_max_col or overlap_min_row >= overlap_max_row:
        return 0.0
    
    overlap_area = (overlap_max_col - overlap_min_col) * (overlap_max_row - overlap_min_row)
    total_area = bbox1.area + bbox2.area
    
    return overlap_area / total_area if total_area > 0 else 0.0


def select_optimal_bbox_combination(candidates, robot_count, overlap_penalty=0.1):
    """최적 bbox 조합 선택"""
    
    # 1. 후보들을 점수순으로 정렬
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    # 2. 상위 후보들만 고려 (성능 최적화)
    top_candidates = candidates[:min(50, len(candidates))]
    
    best_combination = []
    best_score = -float('inf')
    
    # 3. 가능한 조합들 탐색 (최대 robot_count개)
    for num_bboxes in range(1, min(robot_count + 1, len(top_candidates) + 1)):
        for combination in combinations(top_candidates, num_bboxes):
            
            # 조합 점수 계산
            total_coverage = sum(bbox.coverage for bbox in combination)
            total_boundary_violation = sum(bbox.boundary_violation for bbox in combination)
            
            # 겹침 페널티 계산
            overlap_penalty_total = 0.0
            for i in range(len(combination)):
                for j in range(i + 1, len(combination)):
                    overlap_ratio = calculate_bbox_overlap(combination[i], combination[j])
                    overlap_penalty_total += overlap_ratio
            
            combination_score = (total_coverage - 
                               0.5 * total_boundary_violation - 
                               overlap_penalty * overlap_penalty_total)
            
            if combination_score > best_score:
                best_score = combination_score
                best_combination = list(combination)
    
    return best_combination, best_score


def multi_inscribed_bbox_packing(nx, ny, obs_pos, min_width, min_height, robot_count,
                                overlap_penalty=0.1, boundary_penalty=0.5, 
                                max_directions=8, time_limit_seconds=10):
    """
    메인 함수: 그리드 기반 다중 inscribed bbox 패킹
    
    Args:
        nx, ny: 그리드 크기
        obs_pos: 장애물 위치 리스트
        min_width, min_height: bbox 최소 크기 (hard constraint)
        robot_count: 로봇 수 (bbox 개수 제한)
        overlap_penalty: bbox 겹침 페널티
        boundary_penalty: 경계 침범 페널티  
        max_directions: 최대 탐색 방향 수
        time_limit_seconds: 시간 제한
    
    Returns:
        List[BBoxCandidate]: 최적 bbox 조합
    """
    start_time = time.time()
    print(f"Starting multi-inscribed bbox packing...")
    print(f"Grid: {nx}x{ny}, Obstacles: {len(obs_pos)}, Robots: {robot_count}")
    print(f"Min bbox size: {min_width}x{min_height}")
    
    # 1. Open space 좌표 추출
    open_coords = get_open_space_coordinates(nx, ny, obs_pos)
    print(f"Open space cells: {len(open_coords)}")
    
    if not open_coords:
        return []
    
    # 2. 방향 찾기 (PCA > Hull edge 순)
    directions = find_mixed_directions(open_coords, max_directions)
    print(f"Found {len(directions)} directions")
    
    # 3. 각 방향별로 bbox 후보 생성
    all_candidates = []
    
    for i, direction in enumerate(directions):
        if time.time() - start_time > time_limit_seconds:
            print(f"Time limit reached at direction {i}")
            break
            
        print(f"Processing direction {i+1}/{len(directions)}: {np.degrees(direction):.1f}°")
        
        candidates = generate_bbox_candidates_for_direction(
            open_coords, direction, min_width, min_height, nx, ny
        )
        
        print(f"  Generated {len(candidates)} candidates")
        all_candidates.extend(candidates)
    
    print(f"Total candidates: {len(all_candidates)}")
    
    # 4. 최적 조합 선택
    if not all_candidates:
        print("No valid candidates found")
        return []
    
    print("Selecting optimal combination...")
    optimal_bboxes, best_score = select_optimal_bbox_combination(
        all_candidates, robot_count, overlap_penalty
    )
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f}s")
    print(f"Selected {len(optimal_bboxes)} bboxes with score: {best_score:.3f}")
    
    return optimal_bboxes


def visualize_multi_bbox_result(nx, ny, obs_pos, bboxes, title="Multi Inscribed BBox Result"):
    """결과 시각화"""
    plt.figure(figsize=(12, 8))
    
    # 그리드 생성
    grid = np.ones((ny, nx))
    for obs in obs_pos:
        row, col = obs // nx, obs % nx
        if 0 <= row < ny and 0 <= col < nx:
            grid[row, col] = 0
    
    plt.imshow(grid, cmap='RdYlBu', origin='lower', alpha=0.7)
    
    # bbox들 그리기
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    total_area = 0
    
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        
        rect = plt.Rectangle((bbox.min_col, bbox.min_row), bbox.width, bbox.height,
                           fill=False, edgecolor=color, linewidth=2, alpha=0.8,
                           label=f'BBox {i+1}: {bbox.width}×{bbox.height} (score:{bbox.score:.2f})')
        plt.gca().add_patch(rect)
        
        total_area += bbox.area
        
        # 중심에 번호 표시
        center_x = (bbox.min_col + bbox.max_col) / 2
        center_y = (bbox.min_row + bbox.max_row) / 2
        plt.text(center_x, center_y, f'{i+1}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)
    
    plt.title(f'{title}\nTotal BBoxes: {len(bboxes)}, Total Area: {total_area}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 테스트 함수들
def create_test_shapes():
    """다양한 테스트 형태 생성"""
    
    def create_l_shape():
        nx, ny = 30, 20
        obs_pos = []
        for i in range(ny):
            for j in range(nx):
                vertical = (j >= 5 and j <= 12 and i >= 3 and i <= 17)
                horizontal = (j >= 12 and j <= 25 and i >= 12 and i <= 17)
                if not (vertical or horizontal):
                    obs_pos.append(i * nx + j)
        return nx, ny, obs_pos, "L Shape"
    
    def create_complex_shape():
        nx, ny = 40, 30
        obs_pos = []
        for i in range(ny):
            for j in range(nx):
                # 복잡한 형태: Y + 추가 가지들
                center_x, center_y = 20, 15
                
                # Y의 3개 가지
                branch1 = (abs(j - center_x) <= 2 and i >= center_y and i <= center_y + 10)
                branch2 = (i <= center_y + 2 and i >= center_y - 8 and 
                          abs((i - center_y) + (j - center_x) * 0.7) <= 2)
                branch3 = (i <= center_y + 2 and i >= center_y - 8 and 
                          abs((i - center_y) - (j - center_x) * 0.7) <= 2)
                
                # 추가 가지
                branch4 = (j >= center_x - 2 and j <= center_x + 15 and 
                          abs(i - (center_y + 5)) <= 1.5)
                
                if not (branch1 or branch2 or branch3 or branch4):
                    obs_pos.append(i * nx + j)
        return nx, ny, obs_pos, "Complex Y+ Shape"
    
    return [create_l_shape(), create_complex_shape()]


def demo_multi_inscribed_bbox():
    """데모 실행"""
    test_cases = create_test_shapes()
    
    for nx, ny, obs_pos, name in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        
        # 파라미터 설정
        min_width, min_height = 8, 6  # 로봇 제약 조건
        robot_count = 4
        
        # bbox 패킹 실행
        optimal_bboxes = multi_inscribed_bbox_packing(
            nx, ny, obs_pos, min_width, min_height, robot_count,
            time_limit_seconds=5
        )
        
        # 결과 시각화
        visualize_multi_bbox_result(nx, ny, obs_pos, optimal_bboxes, f"Result: {name}")
        
        # 상세 정보 출력
        print(f"\nDetailed Results:")
        total_coverage = 0
        for i, bbox in enumerate(optimal_bboxes):
            print(f"BBox {i+1}: {bbox.width}×{bbox.height}, "
                  f"Coverage: {bbox.coverage:.3f}, "
                  f"Boundary violation: {bbox.boundary_violation:.3f}, "
                  f"Score: {bbox.score:.3f}")
            total_coverage += bbox.coverage
        
        print(f"Average coverage: {total_coverage / len(optimal_bboxes):.3f}")


if __name__ == "__main__":
    demo_multi_inscribed_bbox()