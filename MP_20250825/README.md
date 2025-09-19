# 파일 목록
1. polygon_grid_projector.py
    - 실제 모듈에 넣어서 사용할 코드
    - ※ class로 만들어서 사용하도록 했습니다.

2. multiRobotPathPalnner.py
    - 사용 예제 (line 378 이후 참조)

## 사용예제로 설명

- 입력 예제
```python
    grid_size = 200
    # Map conversion
    left_top = [36.460396, 126.521423]
    left_bottom = [36.423464, 126.520794]
    right_top = [36.460105, 126.571651]
    right_bottom = [36.424007, 126.572087]

    roi_points = [left_top, left_bottom, right_top, right_bottom]

    polygon_points = [
    [36.460396, 126.526000],  # 1) 윗변-왼쪽(상단 베이스 시작; top edge 위)
    [36.460396, 126.566000],  # 2) 윗변-오른쪽(상단 베이스 끝; top edge 위)
    [36.452500, 126.546000],  # 3) 윗삼각형 '아래' 꼭짓점(허리 쪽, 중앙 부근)
    [36.444000, 126.551000],  # 4) 허리 직사각형 상단-오른쪽
    [36.440000, 126.551000],  # 5) 허리 직사각형 하단-오른쪽
    [36.423464, 126.569000],  # 6) 아랫변-오른쪽(하단 베이스 끝; bottom edge 위)
    [36.423464, 126.522000],  # 7) 아랫변-왼쪽(하단 베이스 시작; bottom edge 위)
    [36.440000, 126.541000],  # 8) 허리 직사각형 하단-왼쪽
    [36.444000, 126.541000],  # 9) 허리 직사각형 상단-왼쪽
    ]

```
- Obstacle index `obs_pos` 생성
```python
    # 클래스 초기화
    pgp = PolygonGridProjector(grid_size=grid_size)

    # obs_pos
    obs_pos, _, bbox = pgp.obs_pos_from_polygon_lonlat(
    roi_points, polygon_points, use_hull=False  # 필요 시 True, # Default : False)
    )
    # False : polygon_points 그대로 사용
    # True : polygon_points가 너무 복잡할 경우를 대비해 외각 convex로 사용
```

- Agent pos `in_pos`를 free space 안으로 projection
```python
    # in_pos를 자유공간으로 투영 (장애물에서 3칸 이상)
    in_pos_safe = pgp.project_in_pos_with_clearance(obs_pos=obs_pos, in_pos=in_pos, min_clearance=3)

    ...

    args.in_pos = in_pos_safe # 기존 in_pos 를 in_pos_safe 로 대체하여 사용
```

## TODO
1. ICD 로부터 `roi_points`, `polygon_points` 리스트 생성
2. in_pos 가 free space에 제대로 매핑되는지 점검 필요
3. 자체 동작 점검
    - 단순 코드 점검은 했으나, 시간상 충분한 점검을 하지는 못했습니다. 점검 필요합니다.
