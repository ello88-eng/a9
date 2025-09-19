from typing import Dict, Iterable, List, Tuple

from commu.sys.receiver.data.fs2mps import TargetFusionResult
from commu.sys.receiver.data.gcs2mps import AvsInfo


def check_mission(tasks: List[str], criteria: Iterable[str]) -> bool:
    """주어진 tasks 리스트에 criteria 중 하나라도 포함되면 True. 대소문자 구분은 그대로 유지.

    Args:
        tasks (List[str]): _description_
        criteria (Iterable[str]): _description_

    Returns:
        bool: _description_
    """
    criteria_set = set(criteria)
    # 교집합이 있으면 True
    return bool(criteria_set & set(tasks))


def get_avail_avs(
    avs_to_avail_task: Dict[int, List[str]], criteria: Iterable[str], avs_info_dict: Dict[int, AvsInfo]
) -> Tuple[List[int], List[List[float]]]:
    """criteria 조건에 맞는 AVS ID와 해당 위치를 반환

    Args:
        avs_to_avail_task (Dict[int, List[str]]): _description_
        criteria (Iterable[str]): _description_
        avs_info_dict (Dict[int, AvsInfo]): _description_

    Returns:
        Tuple[List[int], List[List[float]]]: _description_
    """
    avs_id_list: List[int] = []
    avs_pos_list: List[List[float]] = []

    # TODO: avs_to_avail_task 이랑 infos가 서로 싱크가 안 맞음
    for avs_id, tasks in avs_to_avail_task.items():
        if check_mission(tasks=tasks, criteria=criteria):
            avs_info = avs_info_dict.get(avs_id)
            if avs_info is None:
                continue
            avs_id_list.append(avs_id)
            avs_pos_list.append(avs_info.position.tolist())

    return avs_id_list, avs_pos_list


def get_avail_trg(
    trg_fus_res_dict: Dict[int, TargetFusionResult], trg_state_dict: Dict[int, int], expected_state: int = 0
) -> Tuple[List[int], List[List[float]]]:
    """상태가 expected_state인 표적의 ID 목록과 위치 목록을 반환.

    Args:
        trg_fus_res_dict (Dict[int, TargetFusionResult]): _description_
        trg_state_dict (Dict[int, int]): _description_
        expected_state (int, optional): _description_. Defaults to 0.

    Returns:
        Tuple[List[int], List[List[float]]]: _description_
    """
    pairs = [
        (trg_id, res.position.tolist())
        for trg_id, res in trg_fus_res_dict.items()
        if trg_state_dict.get(trg_id) == expected_state
    ]

    if not pairs:
        return [], []
    trg_ids, trg_positions = map(list, zip(*pairs))

    return trg_ids, trg_positions
