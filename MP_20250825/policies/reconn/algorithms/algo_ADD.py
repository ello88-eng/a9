import random
import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import SurveillanceSelectionResult
from manager.manager import Manager
from policies.reconn.algorithms.algo_utils import same_size_clustering
from utils.coordinates import convert_waypoints_enu_to_lla
from utils.recognition import get_avail_avs, get_avail_trg


class ADDReconnaissanceAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        #
        mp_output_dict: Dict[int, SurveillanceSelectionResult] = {}
        # 정찰이 필요한 표적 확인
        trg_list, trg_pos_list = get_avail_trg(
            trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
            trg_state_dict=self.manager.trg_to_state_dict,
            expected_state=0,
        )
        avs_list, avs_pos_list = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["R"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )
        start_time = time.time()
        # Same size clustering
        cluster_size = 5
        num_cluster = int(np.floor(len(avs_list) / cluster_size))
        is_ok = False

        while not is_ok:
            try:
                # K-means clustering
                k_means = KMeans(n_clusters=num_cluster, random_state=random.randint(0, 100)).fit(avs_pos_list)
                cluster_label = k_means.labels_
                # same size clustering
                label, centers, _ = same_size_clustering(
                    avs_pos_list, cluster_size, num_cluster, cluster_label, 100000
                )
                is_ok = True
            except:
                is_ok = False

        # row: cluster center, col: task pos
        dist_mat = cdist(centers, trg_pos_list)
        # local index
        cluster_id, trg_ids = linear_sum_assignment(dist_mat)
        for i, trg_id in enumerate(trg_ids):
            c = cluster_id[i]
            avs_ids = []
            for p1 in range(0, len(np.argwhere(label == c))):
                avs_ids.append(avs_list[int(np.argwhere(label == c)[p1])])

            for avs_id in avs_ids:
                trg_pt = np.array(trg_pos_list[trg_id])
                wps = trg_pt.copy()
                wps[-1] += 300
                mp_output_dict.update(
                    {
                        avs_id: SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=trg_list[trg_id],
                            num_avs_per_group=5,
                            smp_mode=2,
                            speed=27,
                            waypoint_count=1,
                            waypoints=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=wps,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            turn_radius=300,
                            turn_dir=0,
                            trg_pt=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=trg_pt,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_list[trg_id],
                            attack_prio=0,
                            is_bd=0,
                        )
                    }
                )
        return time.time() - start_time, mp_output_dict, deepcopy(self.manager.mp_input)


class ADDPartialReconnaissanceAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, SurveillanceSelectionResult], MissionPlanInput]:
        #
        mp_output_dict: Dict[int, SurveillanceSelectionResult] = {}

        # 정찰이 필요한 표적 확인
        trg_list, _ = get_avail_trg(
            trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
            trg_state_dict=self.manager.trg_to_state_dict,
            expected_state=1,
        )
        start_time = time.time()

        for trg_id in trg_list:
            for avs_id in self.manager.smp_group_id_to_avs_dict[trg_id]:
                trg_pt = np.array(self.manager.mp_input.trg_fus_res_dict[trg_id].position)
                wps = trg_pt.copy()
                wps[-1] += 300
                mp_output_dict.update(
                    {
                        avs_id: SurveillanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=self.manager.avs_to_smp_group_id_dict[avs_id],
                            num_avs_per_group=5,
                            smp_mode=2,
                            speed=27,
                            waypoint_count=1,
                            waypoints=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=wps,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            turn_radius=300,
                            turn_dir=0,
                            trg_pt=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=trg_pt,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_id,
                            attack_prio=0,
                            is_bd=0,
                        )
                    }
                )
        return time.time() - start_time, mp_output_dict, deepcopy(self.manager.mp_input)
