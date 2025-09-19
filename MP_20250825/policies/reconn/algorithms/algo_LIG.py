import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from commu.sys.receiver.data.gcs2mps import AvsInfo
from commu.sys.receiver.data.sys_in import MissionPlanInput
from commu.sys.sender.data.sys_out import ReconnaissanceSelectionResult
from config.avs_config import SmpMode
from config.mp_config import MAX_GROUPS_OF_5, REPLAN_MODE
from manager.manager import Manager
from utils.coordinates import convert_waypoints_enu_to_lla
from utils.recognition import get_avail_avs, get_avail_trg


class LIGReconnaissanceAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult], MissionPlanInput]:
        """20250609

        Returns:
            Tuple[float, Dict[int, MissionPlanOutput]]: _description_
        """

        #
        start_time = time.time()

        #
        mp_selection_result_dict: Dict[int, ReconnaissanceSelectionResult] = {}

        # 임무 재계획 모드가 `수동`이면 운용자 요청 기준으로 표적 확인
        if REPLAN_MODE == "Manual":
            if self.manager.mp_input.mp_cmd_dict:
                # 정찰이 필요한 표적 확인
                trg_list = [
                    trg_id
                    for trg_id, mp_cmd in self.manager.mp_input.mp_cmd_dict.items()
                    if mp_cmd.reconn_request == 1
                ]
                trg_pos_list = [
                    self.manager.mp_input.trg_fus_res_dict[trg_id].position.tolist() for trg_id in trg_list
                ]
        else:
            trg_list, trg_pos_list = get_avail_trg(
                trg_fus_res_dict=self.manager.mp_input.trg_fus_res_dict,
                trg_state_dict=self.manager.trg_to_state_dict,
                expected_state=0,
            )

        # 정찰 선별이 가능한 비행체 확인
        avs_list, avs_pos_list = get_avail_avs(
            avs_to_avail_task=self.manager.avs_to_available_task_dict,
            criteria=["R"],
            avs_info_dict=self.manager.mp_input.avs_info_dict,
        )

        # Convert to NumPy arrays for vectorized operations
        avs_pos_array = np.array(avs_pos_list)  # Shape: (N, 3)
        trg_pos_array = np.array(trg_pos_list)  # Shape: (M, 3)
        avs_ids_array = np.array(avs_list)

        # Track available drone indices and assignments
        avail_avs_mask = np.ones(len(avs_list), dtype=bool)
        groups_of_5_count = 0

        # Special case: If total drones < 4, assign all to first target
        if len(avs_list) < 4:
            for trg_idx, trg_pos in enumerate(trg_pos_array):
                # Create MissionPlanOutput for each assigned drone
                for avs_id in avs_list:
                    trg_pos_as_enu = np.array(trg_pos_list[trg_idx])
                    trg_pos_array_as_lla = convert_waypoints_enu_to_lla(
                        waypoint_array_as_enu=trg_pos_as_enu,
                        lat0=self.manager.lat0,
                        lon0=self.manager.lon0,
                        alt0=self.manager.alt0,
                    )
                    avs_info: AvsInfo = self.manager.mp_input.avs_info_dict.get(avs_id)
                    mp_selection_result_dict.update(
                        {
                            avs_id: ReconnaissanceSelectionResult(
                                avs_id=avs_id,
                                system_group_id=avs_info.system_group_id,
                                smp_group_id=trg_list[trg_idx],
                                smp_mode=SmpMode.RECONN.value,
                                coordinate=trg_pos_array_as_lla[:, :-1],
                                turning_radius=self.manager.mp_input.mission_init_info.loiter_radius,
                                turning_direction=self.manager.mp_input.mission_init_info.loiter_direction,
                                target_position=trg_pos_array_as_lla,
                                target_id=trg_list[trg_idx],
                            )
                        }
                    )

            return time.time() - start_time, mp_selection_result_dict, deepcopy(self.manager.mp_input)

        # Process each target
        # Normal case: Process each target with standard rules
        for trg_idx, trg_pos in enumerate(trg_pos_array):
            # Get indices of available drones
            avail_indices = np.where(avail_avs_mask)[0]
            num_remaining_uav = len(avail_indices)

            # Rule 4: If less than 4 drones remaining, stop assignment
            if num_remaining_uav < 4:
                break

            # Modified Rule 3: Determine assignment size based on max_groups_of_5 limit
            if MAX_GROUPS_OF_5 is not None and groups_of_5_count >= MAX_GROUPS_OF_5:
                # Force 4-drone assignment if max groups of 5 reached
                num_uav_to_assign = 4
            elif num_remaining_uav < 5:
                # Assign 4 drones if less than 5 remaining
                num_uav_to_assign = 4
            else:
                # Normal case: assign 5 drones
                num_uav_to_assign = 5

            # Vectorized distance calculation for all available drones
            avail_pos_array = avs_pos_array[avail_indices]  # Shape: (remaining_drones, 3)

            # Calculate squared distances (avoid sqrt for speed, since we only need relative ordering)
            diff = avail_pos_array - trg_pos  # Broadcasting: (remaining_drones, 3)
            squared_distances = np.sum(diff**2, axis=1)  # Shape: (remaining_drones,)

            # Rule 1: Find indices of closest drones using argpartition (faster than full sort)
            if num_uav_to_assign < num_remaining_uav:
                closest_indices = np.argpartition(squared_distances, num_uav_to_assign)[:num_uav_to_assign]
            else:
                closest_indices = np.arange(num_remaining_uav)

            # Get the actual drone indices and IDs
            selected_uav_indices = avail_indices[closest_indices]
            closest_uav_ids = avs_ids_array[selected_uav_indices].tolist()

            # Create MissionPlanOutput for each assigned drone
            for avs_id in closest_uav_ids:
                trg_pos_as_enu = np.array(trg_pos_list[trg_idx])
                trg_pos_array_as_lla = convert_waypoints_enu_to_lla(
                    waypoint_array_as_enu=trg_pos_as_enu,
                    lat0=self.manager.lat0,
                    lon0=self.manager.lon0,
                    alt0=self.manager.alt0,
                )
                avs_info: AvsInfo = self.manager.mp_input.avs_info_dict.get(avs_id)
                mp_selection_result_dict.update(
                    {
                        avs_id: ReconnaissanceSelectionResult(
                            avs_id=avs_id,
                            system_group_id=avs_info.system_group_id,
                            smp_group_id=trg_list[trg_idx],
                            smp_mode=SmpMode.RECONN.value,
                            coordinate=trg_pos_array_as_lla[:, :-1],
                            turning_radius=self.manager.mp_input.mission_init_info.loiter_radius,
                            turning_direction=self.manager.mp_input.mission_init_info.loiter_direction,
                            target_position=trg_pos_array_as_lla,
                            target_id=trg_list[trg_idx],
                        )
                    }
                )

            # Rule 2: Assign drones to target and mark them as unavailable
            avail_avs_mask[selected_uav_indices] = False

            # Update groups_of_5_count if 5 drones were assigned
            if num_uav_to_assign == 5:
                groups_of_5_count += 1

        return time.time() - start_time, mp_selection_result_dict, deepcopy(self.manager.mp_input)


class LIGPartialReconnaissanceAlgorithmRunner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def select(self) -> Tuple[float, Dict[int, ReconnaissanceSelectionResult], MissionPlanInput]:
        """20250609

        Returns:
            Tuple[float, Dict[int, MissionPlanOutput]]: _description_
        """
        mp_output_dict: Dict[int, ReconnaissanceSelectionResult] = {}
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

        # Convert to NumPy arrays for vectorized operations
        avs_pos_array = np.array(avs_pos_list)  # Shape: (N, 3)
        trg_pos_array = np.array(trg_pos_list)  # Shape: (M, 3)
        avs_ids_array = np.array(avs_list)

        # Track available drone indices and assignments
        avail_avs_mask = np.ones(len(avs_list), dtype=bool)
        groups_of_5_count = 0

        # Special case: If total drones < 4, assign all to first target
        if len(avs_list) < 4:
            for trg_idx, trg_pos in enumerate(trg_pos_array):
                # Create MissionPlanOutput for each assigned drone
                for avs_id in avs_list:
                    trg_pt = np.array(trg_pos_list[trg_idx])
                    wps = convert_waypoints_enu_to_lla(
                        waypoint_array_as_enu=trg_pt,
                        lat0=self.manager.lat0,
                        lon0=self.manager.lon0,
                        alt0=self.manager.alt0,
                    )
                    wps[:, -1] += 300
                    mp_output_dict.update(
                        {
                            avs_id: ReconnaissanceSelectionResult(
                                avs_id=avs_id,
                                approval_flag=0,
                                sys_group_id=trg_list[trg_idx],
                                num_avs_per_group=len(avs_list),
                                smp_mode=2,
                                speed=27,
                                waypoint_count=1,
                                waypoints=wps,
                                turn_radius=300,
                                turn_dir=0,
                                trg_pt=convert_waypoints_enu_to_lla(
                                    waypoint_array_as_enu=trg_pt,
                                    lat0=self.manager.lat0,
                                    lon0=self.manager.lon0,
                                    alt0=self.manager.alt0,
                                ),
                                trg_id=trg_list[trg_idx],
                                attack_prio=0,
                                is_bd=0,
                            )
                        }
                    )

            return (
                time.time() - start_time,
                mp_output_dict,
                deepcopy(self.manager.mp_input),
            )

        # Process each target
        # Normal case: Process each target with standard rules
        for trg_idx, trg_pos in enumerate(trg_pos_array):
            # Get indices of available drones
            avail_indices = np.where(avail_avs_mask)[0]
            num_remaining_uav = len(avail_indices)

            # Rule 4: If less than 4 drones remaining, stop assignment
            if num_remaining_uav < 4:
                break

            # Modified Rule 3: Determine assignment size based on max_groups_of_5 limit
            if MAX_GROUPS_OF_5 is not None and groups_of_5_count >= MAX_GROUPS_OF_5:
                # Force 4-drone assignment if max groups of 5 reached
                num_uav_to_assign = 4
            elif num_remaining_uav < 5:
                # Assign 4 drones if less than 5 remaining
                num_uav_to_assign = 4
            else:
                # Normal case: assign 5 drones
                num_uav_to_assign = 5

            # Vectorized distance calculation for all available drones
            avail_pos_array = avs_pos_array[avail_indices]  # Shape: (remaining_drones, 3)

            # Calculate squared distances (avoid sqrt for speed, since we only need relative ordering)
            diff = avail_pos_array - trg_pos  # Broadcasting: (remaining_drones, 3)
            squared_distances = np.sum(diff**2, axis=1)  # Shape: (remaining_drones,)

            # Rule 1: Find indices of closest drones using argpartition (faster than full sort)
            if num_uav_to_assign < num_remaining_uav:
                closest_indices = np.argpartition(squared_distances, num_uav_to_assign)[:num_uav_to_assign]
            else:
                closest_indices = np.arange(num_remaining_uav)

            # Get the actual drone indices and IDs
            selected_uav_indices = avail_indices[closest_indices]
            closest_uav_ids = avs_ids_array[selected_uav_indices].tolist()

            # Create MissionPlanOutput for each assigned drone
            for avs_id in closest_uav_ids:
                trg_pt = np.array(trg_pos_list[trg_idx])
                wps = convert_waypoints_enu_to_lla(
                    waypoint_array_as_enu=trg_pt,
                    lat0=self.manager.lat0,
                    lon0=self.manager.lon0,
                    alt0=self.manager.alt0,
                )
                wps[:, -1] += 300
                mp_output_dict.update(
                    {
                        avs_id: ReconnaissanceSelectionResult(
                            avs_id=avs_id,
                            approval_flag=0,
                            sys_group_id=trg_list[trg_idx],
                            num_avs_per_group=num_uav_to_assign,
                            smp_mode=2,
                            speed=27,
                            waypoint_count=1,
                            waypoints=wps,
                            turn_radius=300,
                            turn_dir=0,
                            trg_pt=convert_waypoints_enu_to_lla(
                                waypoint_array_as_enu=trg_pt,
                                lat0=self.manager.lat0,
                                lon0=self.manager.lon0,
                                alt0=self.manager.alt0,
                            ),
                            trg_id=trg_list[trg_idx],
                            attack_prio=0,
                            is_bd=0,
                        )
                    }
                )

            # Rule 2: Assign drones to target and mark them as unavailable
            avail_avs_mask[selected_uav_indices] = False

            # Update groups_of_5_count if 5 drones were assigned
            if num_uav_to_assign == 5:
                groups_of_5_count += 1

        return time.time() - start_time, mp_output_dict, deepcopy(self.manager.mp_input)
