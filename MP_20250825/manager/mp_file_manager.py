import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List
from xml.etree.ElementTree import Element

import numpy as np

from commu.sys.receiver.data.gcs2mps import AvsInfo, Boundary, MissionInitialInfo
from commu.sys.receiver.data.mp_file import AvsLastWaypoint, Waypoint
from commu.sys.receiver.data.sys_in import MissionPlanInput
from config.avs_config import FlightControlMode, SmpMode, SystemMode
from config.main_config import MainArgs
from config.mp_config import ALGO_TO_NAME_DICT, NUM_COLS, NUM_ROWS, SHOW_OUTPUT
from manager.manager import Manager
from planner.planner import Planner
from utils.coordinates import sort_coordinates_as_clockwise, split_origin, true_round


class MissionPlanFileManager:
    def __init__(self, args: MainArgs, mission_init_info: MissionInitialInfo) -> None:
        self.args = args
        self.mission_init_info = mission_init_info
        self.avs_last_wp_list: List[AvsLastWaypoint] = []

    def _parse_mission_file_path(self) -> None:
        tree = ET.parse(self.mission_init_info.mp_file_path)
        root = tree.getroot()

        def _parse_boundary() -> Boundary:
            map_area = root.find("MAP_AREA")
            if map_area is None:
                # 없으면 빈 꼭지점
                return Boundary(vertices=np.zeros((0, 2), dtype=float))

            polygon = map_area.find("Polygon")
            if polygon is None:
                return Boundary(vertices=np.zeros((0, 2), dtype=float))

            points = polygon.findall("Point")
            if not points:
                return Boundary(vertices=np.zeros((0, 2), dtype=float))

            verts: List[List[float]] = []
            for point in points:
                lat = float(point.findtext("Latitude"))
                lon = float(point.findtext("Longitude"))
                verts.append([lat, lon])

            return Boundary(np.asarray(sort_coordinates_as_clockwise(verts), dtype=float))

        # FLIGHTROUTE ID -> 마지막 Waypoint 매핑
        last_wp_map = {}
        for fr in root.findall("FLIGHTROUTE"):
            fr_id = int(fr.findtext("ID"))
            waypoints = fr.find("Waypoints").findall("Waypoint")
            # Number 순서대로 정렬한 뒤 마지막 요소 선택
            waypoints_sorted: List[Element] = sorted(waypoints, key=lambda w: true_round(w.findtext("Number")))
            last: Element = waypoints_sorted[-1]
            last_wp_map[fr_id] = Waypoint(
                waypoint_id=int(last.findtext("ID")),
                position=np.array(
                    [
                        float(last.findtext("Latitude")),
                        float(last.findtext("Longitude")),
                        float(last.findtext("Altitude")),
                    ]
                ),
            )

        for avs in root.findall("AVS"):
            vid = int(avs.findtext("VehicleID"))
            gid = int(avs.findtext("GroupID"))
            mpid = int(avs.findtext("MissionProfileID"))
            if mpid not in last_wp_map:
                raise ValueError(f"MissionProfileID={mpid}에 해당하는 FLIGHTROUTE가 없습니다.")
            self.avs_last_wp_list.append(
                AvsLastWaypoint(
                    avs_id=vid, system_group_id=gid, mission_profile_id=mpid, last_waypoint=last_wp_map[mpid]
                )
            )

        # self.boundary = _parse_boundary()

    def _convert_avs_last_wp_list_to_avs_info_dict(self, temp_mp_input: MissionPlanInput) -> None:
        lat0, lon0, alt0 = split_origin(temp_mp_input.boundary.vertices)

        avs_info_dict: Dict[int, AvsInfo] = {}
        for avs_last_wp in self.avs_last_wp_list:
            avs_info = AvsInfo(
                avs_id=avs_last_wp.avs_id,
                system_group_id=avs_last_wp.system_group_id,
                system_mode=SystemMode.MOVE,
                flight_control_mode=FlightControlMode.AUTO,
                implement_mode=SmpMode.STAND_BY,
                position=avs_last_wp.last_waypoint.position,
            )
            avs_info_dict[avs_last_wp.avs_id] = avs_info

        # self.avs_info_dict = convert_avs_pos_lla_to_enu(infos=avs_info_dict, lat0=lat0, lon0=lon0, alt0=alt0)
        self.avs_info_dict = avs_info_dict

    def _preprocess(self, temp_mp_input: MissionPlanInput):
        self.manager = Manager(
            mission_list=["Surv"], num_rows=NUM_ROWS, num_columns=NUM_COLS, surv_algo=ALGO_TO_NAME_DICT.get("Surv")
        )
        self._convert_avs_last_wp_list_to_avs_info_dict(temp_mp_input)
        mp_input = MissionPlanInput(
            num_avs=len(self.avs_info_dict),
            boundary=temp_mp_input.boundary,
            avs_info_dict=self.avs_info_dict,
            grid_coverage=temp_mp_input.grid_coverage,
            circular_area=temp_mp_input.circular_area,
            polygon_area=temp_mp_input.polygon_area,
            mission_init_info=self.mission_init_info,
        )
        self.manager.update_input(deepcopy(mp_input))

    def generate_surv_path(self, temp_mp_input: MissionPlanInput):
        self._parse_mission_file_path()
        self._preprocess(temp_mp_input)

        planner = Planner(self.args)
        planner.plan(manager=self.manager, show_output=SHOW_OUTPUT)
