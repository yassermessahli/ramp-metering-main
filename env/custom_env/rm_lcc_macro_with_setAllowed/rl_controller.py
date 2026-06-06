import random

import numpy as np
import traci
from colorama import Fore

from ..sumo_env import SumoEnv


class RLController(SumoEnv):
    """Variant: joint ramp metering + network-level lane control via setAllowed.

    Action space is 16 = 8 green-time choices x 2 lane states.
        green_idx = action % 8   -> green time in {5,10,...,40} s
        lane_idx  = action // 8  -> 0 = lane open, 1 = lane closed

    Lane closure is implemented via traci.lane.setAllowed(vsl_zone_0, ["custom1"]).
    On-ramp vehicles are assigned vClass="custom1" so they are unaffected.
    Mainline vehicles (vClass="passenger") detect the restriction from far upstream
    and re-route to vsl_zone_1/2 autonomously via SUMO's LC2013 model.
    One API call per RL cycle replaces the per-vehicle changeLane loop.
    """

    CONTROLLED_LANE_ID = "vsl_zone_0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CYCLE_DURATION_SEC = 40.0
        self.ty = 3

        self.green_time_actions_sec = np.array(
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )
        self.lane_state_actions = np.array([0, 1])  # 0=open, 1=closed
        self.action_space_n = len(self.green_time_actions_sec) * len(
            self.lane_state_actions
        )  # 16

        self.green_phase_index = 0
        self.red_phase_index = 1

        self.upstream_mainline_all_detector_ids = self.get_edge_induction_loops(
            self.UPSTREAM_EDGE
        )
        self.bottleneck_edge_all_detector_ids = self.get_edge_induction_loops(
            self.MERGING_EDGE
        )
        self.downstream_mainline_all_detector_ids = self.get_edge_induction_loops(
            self.DOWNSTREAM_EDGE
        )

        self.upstream_detector_ids_state = [
            "up_stream_sens_0",
            "up_stream_sens_1",
            "up_stream_sens_2",
        ]
        self.bottleneck_detector_ids_state = [
            "bottle_neck_sens_0",
            "bottle_neck_sens_1",
            "bottle_neck_sens_2",
            "bottle_neck_sens_3",
        ]
        self.outflow_detector_ids_reward = self.downstream_mainline_all_detector_ids
        self.ramp_queue_detector_id = "queue_sens"

        # State: 15 features (14 macro + last lane action)
        self.observation_space_n = 15

        self.last_action_value_sec = self.green_time_actions_sec[0]
        self.last_lane_action = 0  # 0=open, 1=closed

        self._reset_cycle_aggregators()

        self.processed_flow_upstream_vph = 0.0
        self.processed_flow_merging_vph = 0.0
        self.processed_mainline_flow_downstream_vph = 0.0

        self.processed_occ_upstream_percent = 0.0
        self.processed_occ_bottleneck_percent = 0.0
        self.processed_occ_downstream_percent = 0.0

        self.processed_speed_bottleneck_mps = 0.0
        self.processed_speed_upstream_mps = 0.0
        self.processed_mainline_speed_downstream_mps = 0.0

        self.processed_ramp_queue_veh = 0.0
        self.sum_queue = 0.0

        self._last_detailed_info = {}
        self._initialize_last_detailed_info_placeholders()

    def _initialize_last_detailed_info_placeholders(self):
        self._last_detailed_info = {
            "mainline_flow_upstream_v/h": 0.0,
            "mainline_occ_upstream_percent": 0.0,
            "mainline_speed_upstream_km/h": 0.0,
            "mainline_flow_mergeArea_v/h": 0.0,
            "mainline_occ_mergeArea_percent": 0.0,
            "mainline_speed_mergeArea_km/h": 0.0,
            "mainline_flow_downstream_v/h": 0.0,
            "mainline_occ_downstream_percent": 0.0,
            "mainline_speed_downstream_km/h": 0.0,
            "ramp_queue_veh": 0.0,
            "current_tl_phase_index": -1,
            "current_tl_ryg_state": "N/A",
            "chosen_green_time_sec": 0.0,
            "lane_closed": 0,
            "reward_outflow_speed_comp": 0.0,
            "reward_throughput_comp": 0.0,
            "penalty_ramp_queue_comp": 0.0,
            "penalty_bottleneck_occ_comp": 0.0,
            "penalty_spillback_comp": 0.0,
            "sim_time": 0.0,
            "episode": 0,
            "total_running_vehicles": 0,
            "total_departed": 0,
            "total_arrived": 0,
            "l": 0,
            "r": 0.0,
            "TimeLimit.truncated": False,
            "done": False,
        }

    def _reset_cycle_aggregators(self):
        self.sum_interval_upstream_veh_count = 0
        self.sum_interval_merging_veh_count = 0
        self.list_interval_upstream_occ = []
        self.list_interval_upstream_speed = []
        self.list_interval_bottleneck_occ = []
        self.list_interval_bottleneck_speed = []
        self.list_interval_ramp_queue = []
        self.sum_interval_outflow_veh_count = 0
        self.list_interval_outflow_speed = []
        self.sum_queue = 0
        self.current_ramp_queue_veh = 0

    def _apply_lane_control(self, lane_idx):
        """Open or close vsl_zone_0 to mainline traffic via setAllowed.

        lane_idx=1: allow only custom1 (on-ramp vClass) → effective no-entry for mainline.
        lane_idx=0: allow all classes → lane open.
        Called once per RL cycle and on reset.
        """
        if lane_idx == 1:
            traci.lane.setAllowed(self.CONTROLLED_LANE_ID, ["custom1"])
        else:
            traci.lane.setAllowed(self.CONTROLLED_LANE_ID, [])
        self._update_lane_indicator(lane_idx)

    def _update_lane_indicator(self, lane_idx):
        """Create or update a POI flag at the start of vsl_zone_0 showing lane state.

        Green = lane open, Red = lane closed.
        Only active when the SUMO GUI is running.
        """
        if not self.gui:
            return
        poi_id = "lcc_indicator_" + self.CONTROLLED_LANE_ID
        if poi_id not in traci.poi.getIDList():
            shape = traci.lane.getShape(self.CONTROLLED_LANE_ID)
            x, y = shape[0]
            traci.poi.add(
                poi_id,
                x,
                y,
                (0, 255, 0, 255),
                poiType="lcc_flag",
                layer=100,
                width=3,
                height=3,
            )
        color = (255, 0, 0, 255) if lane_idx == 1 else (0, 255, 0, 255)
        traci.poi.setColor(poi_id, color)

    def _generate_route_file(self):
        """Override to assign vClass='custom1' to on-ramp vehicle types.

        Mainline vehicles keep vClass='passenger' so setAllowed("custom1") blocks them.
        On-ramp vehicles (def_ramp / con_ramp) use custom1 and pass through unaffected.
        """
        main_flow = random.choices(
            self.args["veh_per_hour_main"],
            weights=self.args["veh_per_hour_main_weights"],
        )[0]
        on_ramp_flow = random.choices(
            self.args["veh_per_hour_on_ramp"],
            weights=self.args["veh_per_hour_on_ramp_weights"],
        )[0]
        off_ramp_flow = random.choices(
            self.args["veh_per_hour_off_ramp"],
            weights=self.args["veh_per_hour_off_ramp_weights"],
        )[0]

        min_pen, max_pen = self.args["con_penetration_rate_range"]
        pen_rate = random.uniform(min_pen, max_pen)

        self.main_flow_vph = main_flow
        self.on_ramp_flow_vph = on_ramp_flow
        self.off_ramp_flow_vph = off_ramp_flow
        self.pen_rate = pen_rate

        main_con = int(main_flow - 1)
        main_def = 1
        on_ramp_con = int(on_ramp_flow - 1)
        on_ramp_def = 1
        off_ramp_con = int(off_ramp_flow - 1)
        off_ramp_def = 1

        xml_content = f"""<!-- Generated on-the-fly for episode {self.ep_count + 1} -->
    <routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">

        <vType id=\"def\"      vClass=\"passenger\" length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.9\" />
        <vType id=\"con\"      vClass=\"passenger\" length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.8\" color=\"1,0,0\" />
        <vType id=\"def_ramp\" vClass=\"custom1\"   length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.9\" />
        <vType id=\"con_ramp\" vClass=\"custom1\"   length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.8\" color=\"0,0,1\" />

        <route id=\"entry_to_end_main_road\" edges=\"entry off_ramp_up_stream main_road vsl_zone acceleration_area end_main_road\" />
        <route id=\"entry_to_off_ramp\" edges=\"entry off_ramp_up_stream off_ramp_beginning off_ramp\" />
        <route id=\"on_ramp_to_end_main_road\" edges=\"on_ramp passage_area acceleration_area end_main_road\" />

        <flow id=\"main_con\"     type=\"con\"      vehsPerHour=\"{main_con}\"     route=\"entry_to_end_main_road\"   begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"main_def\"     type=\"def\"      vehsPerHour=\"{main_def}\"     route=\"entry_to_end_main_road\"   begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"on_ramp_con\"  type=\"con_ramp\" vehsPerHour=\"{on_ramp_con}\"  route=\"on_ramp_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"on_ramp_def\"  type=\"def_ramp\" vehsPerHour=\"{on_ramp_def}\"  route=\"on_ramp_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"off_ramp_con\" type=\"con\"      vehsPerHour=\"{off_ramp_con}\" route=\"entry_to_off_ramp\"        begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"off_ramp_def\" type=\"def\"      vehsPerHour=\"{off_ramp_def}\" route=\"entry_to_off_ramp\"        begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />

    </routes>
    """

        route_file_path = self.data_dir + self.config + ".rou.xml"
        with open(route_file_path, "w") as f:
            f.write(xml_content)

        print(
            Fore.LIGHTMAGENTA_EX,
            f"Generated new route file for Ep {self.ep_count + 1}: Main={main_flow}, Ramp={on_ramp_flow}, PenRate={pen_rate:.2f}",
            Fore.RESET,
        )

    def _collect_data_at_cycle_end(self):
        self.processed_flow_upstream_vph = self.get_loops_flow_interval(
            self.upstream_detector_ids_state, self.CYCLE_DURATION_SEC
        )
        self.processed_flow_merging_vph = self.get_loops_flow_interval(
            self.bottleneck_detector_ids_state, self.CYCLE_DURATION_SEC
        )
        self.processed_mainline_flow_downstream_vph = self.get_loops_flow_interval(
            self.outflow_detector_ids_reward, self.CYCLE_DURATION_SEC
        )

        self.processed_occ_upstream_percent = self.get_loops_occupancy_interval(
            self.upstream_detector_ids_state
        )
        self.processed_occ_bottleneck_percent = self.get_loops_occupancy_interval(
            self.bottleneck_detector_ids_state
        )
        self.processed_occ_downstream_percent = self.get_loops_occupancy_interval(
            self.outflow_detector_ids_reward
        )

        self.processed_speed_upstream_mps = self.get_loops_flow_weigthed_mean_speed(
            self.upstream_detector_ids_state
        )
        self.processed_speed_bottleneck_mps = self.get_loops_flow_weigthed_mean_speed(
            self.bottleneck_detector_ids_state
        )
        self.processed_mainline_speed_downstream_mps = (
            self.get_loops_flow_weigthed_mean_speed(self.outflow_detector_ids_reward)
        )

        self.processed_ramp_queue_veh = (
            self.sum_queue / self.CYCLE_DURATION_SEC
            if self.CYCLE_DURATION_SEC > 0
            else 0.0
        )

        self.processed_flow_lane_0_merging_vph = self.get_loops_flow_interval(
            [self.bottleneck_detector_ids_state[0]], self.CYCLE_DURATION_SEC
        )
        self.processed_occ_lane_0_bottleneck_percent = (
            self.get_loops_occupancy_interval([self.bottleneck_detector_ids_state[0]])
        )
        self.processed_speed_lane_0_bottleneck_mps = (
            self.get_loops_flow_weigthed_mean_speed(
                [self.bottleneck_detector_ids_state[0]]
            )
        )

        self.processed_flow_lane_0_upstream_vph = self.get_loops_flow_interval(
            [self.upstream_detector_ids_state[1]], self.CYCLE_DURATION_SEC
        )
        self.processed_occ_lane_0_upstream_percent = self.get_loops_occupancy_interval(
            [self.upstream_detector_ids_state[1]]
        )
        self.processed_speed_lane_0_upstream_mps = (
            self.get_loops_flow_weigthed_mean_speed(
                [self.upstream_detector_ids_state[1]]
            )
        )

    def reset(self):
        self.simulation_reset()
        self._reset_cycle_aggregators()
        self.last_action_value_sec = self.green_time_actions_sec[0]
        self.last_lane_action = 0
        self._initialize_last_detailed_info_placeholders()
        self._last_detailed_info.update(super().log_info())

        if self.ramp_meter_id and self.red_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.CYCLE_DURATION_SEC)

        if self.sim_step_length > 0:
            num_init_steps = int(round(max(1.0, 5.0 / self.sim_step_length)))
        else:
            num_init_steps = 5

        for _ in range(num_init_steps):
            if self.is_simulation_end():
                break
            self.simulation_step()

        self._collect_data_at_cycle_end()
        self._apply_lane_control(0)

        current_phase_index_init = -1
        current_ryg_state_init = "N/A"
        if self.ramp_meter_id:
            try:
                current_phase_index_init = self.get_phase(self.ramp_meter_id)
                current_ryg_state_init = self.get_ryg_state(self.ramp_meter_id)
            except Exception:
                pass

        self._last_detailed_info.update(
            {
                "mainline_flow_upstream_v/h": self.processed_flow_upstream_vph,
                "mainline_occ_upstream_percent": self.processed_occ_upstream_percent,
                "mainline_speed_upstream_km/h": self.processed_speed_upstream_mps,
                "mainline_flow_mergeArea_v/h": self.processed_flow_merging_vph,
                "mainline_occ_mergeArea_percent": self.processed_occ_bottleneck_percent,
                "mainline_speed_mergeArea_km/h": self.processed_speed_bottleneck_mps,
                "mainline_flow_downstream_v/h": self.processed_mainline_flow_downstream_vph,
                "mainline_speed_downstream_km/h": self.processed_mainline_speed_downstream_mps,
                "mainline_occ_downstream_percent": self.processed_occ_downstream_percent,
                "ramp_queue_veh": self.processed_ramp_queue_veh,
                "current_tl_phase_index": current_phase_index_init,
                "current_tl_ryg_state": current_ryg_state_init,
                "chosen_green_time_sec": self.last_action_value_sec,
                "lane_closed": self.last_lane_action,
            }
        )
        self._last_detailed_info.update(super().log_info())

        return self._get_current_observation()

    def step(self, action_index):
        if not (0 <= action_index < self.action_space_n):
            action_index = np.clip(action_index, 0, self.action_space_n - 1).item()

        green_idx = int(action_index) % len(self.green_time_actions_sec)
        lane_idx = int(action_index) // len(self.green_time_actions_sec)
        lane_idx = 
        chosen_green_time_sec = self.green_time_actions_sec[green_idx]
        red_time_sec = self.CYCLE_DURATION_SEC - chosen_green_time_sec
        self.last_action_value_sec = chosen_green_time_sec
        self.last_lane_action = lane_idx

        # Apply lane control once per cycle before the green phase starts
        self._apply_lane_control(lane_idx)

        self._reset_cycle_aggregators()

        # Apply green phase first
        if (
            self.ramp_meter_id
            and self.green_phase_index != -1
            and chosen_green_time_sec > 0
        ):
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, chosen_green_time_sec)
            if self.sim_step_length > 0:
                num_steps_green = int(
                    round(chosen_green_time_sec / self.sim_step_length)
                )
            else:
                num_steps_green = 0

            for _ in range(num_steps_green):
                if self.is_simulation_end():
                    break
                self.simulation_step()
                self.sum_queue += self.get_edge_ls_queue_length_vehicles(
                    self.ON_RAMP_EDGE
                )

        # Apply red phase next
        if self.ramp_meter_id and self.red_phase_index != -1 and red_time_sec > 0:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, red_time_sec)
            if self.sim_step_length > 0:
                num_steps_red = int(round(red_time_sec / self.sim_step_length))
            else:
                num_steps_red = 0

            for _ in range(num_steps_red):
                if self.is_simulation_end():
                    break
                self.simulation_step()
                self.sum_queue += self.get_edge_ls_queue_length_vehicles(
                    self.ON_RAMP_EDGE
                )

        self._collect_data_at_cycle_end()

        new_observation = self._get_current_observation()
        reward = self._calculate_reward()
        is_done = (
            self.is_simulation_end() or self.get_current_time() >= self.args["steps"]
        )

        current_phase_index = -1
        current_ryg_state = "N/A"
        if self.ramp_meter_id:
            try:
                current_phase_index = self.get_phase(self.ramp_meter_id)
                current_ryg_state = self.get_ryg_state(self.ramp_meter_id)
            except Exception:
                pass

        info_for_this_step = {
            "mainline_flow_upstream_v/h": self.processed_flow_upstream_vph,
            "mainline_occ_upstream_percent": self.processed_occ_upstream_percent,
            "mainline_speed_upstream_km/h": self.processed_speed_upstream_mps,
            "mainline_flow_mergeArea_v/h": self.processed_flow_merging_vph,
            "mainline_occ_mergeArea_percent": self.processed_occ_bottleneck_percent,
            "mainline_speed_mergeArea_km/h": self.processed_speed_bottleneck_mps,
            "mainline_flow_downstream_v/h": self.processed_mainline_flow_downstream_vph,
            "mainline_speed_downstream_km/h": self.processed_mainline_speed_downstream_mps,
            "mainline_occ_downstream_percent": self.processed_occ_downstream_percent,
            "ramp_queue_veh": self.processed_ramp_queue_veh,
            "current_tl_phase_index": current_phase_index,
            "current_tl_ryg_state": current_ryg_state,
            "chosen_green_time_sec": chosen_green_time_sec,
            "lane_closed": int(lane_idx),
            "reward_outflow_speed_comp": self._reward_outflow_speed(),
            "reward_throughput_comp": self._reward_throughput(),
            "penalty_ramp_queue_comp": self._penalty_ramp_queue(),
            "penalty_bottleneck_occ_comp": self._penalty_bottleneck_occ(),
            "penalty_spillback_comp": self._penalty_spillback(),
        }

        info_for_this_step.update(super().log_info())
        self._last_detailed_info = info_for_this_step.copy()

        return new_observation, reward, is_done, info_for_this_step

    def _get_current_observation(self):
        norm_flow_upstream = np.clip(
            self.processed_flow_upstream_vph / self.MAX_FLOW_UPSTREAM_VPH, 0, 1
        )
        norm_flow_merging = np.clip(
            self.processed_flow_merging_vph / self.MAX_FLOW_MERGING_VPH, 0, 1
        )
        norm_occ_upstream = np.clip(
            self.processed_occ_upstream_percent / self.MAX_OCCUPANCY_PERCENT, 0, 1
        )
        norm_speed_upstream = np.clip(
            self.processed_speed_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_occ_bottleneck = np.clip(
            self.processed_occ_bottleneck_percent / self.MAX_OCCUPANCY_PERCENT, 0, 1
        )
        norm_speed_bottleneck = np.clip(
            self.processed_speed_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_ramp_queue = np.clip(
            self.processed_ramp_queue_veh
            / (self.MAX_RAMP_QUEUE_VEH if self.MAX_RAMP_QUEUE_VEH > 0 else 1.0),
            0,
            1,
        )
        norm_flow_lane_0_bottleneck = np.clip(
            self.processed_flow_lane_0_merging_vph
            / (self.MAX_LANE_FLOW_VPH if self.MAX_LANE_FLOW_VPH > 0 else 1.0),
            0,
            1,
        )
        norm_flow_lane_0_upstream = np.clip(
            self.processed_flow_lane_0_upstream_vph
            / (self.MAX_LANE_FLOW_VPH if self.MAX_LANE_FLOW_VPH > 0 else 1.0),
            0,
            1,
        )
        norm_occ_lane_0_bottleneck = np.clip(
            self.processed_occ_lane_0_bottleneck_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 0.0),
            0,
            1,
        )
        norm_speed_lane_0_bottleneck = np.clip(
            self.processed_speed_lane_0_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_occ_lane_0_upstream = np.clip(
            self.processed_occ_lane_0_upstream_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 0.0),
            0,
            1,
        )
        norm_speed_lane_0_upstream = np.clip(
            self.processed_speed_lane_0_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_last_action = np.clip(
            self.last_action_value_sec
            / (self.CYCLE_DURATION_SEC if self.CYCLE_DURATION_SEC > 0 else 1.0),
            0,
            1,
        )

        return np.array(
            [
                norm_flow_upstream,
                norm_flow_merging,
                norm_occ_upstream,
                norm_speed_upstream,
                norm_occ_bottleneck,
                norm_speed_bottleneck,
                norm_ramp_queue,
                norm_flow_lane_0_bottleneck,
                norm_flow_lane_0_upstream,
                norm_occ_lane_0_bottleneck,
                norm_speed_lane_0_bottleneck,
                norm_occ_lane_0_upstream,
                norm_speed_lane_0_upstream,
                norm_last_action,
                float(self.last_lane_action),
            ],
            dtype=np.float32,
        )

    def _reward_outflow_speed(self):
        return np.clip(
            self.processed_mainline_speed_downstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _reward_upstream_speed(self):
        return np.clip(
            self.processed_speed_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _reward_merging_speed(self):
        return np.clip(
            self.processed_speed_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _penalty_bottleneck_occ(self):
        norm_occ = np.clip(
            self.processed_occ_bottleneck_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_occ

    def _penalty_upstream_occ(self):
        norm_occ = np.clip(
            self.processed_occ_upstream_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_occ

    def _reward_throughput(self):
        if self.get_edge_lane_n(self.DOWNSTREAM_EDGE) > 0:
            max_possible_throughput = self.MAX_LANE_FLOW_VPH * self.get_edge_lane_n(
                self.DOWNSTREAM_EDGE
            )
        else:
            max_possible_throughput = self.MAX_LANE_FLOW_VPH
        return np.clip(
            self.processed_mainline_flow_downstream_vph
            / (max_possible_throughput if max_possible_throughput > 0 else 1.0),
            0,
            1,
        )

    def _penalty_ramp_queue(self):
        norm_queue = np.clip(
            self.processed_ramp_queue_veh
            / (self.MAX_RAMP_QUEUE_VEH if self.MAX_RAMP_QUEUE_VEH > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_queue

    def _penalty_spillback(self):
        spillback_threshold_veh = 0.9 * self.MAX_RAMP_QUEUE_VEH
        if self.processed_ramp_queue_veh > spillback_threshold_veh:
            denominator = self.MAX_RAMP_QUEUE_VEH - spillback_threshold_veh
            if denominator < 1e-6:
                denominator = 1e-6
            spill_amount = (
                self.processed_ramp_queue_veh - spillback_threshold_veh
            ) / denominator
            return -1.0 * np.clip(spill_amount, 0, 1)
        return 0.0

    def _calculate_reward(self):
        w_speed_merge = 1.5
        w_speed_up = 1.0
        w_speed_down = 0.5

        w_occ_bottle = 2.0
        w_occ_upstream = 1.0
        w_queue = 1.0
        w_spillback = 20.0

        r_speed_merge = self._reward_merging_speed()
        r_speed_up = self._reward_upstream_speed()
        r_speed_down = self._reward_outflow_speed()

        p_occ_bottle = self._penalty_bottleneck_occ()
        p_occ_upstream = self._penalty_upstream_occ()
        p_queue = self._penalty_ramp_queue()
        p_spillback = self._penalty_spillback()

        reward = (
            (w_speed_merge * r_speed_merge)
            + (w_speed_up * r_speed_up)
            + (w_speed_down * r_speed_down)
            + (w_occ_bottle * p_occ_bottle)
            + (w_occ_upstream * p_occ_upstream)
            + (w_queue * p_queue)
            + (w_spillback * p_spillback)
        )
        return float(reward)

    def obs(self):
        return self._get_current_observation()

    def rew(self):
        return self._calculate_reward()

    def done(self):
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

    def info(self):
        return self._last_detailed_info
