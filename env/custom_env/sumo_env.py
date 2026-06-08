# sumo_env.py - Core SUMO environment with core TraCI calls

import json
import os
import random
import subprocess
import sys
import threading

import numpy as np
import traci  # noqa
from colorama import Fore
from sumolib import net  # noqa
from traci import constants as tc

from .utils import SUMO_PARAMS


class SumoEnv:
    SUMO_ENV = "env/custom_env/"

    @staticmethod
    def pretty_print(d):
        print(json.dumps(d, sort_keys=True, indent=4))

    @staticmethod
    def arg_max(_list):
        return max(range(len(_list)), key=lambda i: _list[i])

    @staticmethod
    def arg_min(_list):
        return min(range(len(_list)), key=lambda i: _list[i])

    @staticmethod
    def clip(min_clip, max_clip, x):
        return max(min_clip, min([max_clip, x])) if min_clip < max_clip else x

    def __init__(self, gui=False, log=False, rnd=(False, False)):
        self.args = SUMO_PARAMS
        self.config = self.args["config"]
        self.data_dir = self.SUMO_ENV + "data/" + self.config + "/"
        self.net_file_name = self.config + ".net.xml"

        try:
            self.net = net.readNet(self.data_dir + self.net_file_name)
        except Exception as e:
            print(f"Error reading net file: {self.data_dir + self.net_file_name}")
            print(e)
            sys.exit(1)

        self._initialize_grid_params_from_net()

        self.tl_ids = [tl.getID() for tl in self.net.getTrafficLights()]
        if not self.tl_ids:
            print("Warning: No traffic lights (ramp meters) found in the network.")
            self.ramp_meter_id = None
        else:
            self.ramp_meter_id = self.tl_ids[0]

        self.UPSTREAM_EDGE = "main_road"
        self.MERGING_EDGE = "acceleration_area"
        self.DOWNSTREAM_EDGE = "end_main_road"
        self.ON_RAMP_EDGE = "on_ramp"

        self.FREEFLOW_SPEED_MPS = self.args.get("v_max_speed", 27.77)
        self.MAX_RAMP_QUEUE_VEH = self.args.get("max_ramp_queue_veh", 25)
        self.MAX_LANE_FLOW_VPH = self.args.get("max_lane_flow_vph", 1900)
        self.MAX_FLOW_UPSTREAM_VPH = self.args.get("max_flow_upstream_vph", 5490)
        self.MAX_FLOW_MERGING_VPH = self.args.get("max_flow_merging_vph", 5490)
        self.MAX_FLOW_DOWNSTREAM_VPH = self.args.get("max_flow_downstream_vph", 5760)
        self.MAX_OCCUPANCY_PERCENT = 100.0

        self.gui = gui
        self.log = log
        self.rnd_params = rnd
        self.seed = self.args.get("seed", False)
        self.ep_count = 0

        self.main_flow_vph = 0
        self.on_ramp_flow_vph = 0
        self.off_ramp_flow_vph = 0
        self.pen_rate = 0.0

        self.generate_rou = self.args.get("generate_route_file", False)
        if self.generate_rou:
            self._generate_route_file()

        self.params = self.set_params()

        try:
            traci.start(self.params)
            self.sim_step_length = traci.simulation.getDeltaT()
        except traci.TraCIException as e:
            print(f"Error starting TraCI: {e}")
            print(
                "Ensure SUMO_HOME is set correctly and SUMO binaries are in the PATH or SUMO_HOME/bin."
            )
            print(f"SUMO command: {' '.join(self.params)}")
            sys.exit(1)

    def set_params(self):
        sumocfg_path = self.data_dir + self.config + ".sumocfg"
        emission_xml_path = os.path.join(
            os.path.dirname(self.data_dir), "emissions.xml"
        )
        sumo_seed = os.environ.get("SUMO_EVAL_SEED")
        sumo_log_file = os.environ.get("SUMO_EVAL_LOG_FILE")
        sumo_tripinfo_file = os.environ.get("SUMO_TRIPINFO_FILE")

        tripinfo_output = self.data_dir + "tripinfo.xml"
        if sumo_tripinfo_file:
            tripinfo_output = sumo_tripinfo_file
            tripinfo_dir = os.path.dirname(tripinfo_output)
            if tripinfo_dir:
                os.makedirs(tripinfo_dir, exist_ok=True)

        params = [
            "sumo-gui" if self.gui else "sumo",
            "-c",
            sumocfg_path,
            "--tripinfo-output",
            tripinfo_output,
            "--device.emissions.probability",
            "1.0",
            "--time-to-teleport",
            str(self.args.get("time_to_teleport", 300)),
            "--waiting-time-memory",
            str(self.args.get("waiting_time_memory", 1000)),
            "--no-warnings",
            "true",
        ]

        if sumo_seed:
            params += ["--seed", str(sumo_seed)]
        elif self.seed:
            params += ["--seed", str(self.args.get("seed_value", 42))]

        # Per-run step-length override (e.g. smooth GUI inspection). Command line wins
        # over the sumocfg; no-op when unset, so training is unaffected.
        sumo_step_length = os.environ.get("SUMO_STEP_LENGTH")
        if sumo_step_length:
            params += ["--step-length", str(sumo_step_length)]

        if sumo_log_file:
            params += ["--log", sumo_log_file]

        if self.gui:
            params += [
                "--delay",
                str(self.args.get("delay", 0)),
                "--start",
                "true",
                "--quit-on-end",
                "true",
            ]
            gui_settings_file = (
                self.SUMO_ENV + "data/" + self.config + "/gui-settings.cfg"
            )
            if os.path.exists(gui_settings_file):
                params.append("--gui-settings-file")
                params.append(gui_settings_file)
            else:
                print(
                    f"Note: GUI settings file not found at {gui_settings_file}, using SUMO defaults."
                )

        return params

    def _initialize_grid_params_from_net(self):
        self.grid_cols = 5
        self.grid_channels = 2
        self.cell_length_m = self.args.get("cell_length", 8.0)
        self.accel_segment_len = 84.0
        self.passage_segment_len = self.net.getLane("passage_area_0").getLength()

        self.grid_total_length = 216.0

        self.pre_merge_segment_len = self.grid_total_length - self.accel_segment_len
        self.on_ramp_segment_len = self.pre_merge_segment_len - self.passage_segment_len
        self.main_road_segment_len = self.pre_merge_segment_len
        self.grid_rows = int(self.grid_total_length / self.cell_length_m)

        self.internal_to_destination_map = {}
        try:
            for node in self.net.getNodes():
                for conn in node.getConnections():
                    internal_lane_id = conn._via
                    to_lane_obj = conn._toLane
                    if internal_lane_id and to_lane_obj:
                        self.internal_to_destination_map[internal_lane_id] = (
                            to_lane_obj.getID()
                        )
        except AttributeError:
            print("--- FATAL ERROR: Could not read connections from network file. ---")
            sys.exit(1)

        print("--- Internal Lane Map ---")
        for k, v in self.internal_to_destination_map.items():
            print(f"  '{k}' -> '{v}'")
        print("-------------------------\n")

    def _create_grid_observation(self):
        grid = np.zeros(
            (self.grid_rows, self.grid_cols, self.grid_channels), dtype=np.float32
        )
        try:
            all_veh_data = traci.vehicle.getSubscriptionResults(None)
        except traci.TraCIException:
            return grid

        v_type_con = self.args.get("v_type_con", "con")
        freeflow_speed = (
            self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 35.0
        )

        column_map = {
            "main_road_2": 0,
            "acceleration_area_3": 0,
            "main_road_1": 1,
            "acceleration_area_2": 1,
            "main_road_0": 2,
            "acceleration_area_1": 2,
            "acceleration_area_0": 3,
            "on_ramp_0": 4,
            "passage_area_0": 4,
        }

        for _veh_id, data in all_veh_data.items():
            if data.get(tc.VAR_TYPE) != v_type_con:
                continue

            original_lane_id = data.get(tc.VAR_LANE_ID)
            lane_pos = data.get(tc.VAR_LANEPOSITION)

            lane_id = self.internal_to_destination_map.get(
                original_lane_id, original_lane_id
            )
            if original_lane_id.startswith(":"):
                lane_pos = 0.0

            col_idx = column_map.get(lane_id)
            if col_idx is None:
                continue

            lane_len = self.net.getLane(lane_id).getLength()
            dist_from_grid_start = -1

            if "on_ramp" in lane_id:
                start_of_segment = lane_len - self.on_ramp_segment_len
                if lane_pos >= start_of_segment:
                    dist_from_grid_start = lane_pos - start_of_segment
            elif "passage_area" in lane_id:
                dist_from_grid_start = self.on_ramp_segment_len + lane_pos
            elif "main_road" in lane_id:
                start_of_segment = lane_len - self.main_road_segment_len
                if lane_pos >= start_of_segment:
                    dist_from_grid_start = lane_pos - start_of_segment
            elif "acceleration_area" in lane_id:
                if lane_pos < self.accel_segment_len:
                    if lane_id == "acceleration_area_0":
                        preceding_path_len = (
                            self.on_ramp_segment_len + self.passage_segment_len
                        )
                    else:
                        preceding_path_len = self.main_road_segment_len
                    dist_from_grid_start = preceding_path_len + lane_pos

            if dist_from_grid_start < 0:
                continue

            dist_from_grid_end = self.grid_total_length - dist_from_grid_start
            row_idx = int(dist_from_grid_end / self.cell_length_m)
            row_idx = min(row_idx, self.grid_rows - 1)

            if 0 <= row_idx < self.grid_rows:
                speed = data.get(tc.VAR_SPEED, 0)
                norm_speed = self.clip(0.0, 1.0, speed / freeflow_speed)
                if grid[row_idx, col_idx, 1] == 0:
                    grid[row_idx, col_idx, 0] = norm_speed
                    grid[row_idx, col_idx, 1] = 1.0
        return grid

    def _subscribe_to_vehicles(self):
        for veh_id in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(
                veh_id, [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_TYPE]
            )

    def start(self):
        traci.start(self.params)
        self.sim_step_length = traci.simulation.getDeltaT()

    def stop(self, timeout=5.0):
        """Closes the TraCI connection with a timeout to prevent hanging.

        Args:
            timeout: Maximum time in seconds to wait for traci.close() to complete
        """

        def close_traci():
            traci.close()

        # Try to close TraCI with a timeout
        close_thread = threading.Thread(target=close_traci, daemon=True)
        close_thread.start()
        close_thread.join(timeout=timeout)

        # If thread is still alive after timeout, force kill SUMO processes
        if close_thread.is_alive():
            subprocess.run(["pkill", "-9", "-f", "sumo"], timeout=1)

        sys.stdout.flush()

    def close(self):
        self.stop()

    def simulation_reset(self):
        self.stop()
        self.ep_count += 1

        if self.generate_rou:
            self._generate_route_file()

        self.start()

    def simulation_step(self):
        traci.simulationStep()
        self._subscribe_to_vehicles()

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def obs(self):
        raise NotImplementedError

    def rew(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def info(self):
        return {} if not self.log else self.log_info()

    def is_simulation_end(self):
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except traci.TraCIException:
            return True

    def get_current_time(self):
        try:
            return traci.simulation.getTime()
        except traci.TraCIException:
            return -1

    def get_phase(self, tl_id):
        return traci.trafficlight.getPhase(tl_id)

    def get_ryg_state(self, tl_id):
        return traci.trafficlight.getRedYellowGreenState(tl_id)

    def set_phase(self, tl_id, phase_index):
        traci.trafficlight.setPhase(tl_id, phase_index)

    def set_phase_duration(self, tl_id, duration_sec):
        traci.trafficlight.setPhaseDuration(tl_id, duration_sec)

    def get_lanes_of_edge(self, edge_id):
        edge_lanes = []
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for i in range(num_lanes):
                edge_lanes.append(f"{edge_id}_{i}")
        except traci.TraCIException:
            print(f"Warning: SumoEnv - Could not get lanes for edge {edge_id}")
        return edge_lanes

    def get_edge_lane_n(self, edge_id):
        return traci.edge.getLaneNumber(edge_id)

    def get_edge_induction_loops(self, edge_id):
        lanes = self.get_lanes_of_edge(edge_id)
        if not lanes:
            return []
        try:
            all_loops = traci.inductionloop.getIDList()
        except traci.TraCIException:
            print("Warning: SumoEnv - Could not get induction loop ID list.")
            return []
        return [
            loop_id
            for loop_id in all_loops
            if traci.inductionloop.getLaneID(loop_id) in lanes
        ]

    def get_loops_flow_interval(self, loop_ids, interval_duration_sec):
        if not loop_ids or interval_duration_sec <= 0:
            return 0.0
        total_vehicles = 0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                total_vehicles += traci.inductionloop.getLastIntervalVehicleNumber(
                    loop_id
                )
                valid_loops += 1
            except traci.TraCIException:
                print(
                    f"Warning: SumoEnv - Could not get interval vehicle number for loop {loop_id}"
                )
        return (
            (total_vehicles * 3600.0) / interval_duration_sec
            if valid_loops > 0
            else 0.0
        )

    def get_edge_flow_from_loops_interval(self, edge_id, interval_duration_sec):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_flow_interval(loops, interval_duration_sec)

    def get_loops_occupancy_interval(self, loop_ids):
        if not loop_ids:
            return 0.0
        total_occupancy = 0.0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                total_occupancy += traci.inductionloop.getLastIntervalOccupancy(loop_id)
                valid_loops += 1
            except traci.TraCIException:
                print(
                    f"Warning: SumoEnv - Could not get interval occupancy for loop {loop_id}"
                )
        return total_occupancy / valid_loops if valid_loops > 0 else 0.0

    def get_edge_occupancy_from_loops_interval(self, edge_id):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_occupancy_interval(loops)

    def get_loops_mean_speed_interval(self, loop_ids):
        if not loop_ids:
            return 0.0
        total_speed = 0.0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                speed = traci.inductionloop.getLastIntervalMeanSpeed(loop_id)
                if speed >= 0:
                    total_speed += speed
                    valid_loops += 1
            except traci.TraCIException:
                print(
                    f"Warning: SumoEnv - Could not get interval mean speed for loop {loop_id}"
                )
        return total_speed / valid_loops if valid_loops > 0 else 0.0

    def get_edge_mean_speed_from_loops_interval(self, edge_id):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_mean_speed_interval(loops)

    def get_edge_ls_mean_speed(self, edge_id):
        return traci.edge.getLastStepMeanSpeed(edge_id)

    def get_loops_flow_weigthed_mean_speed(self, loop_ids):
        if not loop_ids:
            return 0.0
        total_speed = 0.0
        total_flow = 0.0
        for loop_id in loop_ids:
            try:
                flow = traci.inductionloop.getLastStepVehicleNumber(loop_id)
                speed = traci.inductionloop.getLastStepMeanSpeed(loop_id)
                if flow > 0 and speed >= 0:
                    total_speed += speed * flow
                    total_flow += flow
            except traci.TraCIException:
                print(f"Warning: SumoEnv - Could not get data for loop {loop_id}")
        return (total_speed / total_flow) if total_flow > 0 else 0.0

    def get_edge_ls_queue_length_vehicles(self, edge_id):
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.TraCIException:
            print(f"Warning: SumoEnv - Could not get vehicle number for edge {edge_id}")
            return 0

    def get_detector_vehicle_count_last_step(self, detector_id):
        try:
            return traci.inductionloop.getLastStepVehicleNumber(detector_id)
        except traci.TraCIException:
            try:
                return traci.laneareadetector.getLastStepVehicleNumber(detector_id)
            except traci.TraCIException:
                print(
                    f"Warning: SumoEnv - Could not get vehicles for detector {detector_id}"
                )
                return 0

    def get_veh_speed(self, veh_id):
        try:
            return traci.vehicle.getSpeed(veh_id)
        except traci.TraCIException:
            return 0.0

    def get_final_simulation_stats(self):
        try:
            teleports = traci.simulation.getStartingTeleportNumber()
            return {"total_teleported_vehicles": teleports}
        except traci.TraCIException:
            return {"total_teleported_vehicles": -1}

    def get_demand_parameters(self):
        return {
            "demand_main_vph": self.main_flow_vph,
            "demand_on_ramp_vph": self.on_ramp_flow_vph,
            "demand_off_ramp_vph": self.off_ramp_flow_vph,
            "demand_pen_rate": self.pen_rate,
        }

    def get_scenario_info(self):
        return {
            "main_flow_vph": self.main_flow_vph,
            "on_ramp_flow_vph": self.on_ramp_flow_vph,
            "off_ramp_flow_vph": self.off_ramp_flow_vph,
            "con_penetration_rate": self.pen_rate,
        }

    def _generate_route_file(self):
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

        <vType id=\"def\" vClass=\"passenger\" length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.9\" />
        <vType id=\"con\" vClass=\"passenger\" length=\"5.0\" minGap=\"2.5\" accel=\"2.6\" decel=\"4.5\" maxSpeed=\"35\" sigma=\"0.8\" color=\"1,0,0\" />

        <route id=\"entry_to_end_main_road\" edges=\"entry off_ramp_up_stream main_road vsl_zone acceleration_area end_main_road\" />
        <route id=\"entry_to_off_ramp\" edges=\"entry off_ramp_up_stream off_ramp_beginning off_ramp\" />
        <route id=\"on_ramp_to_end_main_road\" edges=\"on_ramp passage_area acceleration_area end_main_road\" />

        <flow id=\"main_con\" type=\"con\" vehsPerHour=\"{main_con}\" route=\"entry_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"main_def\" type=\"def\" vehsPerHour=\"{main_def}\" route=\"entry_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"on_ramp_con\" type=\"con\" vehsPerHour=\"{on_ramp_con}\" route=\"on_ramp_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"on_ramp_def\" type=\"def\" vehsPerHour=\"{on_ramp_def}\" route=\"on_ramp_to_end_main_road\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"off_ramp_con\" type=\"con\" vehsPerHour=\"{off_ramp_con}\" route=\"entry_to_off_ramp\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />
        <flow id=\"off_ramp_def\" type=\"def\" vehsPerHour=\"{off_ramp_def}\" route=\"entry_to_off_ramp\" begin=\"0\" end=\"{self.args["steps"]}\" departLane=\"best\" departPos=\"random\" departSpeed=\"max\" />

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

    def log_info(self):
        log_data = {
            "sim_time": self.get_current_time(),
            "episode": self.ep_count,
            **self.get_scenario_info(),
        }

        try:
            if self.args.get("log_overall_metrics", True):
                log_data["total_running_vehicles"] = (
                    traci.simulation.getDepartedNumber()
                    - traci.simulation.getArrivedNumber()
                )
                log_data["total_departed"] = traci.simulation.getDepartedNumber()
                log_data["total_arrived"] = traci.simulation.getArrivedNumber()
        except traci.TraCIException:
            pass

        return log_data

    def get_veh_type(self, veh_id):
        try:
            return traci.vehicle.getTypeID(veh_id)
        except traci.TraCIException:
            return ""

    def is_veh_con(self, veh_id):
        return self.get_veh_type(veh_id) == self.args.get("v_type_con", "con")
