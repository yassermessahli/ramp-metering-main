# rl_env/custom_env/sumo_env.py

# Specifies Python 2/3 compatibility features.

# Import the SUMO_PARAMS dictionary
import json
import random

# Import standard Python libraries.
import sys

from colorama import Fore

from .utils import SUMO_PARAMS  # Make sure SUMO_PARAMS includes 'v_max_speed'

# import numpy as np # Not used in this snippet
# from itertools import permutations # Not used in this snippet

# Define the path to the SUMO installation directory.
SUMO_HOME = "../sumo/"  # Or your actual SUMO_HOME if different

# Append the SUMO tools directory to the Python path
# Ensure this path is correct for your system
if SUMO_HOME not in sys.path:  # Avoid adding multiple times if script is re-run
    sys.path.append(SUMO_HOME + "tools")

# Import SUMO libraries.
try:
    from sumolib import net  # noqa
    import traci  # noqa
except ImportError:
    sys.exit(
        "Please declare the SUMO_HOME environment variable or ensure 'sumo/tools' is in sys.path."
    )


# Define the main class representing the SUMO simulation environment.
class SumoEnv:
    # Define a relative path for environment-specific configuration and data files.
    SUMO_ENV = "env/custom_env/"  # Relative to project root where play.py/train.py are run

    # --- Static Methods (Pretty Print, ArgMax, ArgMin, Clip) ---
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
        # self.gui = False # Temp set for setup - Handled by actual gui flag later
        self.config = self.args["config"]
        self.data_dir = self.SUMO_ENV + "data/" + self.config + "/"
        self.net_file_name = self.config + ".net.xml"  # Or "ramp.net.xml" if names differ

        try:
            self.net = net.readNet(self.data_dir + self.net_file_name)
        except Exception as e:
            print(f"Error reading net file: {self.data_dir + self.net_file_name}")
            print(e)
            sys.exit(1)

        self.tl_ids = [tl.getID() for tl in self.net.getTrafficLights()]
        if not self.tl_ids:
            print("Warning: No traffic lights (ramp meters) found in the network.")
            self.ramp_meter_id = None
        else:
            self.ramp_meter_id = self.tl_ids[0]  # Assume first TL is the ramp meter

        # Edge ID Constants (based on your 1ramp_1x3.net.xml)
        self.UPSTREAM_EDGE = "main_road"
        # self.BOTTLENECK_EDGE = "acceleration_area"
        self.MERGING_EDGE = (
            "acceleration_area"  # Often the same as MERGING_EDGE, or just downstream
        )
        self.DOWNSTREAM_EDGE = "end_main_road"  # Edge after the merge
        self.ON_RAMP_EDGE = "on_ramp"  # Edge before the ramp meter signal

        # Normalization Constants
        self.FREEFLOW_SPEED_MPS = self.args.get("v_max_speed", 27.77)  # m/s, from SUMO_PARAMS
        # Estimate max queue based on on_ramp length (204.44m for on_ramp_0) and veh size
        # (5m veh + 2.5m gap = 7.5m per veh). 204 / 7.5 ~ 27 veh.
        self.MAX_RAMP_QUEUE_VEH = self.args.get("max_ramp_queue_veh", 25)
        self.MAX_LANE_FLOW_VPH = self.args.get("max_lane_flow_vph", 1900)  # veh/hr/lane
        self.MAX_FLOW_UPSTREAM_VPH = self.args.get("max_flow_upstream_vph", 5490)  # veh/hr
        self.MAX_FLOW_MERGING_VPH = self.args.get("max_flow_merging_vph", 5490)  # veh/hr
        self.MAX_FLOW_DOWNSTREAM_VPH = self.args.get("max_flow_downstream_vph", 5760)
        self.MAX_OCCUPANCY_PERCENT = 100.0

        # Set final flags based on constructor arguments.
        self.gui = gui  # Use the passed gui flag
        self.log = log
        self.rnd_params = rnd  # Store rnd for potential use
        self.seed = self.args.get("seed", False)  # Optional seed for reproducibility
        self.ep_count = 0

        # --- ADD THIS LINE ---
        # Generate the first route file before starting SUMO
        self._generate_route_file()

        # Generate the final SUMO command-line parameters.
        self.params = self.set_params()

        # Start TraCI connection
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
        params = [
            "sumo" + ("-gui" if self.gui else ""),
            "-c",
            sumocfg_path,
            "--tripinfo-output",
            self.data_dir + "tripinfo.xml",  # Ensure this dir exists
            "--time-to-teleport",
            str(
                self.args.get("time_to_teleport", self.args["steps"])
            ),  # Default to steps if not specified
            "--waiting-time-memory",
            str(self.args.get("waiting_time_memory", self.args["steps"])),
            "--log",
            "log",  #! added
            "--no-warnings",
            "true",  #! added
        ]
        if self.seed:
            params += [
                "--seed",
                str(self.args.get("seed_value", 42)),
            ]  # Default seed value if not specified
        if self.gui:
            params += [
                "--delay",
                str(self.args.get("delay", 0)),
                "--start",
                "true",
                "--quit-on-end",
                "true",  # Keep SUMO open after simulation ends
            ]
            gui_settings_file = self.SUMO_ENV + "data/" + self.config + "/gui-settings.cfg"
            # Only add gui-settings-file if it exists, to avoid SUMO error
            import os

            if os.path.exists(gui_settings_file):
                params.append("--gui-settings-file")
                params.append(gui_settings_file)
            else:
                print(
                    f"Note: GUI settings file not found at {gui_settings_file}, using SUMO defaults."
                )

        return params

    # --- Simulation Control Wrappers ---
    def start(self):
        try:
            traci.start(self.params)
            self.sim_step_length = traci.simulation.getDeltaT()
        except traci.TraCIException as e:
            print(f"Error starting TraCI during explicit start(): {e}")
            sys.exit(1)

    def stop(self):
        try:
            traci.close()
        except traci.TraCIException:  # SUMO might have already closed
            pass
        sys.stdout.flush()

    def simulation_reset(self):
        self.stop()
        self.ep_count += 1  # Increment episode count on reset

        # --- ADD THIS LINE ---
        # Generate a new route file for the new episode before starting SUMO
        self._generate_route_file()

        self.start()

    def simulation_step(self):
        try:
            traci.simulationStep()
        except traci.TraCIException as e:
            print(f"Error during simulation step: {e}. SUMO may have closed.")
            # Decide how to handle this - maybe raise an exception to end episode
            raise e  # Or return a special status

    # --- Abstract DRL Methods (to be implemented by subclasses like RLController) ---
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
        return {} if not self.log else self.log_info()  # log_info will be more detailed

    # --- General SUMO State Getters ---
    def is_simulation_end(self):
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except traci.TraCIException:  # If connection is lost
            return True

    def get_current_time(self):  # Returns simulation time in seconds
        try:
            return traci.simulation.getTime()
        except traci.TraCIException:
            return -1  # Indicate error or end

    # --- Traffic Light Getters/Setters ---
    def get_phase(self, tl_id):
        return traci.trafficlight.getPhase(tl_id)

    def get_ryg_state(self, tl_id):
        return traci.trafficlight.getRedYellowGreenState(tl_id)

    def set_phase(self, tl_id, phase_index):
        traci.trafficlight.setPhase(tl_id, phase_index)

    def set_phase_duration(self, tl_id, duration_sec):
        traci.trafficlight.setPhaseDuration(tl_id, duration_sec)

    # --- Helper Methods for Detector Data (from your previous input) ---
    def get_lanes_of_edge(self, edge_id):
        edge_lanes = []
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for i in range(num_lanes):
                edge_lanes.append(f"{edge_id}_{i}")
        except traci.TraCIException:
            print(f"Warning: SumoEnv - Could not get lanes for edge {edge_id}")
        return edge_lanes

    # === Edge Information Getters ===
    # ...
    def get_edge_lane_n(self, edge_id):
        """Gets the number of lanes on the specified edge."""
        return traci.edge.getLaneNumber(edge_id)

    def get_edge_induction_loops(self, edge_id):
        lanes = self.get_lanes_of_edge(edge_id)
        if not lanes:
            return []
        all_loops = []
        try:
            all_loops = traci.inductionloop.getIDList()
        except traci.TraCIException:
            print("Warning: SumoEnv - Could not get induction loop ID list.")
            return []
        return [loop_id for loop_id in all_loops if traci.inductionloop.getLaneID(loop_id) in lanes]

    def get_loops_flow_interval(self, loop_ids, interval_duration_sec):
        if not loop_ids or interval_duration_sec <= 0:
            return 0.0
        total_vehicles = 0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                total_vehicles += traci.inductionloop.getLastIntervalVehicleNumber(loop_id)
                valid_loops += 1
            except traci.TraCIException:
                print(
                    f"Warning: SumoEnv - Could not get interval vehicle number for loop {loop_id}"
                )
        return (total_vehicles * 3600.0) / interval_duration_sec if valid_loops > 0 else 0.0

    def get_edge_flow_from_loops_interval(self, edge_id, interval_duration_sec):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_flow_interval(loops, interval_duration_sec)

    def get_loops_occupancy_interval(self, loop_ids):  # Returns average %
        if not loop_ids:
            return 0.0
        total_occupancy = 0.0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                total_occupancy += traci.inductionloop.getLastIntervalOccupancy(loop_id)
                valid_loops += 1
            except traci.TraCIException:
                print(f"Warning: SumoEnv - Could not get interval occupancy for loop {loop_id}")
        return total_occupancy / valid_loops if valid_loops > 0 else 0.0

    def get_edge_occupancy_from_loops_interval(self, edge_id):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_occupancy_interval(loops)

    # --- mean speed over interval from loops ---
    def get_loops_mean_speed_interval(self, loop_ids):  # Returns m/s
        if not loop_ids:
            return 0.0
        total_speed = 0.0
        valid_loops = 0
        for loop_id in loop_ids:
            try:
                speed = traci.inductionloop.getLastIntervalMeanSpeed(loop_id)
                if speed >= 0:  # getLastIntervalMeanSpeed returns -1 if no vehicle passed
                    total_speed += speed
                    valid_loops += 1
            except traci.TraCIException:
                print(f"Warning: SumoEnv - Could not get interval mean speed for loop {loop_id}")
        return total_speed / valid_loops if valid_loops > 0 else 0.0  # Return 0 if no vehicles/data

    def get_edge_mean_speed_from_loops_interval(self, edge_id):
        loops = self.get_edge_induction_loops(edge_id)
        return self.get_loops_mean_speed_interval(loops)

    def get_edge_ls_mean_speed(self, edge_id):
        return traci.edge.getLastStepMeanSpeed(edge_id)  # Returns m/s

    def get_loops_flow_weigthed_mean_speed(self, loop_ids):
        """
        Calculates the flow-weighted mean speed from a list of induction loop IDs.
        Returns the average speed weighted by the number of vehicles detected.
        """
        if not loop_ids:
            return 0.0
        total_speed = 0.0
        total_flow = 0.0
        for loop_id in loop_ids:
            try:
                flow = traci.inductionloop.getLastStepVehicleNumber(loop_id)
                speed = traci.inductionloop.getLastStepMeanSpeed(loop_id)
                if flow > 0 and speed >= 0:  # Only consider valid data
                    total_speed += speed * flow
                    total_flow += flow
            except traci.TraCIException:
                print(f"Warning: SumoEnv - Could not get data for loop {loop_id}")
            # in Km/h
        return (total_speed / total_flow) if total_flow > 0 else 0.0  # return in m/s

    # --- Other existing helpers if needed (getLastStep versions, vehicle specific, etc.) ---
    def get_edge_ls_queue_length_vehicles(self, edge_id):
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.TraCIException:
            print(f"Warning: SumoEnv - Could not get vehicle number for edge {edge_id}")
            return 0

    def get_detector_vehicle_count_last_step(self, detector_id):  # Renamed for clarity
        """Gets vehicle number from a specific detector from the last step."""
        try:  # Try as E1 induction loop first
            return traci.inductionloop.getLastStepVehicleNumber(detector_id)
        except traci.TraCIException:
            try:  # Fallback for E2 lane area detector
                return traci.laneareadetector.getLastStepVehicleNumber(detector_id)
            except traci.TraCIException:
                print(f"Warning: SumoEnv - Could not get vehicles for detector {detector_id}")
                return 0

    def get_veh_speed(self, veh_id):  # Example of keeping a vehicle-specific getter
        try:
            return traci.vehicle.getSpeed(veh_id)
        except traci.TraCIException:
            return 0.0  # Or handle as error

    def _generate_route_file(self):
        """
        Generates a new .rou.xml file for the simulation with randomized
        traffic flows based on weighted choices and a random penetration
        rate for connected vehicles.
        """
        # Select total flows for each route using weighted random choice
        main_flow = random.choices(
            self.args["veh_per_hour_main"], weights=self.args["veh_per_hour_main_weights"]
        )[0]
        on_ramp_flow = random.choices(
            self.args["veh_per_hour_on_ramp"], weights=self.args["veh_per_hour_on_ramp_weights"]
        )[0]
        off_ramp_flow = random.choices(
            self.args["veh_per_hour_off_ramp"], weights=self.args["veh_per_hour_off_ramp_weights"]
        )[0]

        # Generate a random penetration rate for connected vehicles
        min_pen, max_pen = self.args["con_penetration_rate_range"]
        pen_rate = random.uniform(min_pen, max_pen)

        # Calculate the number of vehicles for each type (connected vs. default)
        main_con = int(main_flow * pen_rate)
        main_def = int(main_flow * (1 - pen_rate))
        on_ramp_con = int(on_ramp_flow * pen_rate)
        on_ramp_def = int(on_ramp_flow * (1 - pen_rate))
        off_ramp_con = int(off_ramp_flow * pen_rate)
        off_ramp_def = int(off_ramp_flow * (1 - pen_rate))

        # NOTE: The <route> definitions below are hardcoded from 1 ramp.
        # For other networks (e.g., 3ramp_...), we need to make these dynamic
        # or create separate generation functions.
        xml_content = f"""<!-- Generated on-the-fly for episode {self.ep_count + 1} -->
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

        <!-- Vehicle Type Definitions -->
        <vType id="def" vClass="passenger" length="5.0" minGap="2.5" accel="2.6" decel="4.5" maxSpeed="35" sigma="0.9" />
        <vType id="con" vClass="passenger" length="5.0" minGap="2.5" accel="2.6" decel="4.5" maxSpeed="35" sigma="0.8" color="1,0,0" />

        <!-- Route Definitions -->
        <route id="entry_to_end_main_road" edges="entry off_ramp_up_stream main_road acceleration_area end_main_road" />
        <route id="entry_to_off_ramp" edges="entry off_ramp_up_stream off_ramp_beginning off_ramp" />
        <route id="on_ramp_to_end_main_road" edges="on_ramp passage_area acceleration_area end_main_road" />

        <!-- Flow Definitions -->
        <flow id="main_con" type="con" vehsPerHour="{main_con}" route="entry_to_end_main_road" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />
        <flow id="main_def" type="def" vehsPerHour="{main_def}" route="entry_to_end_main_road" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />
        <flow id="on_ramp_con" type="con" vehsPerHour="{on_ramp_con}" route="on_ramp_to_end_main_road" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />
        <flow id="on_ramp_def" type="def" vehsPerHour="{on_ramp_def}" route="on_ramp_to_end_main_road" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />
        <flow id="off_ramp_con" type="con" vehsPerHour="{off_ramp_con}" route="entry_to_off_ramp" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />
        <flow id="off_ramp_def" type="def" vehsPerHour="{off_ramp_def}" route="entry_to_off_ramp" begin="0" end="{self.args["steps"]}" departLane="best" departPos="random" departSpeed="max" />

    </routes>
    """
        # Write the content to the .rou.xml file, overwriting the previous one
        route_file_path = self.data_dir + self.config + ".rou.xml"
        with open(route_file_path, "w") as f:
            f.write(xml_content)

        print(
            Fore.LIGHTMAGENTA_EX,
            f"Generated new route file for Ep {self.ep_count + 1}: Main={main_flow}, Ramp={on_ramp_flow}, PenRate={pen_rate:.2f}",
            Fore.RESET,
        )

    # --- Logging Information ---
    def log_info(self):
        """
        Calculates and returns simulation metrics for ramp metering.
        This will be populated by data collected and processed by RLController.
        The RLController should pass its computed state and reward components here.
        """
        # Placeholder - RLController will pass the actual data to be logged
        # For now, just basic info.
        log_data = {
            "sim_time": self.get_current_time(),
            "episode": self.ep_count,
            # The actual state and reward values will be added by RLController
            # when it prepares its info dict.
        }
        # Add more specific data here if SumoEnv is to log things independently of RLController's state/reward
        # For example, overall network stats if desired.
        try:
            if self.args.get("log_overall_metrics", True):  # Example: add a param to SUMO_PARAMS
                log_data["total_running_vehicles"] = (
                    traci.simulation.getDepartedNumber() - traci.simulation.getArrivedNumber()
                )
                log_data["total_departed"] = traci.simulation.getDepartedNumber()
                log_data["total_arrived"] = traci.simulation.getArrivedNumber()
        except traci.TraCIException:
            pass  # Could not get overall metrics

        return log_data

    # --- Vehicle Specific Getters ---
    def get_veh_type(self, veh_id):
        try:
            return traci.vehicle.getTypeID(veh_id)
        except traci.TraCIException:
            return ""

    def is_veh_con(self, veh_id):
        return self.get_veh_type(veh_id) == self.args.get("v_type_con", "con")
