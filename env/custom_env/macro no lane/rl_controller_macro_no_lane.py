# rl_env/custom_env/rl_controller.py

import numpy as np

from ..sumo_env import SumoEnv

# import traci # Not strictly needed here if all traci calls are via self.xxx methods from SumoEnv


class RLController(SumoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Calls SumoEnv.__init__

        # --- RLController Specific Initializations ---
        self.CYCLE_DURATION_SEC = 40.0
        self.ty = 3  # duration of the yellow light cycle in seconds
        # self.sim_step_length is inherited from SumoEnv, fetched after traci.start()

        # Action Space Definition
        self.green_time_actions_sec = np.array(
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )
        self.action_space_n = len(self.green_time_actions_sec)

        # Ramp Meter Phase Indices (ensure these match your SUMO TL definition)
        self.green_phase_index = 0
        self.red_phase_index = 1

        # ---- Detector ID Initialization ----
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

        # ---- Observation Space Definition ----
        self.observation_space_n = 8

        # ---- State Variables for RL control cycle ----
        self.last_action_value_sec = self.green_time_actions_sec[
            0
        ]  # Store the actual green time value

        # Aggregators for data collected over one 40s cycle
        self._reset_cycle_aggregators()

        # Variables to store processed data at the end of a cycle for obs/rew
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

        self._last_detailed_info = {}  # To store info from the last step
        self._initialize_last_detailed_info_placeholders()  #  Initialize with placeholders

    def _initialize_last_detailed_info_placeholders(self):
        """Initializes _last_detailed_info with all expected keys and default values."""
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
            "reward_outflow_speed_comp": 0.0,
            "reward_throughput_comp": 0.0,
            "penalty_ramp_queue_comp": 0.0,
            "penalty_bottleneck_occ_comp": 0.0,
            "penalty_spillback_comp": 0.0,
            # Basic info from SumoEnv.log_info() that will be updated
            "sim_time": 0.0,
            "episode": 0,  # Will be updated by SumoEnv.log_info()
            "total_running_vehicles": 0,
            "total_departed": 0,
            "total_arrived": 0,
            # Standard wrapper keys
            "l": 0,
            "r": 0.0,
            "TimeLimit.truncated": False,
            "done": False,
        }
        # Add any other keys that might appear in baselines if necessary for perfect header match
        # For example, if baselines have specific keys not covered here.
        # The current set aims to match the provided baseline log closely.

    def _reset_cycle_aggregators(self):
        """Resets accumulators for data collected over a control cycle."""
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

    def _collect_data_at_cycle_end(self):
        """Processes aggregated data at the end of a 40s cycle to form metrics."""
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

    def reset(self):
        self.simulation_reset()  # Calls super().simulation_reset()
        self._reset_cycle_aggregators()
        self.last_action_value_sec = self.green_time_actions_sec[0]
        self._initialize_last_detailed_info_placeholders()  # Re-initialize placeholders on reset
        self._last_detailed_info.update(
            super().log_info()
        )  # Get initial sim_time, episode

        if self.ramp_meter_id and self.red_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.CYCLE_DURATION_SEC)

        num_init_steps = 0
        if self.sim_step_length > 0:  # pragma: no branch
            num_init_steps = int(
                round(max(1.0, 5.0 / self.sim_step_length))
            )  # Simulate for ~5 seconds, at least 1 step.
        else:  # pragma: no cover
            num_init_steps = 5  # Fallback if sim_step_length is somehow 0

        for _ in range(num_init_steps):
            if self.is_simulation_end():
                break
            self.simulation_step()

        self._collect_data_at_cycle_end()  # Populate processed_ values

        # Populate _last_detailed_info with collected data after initial steps for first observation
        # This ensures the info() method can return something meaningful even before the first agent step.
        current_phase_index_init = -1
        current_ryg_state_init = "N/A"
        if self.ramp_meter_id:
            try:
                current_phase_index_init = self.get_phase(self.ramp_meter_id)
                current_ryg_state_init = self.get_ryg_state(self.ramp_meter_id)
            except Exception:
                pass  # pragma: no cover

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
                "ramp_queue_veh": self.processed_ramp_queue_veh,  # Should be 0 or low initially
                "current_tl_phase_index": current_phase_index_init,
                "current_tl_ryg_state": current_ryg_state_init,
                "chosen_green_time_sec": self.last_action_value_sec,  # Initial assumed action
            }
        )
        self._last_detailed_info.update(super().log_info())

        return self._get_current_observation()

    def step(self, action_index):
        if not (0 <= action_index < self.action_space_n):
            # print(f"Warning: RLController received invalid action_index {action_index}. Clamping.") # Optional: for debugging
            action_index = np.clip(
                action_index, 0, self.action_space_n - 1
            ).item()  # .item() if it's a 0-d array

        chosen_green_time_sec = self.green_time_actions_sec[action_index]
        self.last_action_value_sec = chosen_green_time_sec

        red_time_sec = self.CYCLE_DURATION_SEC - chosen_green_time_sec
        if red_time_sec < 0:
            red_time_sec = 0.0

        self._reset_cycle_aggregators()

        if (
            self.ramp_meter_id
            and self.green_phase_index != -1
            and chosen_green_time_sec > 0
        ):
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, chosen_green_time_sec)
            num_steps_green = 0
            if self.sim_step_length > 0:  # pragma: no branch
                num_steps_green = int(
                    round(chosen_green_time_sec / self.sim_step_length)
                )
            # else: num_steps_green = int(chosen_green_time_sec) # Fallback, though sim_step_length should be >0

            for _ in range(num_steps_green):
                if self.is_simulation_end():
                    break
                self.simulation_step()
                self.sum_queue += self.get_edge_ls_queue_length_vehicles(
                    self.ON_RAMP_EDGE
                )

        if self.ramp_meter_id and self.red_phase_index != -1 and red_time_sec > 0:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, red_time_sec)
            num_steps_red = 0
            if self.sim_step_length > 0:  # pragma: no branch
                num_steps_red = int(round(red_time_sec / self.sim_step_length))
            # else: num_steps_red = int(red_time_sec)

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
                pass  # pragma: no cover

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
            "reward_outflow_speed_comp": self._reward_outflow_speed(),
            "reward_throughput_comp": self._reward_throughput(),
            "penalty_ramp_queue_comp": self._penalty_ramp_queue(),
            "penalty_bottleneck_occ_comp": self._penalty_bottleneck_occ(),
            "penalty_spillback_comp": self._penalty_spillback(),
        }

        info_for_this_step.update(
            super().log_info()
        )  # Adds sim_time, episode, total_...

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

        norm_last_action = np.clip(
            self.last_action_value_sec
            / (self.CYCLE_DURATION_SEC if self.CYCLE_DURATION_SEC > 0 else 1.0),
            0,
            1,
        )

        state = np.array(
            [
                # loop detectors features (normalized flow, occupancy, speed)
                norm_flow_upstream,
                norm_flow_merging,
                norm_occ_upstream,
                norm_speed_upstream,
                norm_occ_bottleneck,
                norm_speed_bottleneck,
                norm_ramp_queue,
                # last action (normalized green time)
                norm_last_action,
            ],
            dtype=np.float32,
        )
        return state

    def _reward_outflow_speed(self):
        norm_speed = np.clip(
            self.processed_mainline_speed_downstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        return norm_speed

    def _reward_upstream_speed(self):
        norm_speed = np.clip(
            self.processed_speed_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        return norm_speed

    def _reward_merging_speed(self):
        norm_speed = np.clip(
            self.processed_speed_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        return norm_speed

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
        max_possible_throughput = (
            self.MAX_LANE_FLOW_VPH * self.get_edge_lane_n(self.DOWNSTREAM_EDGE)
            if self.get_edge_lane_n(self.DOWNSTREAM_EDGE) > 0
            else self.MAX_LANE_FLOW_VPH
        )
        norm_throughput = np.clip(
            self.processed_mainline_flow_downstream_vph
            / (max_possible_throughput if max_possible_throughput > 0 else 1.0),
            0,
            1,
        )
        return norm_throughput

    def _penalty_ramp_queue(self):
        norm_queue = np.clip(
            self.processed_ramp_queue_veh
            / (self.MAX_RAMP_QUEUE_VEH if self.MAX_RAMP_QUEUE_VEH > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_queue

    def reward_merging_throughput(self):
        max_possible_throughput = (
            self.MAX_LANE_FLOW_VPH * self.get_edge_lane_n(self.MERGING_EDGE)
            if self.get_edge_lane_n(self.MERGING_EDGE) > 0
            else self.MAX_LANE_FLOW_VPH
        )
        norm_throughput = np.clip(
            self.processed_flow_merging_vph
            / (max_possible_throughput if max_possible_throughput > 0 else 1.0),
            0,
            1,
        )
        return norm_throughput

    def _penalty_spillback(self):
        spillback_threshold_veh = (
            0.9 * self.MAX_RAMP_QUEUE_VEH
        )  # The queue length where spillback begins

        # If queue is past the threshold but not over capacity, use a graded penalty
        if self.processed_ramp_queue_veh > spillback_threshold_veh:
            denominator = self.MAX_RAMP_QUEUE_VEH - spillback_threshold_veh
            if denominator < 1e-6:
                denominator = 1e-6  # Avoid division by zero

            # This calculates how "far into" the spillback zone we are, from 0.0 to 1.0
            spill_amount = (
                self.processed_ramp_queue_veh - spillback_threshold_veh
            ) / denominator

            # The penalty scales from 0 to -1. The final weight is handled in _calculate_reward
            return -1.0 * np.clip(spill_amount, 0, 1)

        return 0.0

    def _calculate_reward(self):
        # --- Weights for each component ---
        # Give more weight to the critical merging and upstream areas
        w_speed_merge = 1.5  # Most important speed
        w_speed_up = 1.0  # Second most important
        w_speed_down = 0.5  # Least important, just for stability

        # Penalties
        w_occ_bottle = 2.0  # Occupancy in the bottleneck is a key indicator of collapse
        w_occ_upstream = (
            1.0  # Upstream occupancy is also important, but less than bottleneck
        )
        w_queue = 1.0
        w_spillback = 20.0  # A very large weight to make spillback catastrophic

        # --- Calculate each component ---
        # (+) REWARDS for good mainline conditions
        # Use the specific speed rewards you already wrote!
        r_speed_merge = self._reward_merging_speed()
        r_speed_up = self._reward_upstream_speed()
        r_speed_down = self._reward_outflow_speed()  # Renamed from r_speed for clarity

        # (-) PENALTIES for bad conditions
        p_occ_bottle = self._penalty_bottleneck_occ()
        p_occ_upstream = (
            self._penalty_upstream_occ()
        )  # This is already negative, so we add it directly
        p_queue = self._penalty_ramp_queue()
        # The spillback function returns a negative value, so we add it directly.
        # The weight w_spillback will make it highly punitive.
        p_spillback = self._penalty_spillback()

        # --- Combine into the final reward ---
        reward = (
            (w_speed_merge * r_speed_merge)
            + (w_speed_up * r_speed_up)
            + (w_speed_down * r_speed_down)
            + (w_occ_bottle * p_occ_bottle)  # This is already negative
            + (w_occ_upstream * p_occ_upstream)  # This is already negative
            + (w_queue * p_queue)  # This is already negative
            + (w_spillback * p_spillback)
        )  # This is already negative

        return float(reward)

    def obs(self):
        # This method is called by DqnEnv.obs() -> CustomEnvWrapper._obs()
        # The observation is computed at the end of step() and used directly by the DRL agent.
        # This obs() method ensures that if anything else calls it, it gets the latest.
        return self._get_current_observation()

    def rew(self):
        # This method is called by DqnEnv.rew() -> CustomEnvWrapper._rew()
        # Similar to obs(), reward is computed at the end of step().
        return self._calculate_reward()

    def done(self):
        # This method is called by DqnEnv.done() -> CustomEnvWrapper._done()
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

    def info(self):
        """
        Returns the detailed information dictionary from the last completed step.
        This is called by DqnEnv.info() -> CustomEnvWrapper._info().
        """
        # The _last_detailed_info should be populated by __init__/reset and then by step()
        # It already includes sim_time, episode, etc. from super().log_info()
        return self._last_detailed_info
