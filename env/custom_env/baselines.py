import numpy as np
import traci

from .sumo_env import SumoEnv


class BaselineMeta(SumoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space_n = 1
        self.observation_space_n = 1
        self._last_step_info = {}
        self.us_loops = ["up_stream_sens_0", "up_stream_sens_1", "up_stream_sens_2"]
        self.ma_loops = [
            "bottle_neck_sens_0",
            "bottle_neck_sens_1",
            "bottle_neck_sens_2",
            "bottle_neck_sens_3",
        ]
        self.ds_loops = self.get_edge_induction_loops(self.DOWNSTREAM_EDGE)

        self.ramp_meter_id = None
        self.green_phase_index = 0
        self.red_phase_index = 1
        if self.tl_ids:
            self.ramp_meter_id = self.tl_ids[0]
            self._setup_tl_program()

    def _setup_tl_program(self):
        if not self.ramp_meter_id:
            return
        try:
            program_id = "external_control_program"
            phases = [
                traci.trafficlight.Phase(duration=3600, state="G", name="Green"),
                traci.trafficlight.Phase(duration=3600, state="r", name="Red"),
            ]
            logic = traci.trafficlight.Logic(
                programID=program_id, type=0, currentPhaseIndex=0, phases=phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.ramp_meter_id, logic)
            traci.trafficlight.setProgram(self.ramp_meter_id, program_id)
            self.green_phase_index = 0
            self.red_phase_index = 1
        except traci.TraCIException as e:
            print(f"[ERROR] Failed to set up TL program: {e}")

    def simulation_reset(self):
        super().simulation_reset()
        self._setup_tl_program()

    def reset(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def step(self, action):
        raise NotImplementedError

    def obs(self):
        return []

    def rew(self):
        return 0

    def done(self):
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

    def _collect_common_metrics(self):
        metrics = super().log_info()
        detector_period = self.args.get("alinea_detector_period_sec", 40.0)
        metrics["mainline_flow_upstream_v/h"] = self.get_loops_flow_interval(
            self.us_loops, detector_period
        )
        metrics["mainline_occ_upstream_percent"] = self.get_loops_occupancy_interval(self.us_loops)
        metrics["mainline_speed_upstream_km/h"] = self.get_loops_flow_weigthed_mean_speed(
            self.us_loops
        )
        metrics["mainline_flow_mergeArea_v/h"] = self.get_loops_flow_interval(
            self.ma_loops, detector_period
        )
        metrics["mainline_occ_mergeArea_percent"] = self.get_loops_occupancy_interval(self.ma_loops)
        metrics["mainline_speed_mergeArea_km/h"] = self.get_loops_flow_weigthed_mean_speed(
            self.ma_loops
        )
        metrics["mainline_flow_downstream_v/h"] = self.get_loops_flow_interval(
            self.ds_loops, detector_period
        )
        metrics["mainline_occ_downstream_percent"] = self.get_loops_occupancy_interval(
            self.ds_loops
        )
        metrics["mainline_speed_downstream_km/h"] = self.get_loops_flow_weigthed_mean_speed(
            self.ds_loops
        )
        metrics["ramp_queue_veh"] = self.get_edge_ls_queue_length_vehicles(self.ON_RAMP_EDGE)
        if self.ramp_meter_id:
            try:
                metrics["current_tl_phase_index"] = self.get_phase(self.ramp_meter_id)
                metrics["current_tl_ryg_state"] = self.get_ryg_state(self.ramp_meter_id)
            except traci.TraCIException:
                metrics["current_tl_phase_index"] = -1
                metrics["current_tl_ryg_state"] = "unknown"
        return metrics

    def info(self):
        return self._last_step_info

    def _update_log_info(self):
        self._last_step_info = self._collect_common_metrics()


class AlwaysGreenBaseline(BaselineMeta):
    def reset(self):
        self.simulation_reset()
        if self.ramp_meter_id is not None:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
        self._update_log_info()

    def step(self, action):
        self.simulation_step()
        self._update_log_info()


class FixedCycleBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tg_sec = 20.0
        self.tr_sec = 20.0
        self.time_in_phase_sec = 0.0
        self.is_green = True

    def reset(self):
        self.simulation_reset()
        self.time_in_phase_sec = 0.0
        self.is_green = True
        if self.ramp_meter_id:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
        self._update_log_info()

    def step(self, action):
        if self.ramp_meter_id is None:
            self.simulation_step()
            self._update_log_info()
            return

        if self.is_green and self.time_in_phase_sec >= self.tg_sec:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.is_green = False
            self.time_in_phase_sec = 0.0
        elif not self.is_green and self.time_in_phase_sec >= self.tr_sec:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.is_green = True
            self.time_in_phase_sec = 0.0

        self.simulation_step()
        self.time_in_phase_sec += self.sim_step_length
        self._update_log_info()

    def _update_log_info(self):
        super()._update_log_info()
        self._last_step_info.update(
            {
                "baseline_specific_action": "FixedCycle",
                "fixed_cycle_is_green": self.is_green,
                "fixed_cycle_time_in_phase": self.time_in_phase_sec,
            }
        )


class AlineaDsBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CYCLE_LENGTH_SEC = self.args.get("alinea_detector_period_sec", 40.0)
        self.CRITICAL_OCCUPANCY_PERCENT = 17
        self.KR = 60
        self.MIN_METERING_RATE_VPH = 180
        self.MAX_METERING_RATE_VPH = 1900
        self.MIN_GREEN_TIME_SEC = 3.0
        self.RAMP_SATURATION_FLOW_VPS = 0.5
        self.time_in_cycle_sec = 0.0
        self.active_green_time_sec = 0.0
        self.downstream_detector_ids = []
        self.current_metering_rate_vph = 0
        self.measured_downstream_occ_for_log = 0.0

    def reset(self):
        self.simulation_reset()
        self.downstream_detector_ids = [
            "bottle_neck_sens_0",
            "bottle_neck_sens_1",
            "bottle_neck_sens_2",
            "bottle_neck_sens_3",
        ]
        self.current_metering_rate_vph = (
            self.MAX_METERING_RATE_VPH + self.MIN_METERING_RATE_VPH
        ) / 2
        self.time_in_cycle_sec = self.CYCLE_LENGTH_SEC
        self.active_green_time_sec = self.MIN_GREEN_TIME_SEC
        self.measured_downstream_occ_for_log = 0.0
        self._update_log_info()

    def _get_downstream_occupancy(self):
        if not self.downstream_detector_ids:
            return 0.0
        occ = self.get_loops_occupancy_interval(self.downstream_detector_ids)
        self.measured_downstream_occ_for_log = occ
        return occ

    def _calculate_new_cycle_times(self):
        measured_occupancy = self._get_downstream_occupancy()
        occupancy_error = self.CRITICAL_OCCUPANCY_PERCENT - measured_occupancy
        new_metering_rate = self.current_metering_rate_vph + self.KR * occupancy_error
        self.current_metering_rate_vph = np.clip(
            new_metering_rate, self.MIN_METERING_RATE_VPH, self.MAX_METERING_RATE_VPH
        )
        vehs_per_cycle = self.current_metering_rate_vph * (self.CYCLE_LENGTH_SEC / 3600.0)
        if self.RAMP_SATURATION_FLOW_VPS > 0:
            calculated_tg = vehs_per_cycle / self.RAMP_SATURATION_FLOW_VPS
        else:
            calculated_tg = self.MIN_GREEN_TIME_SEC
        self.active_green_time_sec = np.clip(
            calculated_tg, self.MIN_GREEN_TIME_SEC, self.CYCLE_LENGTH_SEC
        )

    def step(self, action):
        if self.ramp_meter_id is None:
            self.simulation_step()
            self._update_log_info()
            return

        if self.time_in_cycle_sec >= self.CYCLE_LENGTH_SEC:
            self._calculate_new_cycle_times()
            self.time_in_cycle_sec = 0.0

        if self.time_in_cycle_sec < self.active_green_time_sec:
            if self.get_phase(self.ramp_meter_id) != self.green_phase_index:
                self.set_phase(self.ramp_meter_id, self.green_phase_index)
        else:
            if self.get_phase(self.ramp_meter_id) != self.red_phase_index:
                self.set_phase(self.ramp_meter_id, self.red_phase_index)

        self.simulation_step()
        self.time_in_cycle_sec += self.sim_step_length
        self._update_log_info()

    def _update_log_info(self):
        super()._update_log_info()
        active_red_time_sec = self.CYCLE_LENGTH_SEC - self.active_green_time_sec
        self._last_step_info.update(
            {
                "baseline_specific_action": "Alinea",
                "alinea_measured_downstream_occ_percent": self.measured_downstream_occ_for_log,
                "alinea_current_metering_rate_vph": self.current_metering_rate_vph,
                "alinea_target_green_time_sec": self.active_green_time_sec,
                "alinea_target_red_time_sec": active_red_time_sec,
            }
        )


class PiAlineaDsBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CYCLE_LENGTH_SEC = self.args.get("alinea_detector_period_sec", 40.0)
        self.CRITICAL_OCCUPANCY_PERCENT = 17
        self.KP = 60.0
        self.KI = 10.0
        self.MIN_METERING_RATE_VPH = 180
        self.MAX_METERING_RATE_VPH = 1800
        self.MIN_GREEN_TIME_SEC = 3.0
        self.RAMP_SATURATION_FLOW_VPS = 0.5
        self.time_in_cycle_sec = 0.0
        self.active_green_time_sec = 0.0
        self.integral_term = 0.0
        self.downstream_detector_ids = []
        self.current_metering_rate_vph = 0
        self.measured_downstream_occ_for_log = 0.0

    def reset(self):
        self.simulation_reset()
        self.downstream_detector_ids = [
            "bottle_neck_sens_0",
            "bottle_neck_sens_1",
            "bottle_neck_sens_2",
            "bottle_neck_sens_3",
        ]
        self.current_metering_rate_vph = (
            self.MAX_METERING_RATE_VPH + self.MIN_METERING_RATE_VPH
        ) / 2
        self.integral_term = 0.0
        self.time_in_cycle_sec = self.CYCLE_LENGTH_SEC
        self.active_green_time_sec = self.MIN_GREEN_TIME_SEC
        self.measured_downstream_occ_for_log = 0.0
        self._update_log_info()

    def _get_downstream_occupancy(self):
        if not self.downstream_detector_ids:
            return 0.0
        occ = self.get_loops_occupancy_interval(self.downstream_detector_ids)
        self.measured_downstream_occ_for_log = occ
        return occ

    def _calculate_new_cycle_times(self):
        measured_occupancy = self._get_downstream_occupancy()
        error = self.CRITICAL_OCCUPANCY_PERCENT - measured_occupancy
        self.integral_term += error
        rate_change = self.KP * error + self.KI * self.integral_term
        new_metering_rate = self.current_metering_rate_vph + rate_change
        if (
            new_metering_rate > self.MAX_METERING_RATE_VPH
            or new_metering_rate < self.MIN_METERING_RATE_VPH
        ):
            self.integral_term -= error
        self.current_metering_rate_vph = np.clip(
            new_metering_rate, self.MIN_METERING_RATE_VPH, self.MAX_METERING_RATE_VPH
        )
        vehs_per_cycle = self.current_metering_rate_vph * (self.CYCLE_LENGTH_SEC / 3600.0)
        if self.RAMP_SATURATION_FLOW_VPS > 0:
            calculated_tg = vehs_per_cycle / self.RAMP_SATURATION_FLOW_VPS
        else:
            calculated_tg = self.MIN_GREEN_TIME_SEC
        self.active_green_time_sec = np.clip(
            calculated_tg, self.MIN_GREEN_TIME_SEC, self.CYCLE_LENGTH_SEC
        )

    def step(self, action):
        if self.ramp_meter_id is None:
            self.simulation_step()
            self._update_log_info()
            return

        if self.time_in_cycle_sec >= self.CYCLE_LENGTH_SEC:
            self._calculate_new_cycle_times()
            self.time_in_cycle_sec = 0.0

        if self.time_in_cycle_sec < self.active_green_time_sec:
            if self.get_phase(self.ramp_meter_id) != self.green_phase_index:
                self.set_phase(self.ramp_meter_id, self.green_phase_index)
        else:
            if self.get_phase(self.ramp_meter_id) != self.red_phase_index:
                self.set_phase(self.ramp_meter_id, self.red_phase_index)

        self.simulation_step()
        self.time_in_cycle_sec += self.sim_step_length
        self._update_log_info()

    def _update_log_info(self):
        super()._update_log_info()
        active_red_time_sec = self.CYCLE_LENGTH_SEC - self.active_green_time_sec
        self._last_step_info.update(
            {
                "baseline_specific_action": "PiAlinea",
                "pialinea_measured_downstream_occ_percent": self.measured_downstream_occ_for_log,
                "pialinea_current_metering_rate_vph": self.current_metering_rate_vph,
                "pialinea_target_green_time_sec": self.active_green_time_sec,
                "pialinea_target_red_time_sec": active_red_time_sec,
            }
        )
