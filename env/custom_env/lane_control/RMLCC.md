# RMLCC — Ramp Metering + Lane Control Coupled Variant

> A new variant of the project that augments the existing ramp meter controller with a co-located **Variable Speed Limit (VSL)** actuator on the merging-area lane closest to the on-ramp. A single DRL agent jointly chooses, every 40 s, the ramp green time and whether that mainline lane is "open" or "closed" (forced lane-change incentive).

---

## Motivation

The active baseline variant (`env/custom_env/rl_controller.py`, "micro + macro lane") controls only the ramp meter. When mainline demand is high, ramp vehicles struggle to merge even with optimal metering because the rightmost mainline lane is saturated.

**Hypothesis:** Setting a low speed limit (5 m/s) on the rightmost mainline lane in the merging zone (`acceleration_area_1`) discourages its use, pushing mainline traffic leftward and opening a smoother merge corridor for ramp vehicles. Jointly tuning this VSL with the ramp meter should improve bottleneck throughput and shorten ramp queues compared to ramp metering alone.

---

## MDP Formulation

### State Space *(unchanged from base variant — 284-d)*

| Block | Dim | Content |
|---|---|---|
| Macro | 14 | norm. flows / occupancies / speeds for upstream + merging + lane-0; ramp queue; last green time |
| Micro grid | 270 | 2 channels × 27 rows × 5 cols (speed, presence) of connected vehicles in a 216 m window around the merge |

The lane-closure status is **not** explicitly encoded — the agent infers its effect from the micro grid and the lane-0 macro features. This keeps parity with the active variant and avoids touching the network architecture. (Future extension: append a binary `last_lane_action` flag → 15-d macro.)

### Action Space — 16 discrete actions (joint)

```
green_idx = action_index % 8       # {0..7}  → green time ∈ {5, 10, 15, 20, 25, 30, 35, 40} s
lane_idx  = action_index // 8      # {0, 1}  → 0 = lane open, 1 = lane closed (VSL = 5 m/s)
```

| Action | Green (s) | Lane | … | Action | Green (s) | Lane |
|---|---|---|---|---|---|---|
| 0 | 5 | open | … | 8 | 5 | closed |
| 1 | 10 | open | … | 9 | 10 | closed |
| … | … | open | … | … | … | closed |
| 7 | 40 | open | … | 15 | 40 | closed |

### Transition Model

One agent step = one 40 s control cycle:

1. Decode `(green_idx, lane_idx)` from action.
2. **Apply VSL**: `traci.lane.setMaxSpeed("acceleration_area_1", 5.0 if lane_idx==1 else default)`. Setting persists for the full 40 s cycle (both sub-phases).
3. **Green sub-phase**: ramp light green for `chosen_green_time_sec`; SUMO simulated for that many seconds; ramp queue length accumulated each step.
4. **Red sub-phase**: ramp light red for `40 − green_time` seconds; same accumulation.
5. Aggregate detector data at cycle end → observation, reward, info.

On `reset()`, SUMO reloads the network so the lane's original `maxSpeed` is restored automatically.

### Reward *(unchanged formula and weights)*

```
reward = +1.5 · r_speed_merge
       + 1.0 · r_speed_upstream
       + 0.5 · r_speed_downstream
       − 2.0 · p_occ_bottleneck
       − 1.0 · p_occ_upstream
       − 1.0 · p_queue
       − 20.0 · p_spillback        # triggers at >90 % of MAX_RAMP_QUEUE_VEH
```

Theoretical range **[−24, +3]**. No explicit cost is added for using the lane closure — the agent must discover that closure helps when the bottleneck is congested and hurts otherwise, purely through the existing speed/occupancy/queue terms.

### Episode Termination

End of SUMO simulation (`is_simulation_end()`) **or** `current_time ≥ args["steps"]` (default 3600 s ≈ 90 agent steps).

---

## Lane-VSL Implementation Details

| Item | Value |
|---|---|
| Target lane ID | `acceleration_area_1` (2nd from right on the merging edge — the rightmost *mainline* lane next to the acceleration lane) |
| TraCI call | `traci.lane.setMaxSpeed(lane_id, speed)` (soft VSL — vehicles slow but are not forbidden) |
| Closed speed | **5.0 m/s** (~18 km/h) |
| Open speed | `traci.lane.getMaxSpeed(lane_id)` captured once at `reset()` after SUMO start |
| Decision cadence | Every 40 s control cycle (re-evaluated, no minimum hold time) |

---

## Network / Hyperparameters

The `TwoStreamHybridNetwork` is **structurally identical** to the active variant:

- Macro stream: 14-d FC features.
- Micro stream: CNN over (2, 27, 5) → flatten.
- Concat → dense `[512, 256]` → Q-values.

Only the **output head width** differs — 16 actions instead of 8 — and propagates automatically from `RLController.action_space_n` through `DqnEnv → CustomEnvWrapper → spaces.Discrete(16)` and into `train.py` (`output_dim=self.env.action_space.n`).

Save/log directories are tagged so checkpoints and TensorBoard logs don't collide with prior variants:

```
./save/1ramp_1x3/lane_control/
./logs/train/1ramp_1x3/lane_control/
```

All other DQN hyperparameters (lr=1e-4, γ=0.99, ε 1.0→0.01 over 2M steps, batch 32, replay 100k–1M, soft τ=1e-3, Huber loss, 2.1M total steps) are unchanged.

---

## Files Created

| Path | Purpose |
|---|---|
| `env/custom_env/lane_control/__init__.py` | Re-exports `RLController`. |
| `env/custom_env/lane_control/rl_controller.py` | New controller: 16-action joint space, decodes `(green_idx, lane_idx)`, applies VSL via TraCI, then runs the existing green/red simulation loop. Adds `lane_closed` to the step `info` dict for telemetry. Captures lane default speed once per episode in `reset()`. |
| `env/custom_env/lane_control/dqn_config.py` | Clone of the active config; sets `VARIANT_TAG = "lane_control"` and appends it to `save_dir` / `log_dir`. Same network architecture and hyperparameters as the base variant. |
| `env/custom_env/lane_control/RMLCC.md` | This document. |

## Files Changed

| Path | Change |
|---|---|
| `env/custom_env/__init__.py` | `RLController` now imported from `.lane_control` instead of `.rl_controller`. Comment shows how to revert. |
| `env/__init__.py` | `HYPER_PARAMS` and `network_config` now imported from `.custom_env.lane_control.dqn_config` instead of `.dqn_config`. Comment shows how to revert. |

## Files Untouched

- `train.py`, `play.py`, `observe.py`, `evaluate.py` — generic entry points.
- `dqn/*` — agent, networks, replay memory, wrappers.
- `env/custom_env/sumo_env.py` — TraCI base class.
- `env/custom_env/baselines.py` — classical baselines (AlwaysGreen, FixedCycle, ALINEA, PI-ALINEA). They remain ramp-meter-only, which is the correct comparison for RL-with-lane-control.
- `env/custom_env/rl_controller.py` and `env/dqn_config.py` — preserved as the previous active variant.

---

## Reverting to the Previous Variant

Two-line revert:

```python
# env/custom_env/__init__.py
from .rl_controller import RLController          # was: from .lane_control import RLController

# env/__init__.py
from .dqn_config import HYPER_PARAMS, network_config   # was: from .custom_env.lane_control.dqn_config ...
```

---

## Verification Checklist

1. **Smoke run** — `python train.py -max_total_steps 50000 -min_mem 5000` should complete the replay-buffer fill and a few learning steps; TensorBoard files should appear under `./logs/train/1ramp_1x3/lane_control/`.
2. **GUI inspection** — `python train.py --gui True -max_total_steps 200` and confirm that when the chosen action has `lane_idx == 1`, vehicles visibly slow on `acceleration_area_1` and lane-change leftward.
3. **TraCI sanity** — during a manual rollout, `traci.lane.getMaxSpeed("acceleration_area_1")` should toggle between the default and `5.0` cycle by cycle in step with `info["lane_closed"]`.
4. **Regression** — revert the two imports above and confirm the previous variant still trains.
5. **Full training** — 2.1 M steps. Compare TensorBoard reward curve, mean ramp queue, downstream throughput, and bottleneck occupancy against the prior 8-action variant and the four classical baselines (via `evaluate.py`).

---

## Known Limitations / Future Work

- **Non-Markov in lane state**: the previous lane action is not in the observation. If learning is unstable, append a binary feature to the macro vector (14 → 15) and let the network grow by one input weight.
- **No yellow phase / hard transitions** — inherited from the base project's "Known Issues" list in `CLAUDE.md`.
- **Single-lane VSL only** — extending to multiple lanes or graded speed levels would multiply the action space; consider factoring the head (separate green-time and lane heads) before adding more actuators.
- **No matching baselines** — the four classical baselines do not exercise the VSL. A useful future addition is "ALINEA + always-closed-lane" or "ALINEA + lane closed when downstream occ > O_crit", to isolate the contribution of the lane actuator vs the RL policy.
