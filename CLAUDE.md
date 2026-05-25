# CLAUDE.md — Project Context for AI Assistants

> This file provides essential project context for AI coding assistants working on this codebase.
> It avoids redundancy by summarizing the architecture, conventions, and key details
> so that every session starts with a shared understanding.

---

## Project Overview

**Deep Reinforcement Learning for Automatic Ramp Metering Control** on a highway corridor.

A DRL agent controls a traffic signal (ramp meter) at a highway on-ramp to:
1. Maintain free-flow speed on the mainline at and upstream of the merge zone.
2. Maximize downstream throughput.
3. Limit on-ramp queue length to prevent spillback onto the arterial network.

The agent is trained and evaluated in **SUMO** (Simulation of Urban MObility) via the **TraCI** Python interface.

---

## Technology Stack

| Component         | Technology                      |
| ----------------- | ------------------------------- |
| Language          | Python ≥ 3.12                   |
| Package Manager   | `uv` (pyproject.toml + uv.lock) |
| DL Framework      | PyTorch                         |
| Traffic Simulator | SUMO (via TraCI / sumolib)      |
| RL API            | Gymnasium                       |
| Logging           | TensorBoard                     |
| Serialization     | msgpack (model save/load)       |
| Visualization     | Matplotlib, Seaborn, Jupyter    |

---

## Directory Structure

```
.
├── train.py                 # Training entry point
├── play.py                  # Run baselines or human-play mode
├── observe.py               # Run trained agent for inference/observation
├── evaluate.py              # Batch evaluation pipeline (N episodes, seeded)
├── pyproject.toml           # uv-managed project config
│
├── env/                     # Environment package
│   ├── __init__.py          # Exports DqnEnv, DQN_CONFIG
│   ├── dqn_config.py        # Central hyperparameters + network architecture config
│   ├── dqn_env.py           # DqnEnv — adapter between framework and SUMO env
│   ├── view.py              # Pyglet visualization (disabled by default)
│   └── custom_env/
│       ├── __init__.py      # Exports RLController, baselines, SUMO_PARAMS
│       ├── utils.py         # SUMO_PARAMS — global simulation config dict
│       ├── sumo_env.py      # SumoEnv — base TraCI interface (lifecycle, detectors, routes)
│       ├── rl_controller.py # RLController — RL logic: state, action, reward
│       ├── baselines.py     # Classical baselines: AlwaysGreen, FixedCycle, ALINEA, PI-ALINEA
│       ├── macro no lane/       # Variant 1 configs (8-d state, MLP)
│       ├── macro + lane/        # Variant 2 configs (14-d state, MLP)
│       └── micro + macro lane/  # Variant 3 configs (284-d hybrid, CNN+MLP) ← ACTIVE
│
├── dqn/                     # Agent package
│   ├── __init__.py          # Re-exports all agent classes and env utilities
│   ├── agent.py             # Agent hierarchy: DQN → Double → Dueling → PER variants
│   ├── network.py           # DeepQNetwork, DuelingDeepQNetwork, TwoStreamHybridNetwork
│   ├── replay_memory.py     # ReplayMemoryNaive (uniform), ReplayMemoryPrioritized (SumTree)
│   ├── env_wrap.py          # CustomEnvWrapper — Gymnasium-compatible reset/step
│   ├── env_make.py          # Environment factory with wrappers
│   └── utils/
│       ├── sum_tree.py      # SumTree data structure for PER
│       ├── better_abc.py    # Enhanced ABC metaclass
│       ├── msgpack_numpy.py # Model serialization helpers
│       └── baselines_wrappers/
│           ├── monitor.py         # Episode monitoring
│           ├── dummy_vec_env.py   # Single-process vectorized env wrapper
│           └── subproc_vec_env.py # Multi-process vectorized env wrapper
│
├── evaluation/
│   ├── parsers.py           # Parse tripinfo.xml, SUMO logs, framework CSVs
│   ├── results/             # Per-strategy result CSVs + Jupyter notebooks + plots/
│   └── reward/              # Training reward visualization notebook
│
├── save/                    # Saved model checkpoints (msgpack format)
│   └── 1ramp_1x3/
├── logs/                    # TensorBoard event files
│   └── train/1ramp_1x3/
├── tripinfo_runs/           # SUMO trip info XML outputs
└── bin/
    └── environment.yml      # Legacy Conda environment definition
```

---

## Architecture

### Environment Stack (bottom → top)

```
SumoEnv          ← TraCI lifecycle, detector queries, route generation, vehicle subscriptions
    ↑
RLController     ← State construction (macro + micro grid), action execution (green/red phases), reward
    ↑
DqnEnv           ← Thin adapter matching framework's generic interface
    ↑
CustomEnvWrapper ← Gymnasium-compatible reset()/step() API
    ↑
DummyVecEnv      ← Vectorized env wrapper (always n_env=1)
```

### Agent Hierarchy

```
Agent (abstract)
├── SimpleAgent          → DQNAgent (vanilla DQN)
├── DoubleAgent          → DoubleDQNAgent, DuelingDoubleDQNAgent  ← ACTIVE
└── PerDoubleAgent       → PerDuelingDoubleDQNAgent
```

### Network Architectures

- **DeepQNetwork**: Configurable MLP body → Q-values
- **DuelingDeepQNetwork**: MLP body → value stream + advantage stream → Q = V + (A − mean(A))
- **TwoStreamHybridNetwork** *(active)*: Splits 284-d input → macro (14-d) via FC + micro (270-d reshaped to 2×27×5) via CNN → concatenate → FC layers

---

## SUMO Network

- **File**: `1ramp_1x3.net.xml` — single on-ramp, 3-lane mainline highway
- **Key edges**: `entry`, `off_ramp_up_stream`, `main_road`, `on_ramp`, `passage_area`, `acceleration_area`, `end_main_road`
- **Ramp meter**: Traffic light `ramp_meter` with two phases (G/r)
- **Detectors**: Upstream (3 lanes), bottleneck (4 lanes), downstream (3 lanes), ramp queue — all with 40s collection period

---

## Active Configuration

| Parameter              | Value                               |
| ---------------------- | ----------------------------------- |
| Algorithm              | `DuelingDoubleDQNAgent`             |
| State dimension        | 284 (14 macro + 270 micro grid)     |
| Action space           | 8 discrete (green time: 5–40s, 5s steps) |
| Control cycle          | 40 seconds                          |
| Learning rate          | 1e-4 (Adam)                         |
| Discount (γ)           | 0.99                                |
| Epsilon                | 1.0 → 0.01 (exponential, over 2M steps) |
| Batch size             | 32                                  |
| Replay buffer          | 1M transitions (uniform sampling)  |
| Min buffer before learn| 100K transitions                    |
| Target update          | Soft Polyak (τ = 1e-3) every step   |
| Loss                   | SmoothL1Loss (Huber)                |
| Max training steps     | 2.1M agent steps                    |
| Episode length         | 3600s (1 hour sim time, ~90 steps)  |

---

## Reward Function

Weighted multi-objective: speed rewards (positive) + occupancy/queue penalties (negative).

```
reward = 1.5 × r_speed_merge      (+)
       + 1.0 × r_speed_upstream   (+)
       + 0.5 × r_speed_downstream (+)
       - 2.0 × p_occ_bottleneck   (−)
       - 1.0 × p_occ_upstream     (−)
       - 1.0 × p_queue            (−)
       - 20.0 × p_spillback       (−, triggers at 90% max queue)
```

Theoretical range: **[−24, +3]**.

---

## Entry Points

| Command                          | Description                                    |
| -------------------------------- | ---------------------------------------------- |
| `python train.py`                | Train from scratch                             |
| `python train.py --load True`    | Resume training from checkpoint                |
| `python train.py --gui True`     | Train with SUMO GUI visible                    |
| `python play.py --strategy X`    | Run a baseline strategy (AlwaysGreen, etc.)    |
| `python observe.py --load path`  | Run trained model for inference                |
| `python evaluate.py`             | Batch evaluate over N episodes                 |

---

## Baselines

| Baseline       | Strategy                                                     |
| -------------- | ------------------------------------------------------------ |
| AlwaysGreen    | No metering — ramp signal permanently green                  |
| FixedCycle     | Fixed 20s green / 20s red alternation                        |
| ALINEA         | Proportional feedback on downstream occupancy (O_crit=17%)   |
| PI-ALINEA      | Proportional-integral feedback (K_P=60, K_I=10)              |

---

## Model Persistence

Models are saved as **msgpack** files (not PyTorch `.pt`), containing:
- Network parameters as NumPy arrays
- Training step count, episode count, mean reward, mean episode length

Location: `save/1ramp_1x3/<AgentClass>_lr<lr>/`

---

## Known Issues (from original project)

These are documented weaknesses in the base project:

1. **Penetration rate bug**: In `SumoEnv._generate_route_file()`, nearly all vehicles spawn as connected regardless of the drawn penetration rate. The correct split code is commented out.
2. **Double observation computation**: `RLController.step()` computes observation/reward, then `CustomEnvWrapper` computes them again via separate accessor methods.
3. **Queue measurement**: `get_edge_ls_queue_length_vehicles()` uses `getLastStepVehicleNumber()` (total vehicles) instead of `getLastStepHaltingNumber()` (halting vehicles).
4. **No yellow phase**: Direct green↔red transitions (yellow time defined but unused).
5. **Hard-coded IDs**: Detector and edge IDs are string literals scattered across the code.
6. **No gradient clipping**: None of the `learn()` methods apply gradient clipping.
7. **No validation during training**: No periodic evaluation on held-out scenarios.
8. **No automated tests**: No unit or integration tests exist.
9. **Model variants via file swapping**: Changing architectures requires manually copying files between subdirectories.

---

## Conventions

- **Config management**: Centralized in `env/dqn_config.py` (DQN_CONFIG dict) and `env/custom_env/utils.py` (SUMO_PARAMS dict).
- **Network variants**: Stored as alternative `rl_controller.py` + `dqn_config.py` in subdirectories under `env/custom_env/`. Active config lives at the top level.
- **Logging**: TensorBoard for training metrics → `logs/train/1ramp_1x3/<agent>_lr<lr>/`
- **Evaluation results**: CSVs per strategy → `evaluation/results/results_<StrategyName>.csv`
- **Console styling**: Uses `colorama` for colored terminal output.
- **Episode seeding**: Evaluation uses deterministic seeds (`master_seed + episode_id`) for reproducibility across strategies.

---

## Demand Scenarios

Flows are drawn from weighted discrete distributions each episode:

| Flow Type  | Values (veh/h)                         | Bias              |
| ---------- | -------------------------------------- | ----------------- |
| Mainline   | 4000, 4500, 5000, 5500, 6000, 6500    | Towards higher    |
| On-ramp    | 1400, 1500, 1600, 1700, 1800, 1900, 2000 | Towards higher |
| Off-ramp   | 100, 300, 500                          | Uniform           |

---

## Micro Grid (Connected Vehicle Data)

The hybrid state includes a spatial grid populated by connected vehicle positions:

- **Grid**: 27 rows × 5 columns × 2 channels (speed, presence) = 270 features
- **Cell size**: 8m × 1 lane
- **Coverage**: 216m communication range around merge zone
- **Columns**: 3 mainline lanes + 1 acceleration lane + 1 ramp/passage lane
- **Population**: Via TraCI vehicle subscriptions for `v_type="con"`

---

*This file is intended for AI assistant context. See `README.md` for the full technical analysis.*
