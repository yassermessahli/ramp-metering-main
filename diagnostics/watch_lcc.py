"""Constant-action GUI runner for visually verifying the LCC mechanism.

Runs the *current active variant* (whatever env/custom_env/__init__.py exports) in the
SUMO GUI with no trained model. A trivial constant policy feeds a hardcoded action every
cycle — default action 8 = 5s green, lane_idx=1 (lane closed) — so you can watch whether
mainline vehicles divert out of the controlled lane before the merge.

Smooth motion comes from <step-length value="0.1"/> in the sumocfg (revert before training).

Run:
    uv run python diagnostics/watch_lcc.py -action 8 -seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dqn import CustomEnvWrapper, make_env
from env import CustomEnv, View


class ConstantPolicy:
    """Drop-in replacement for the trained network: always returns the same action."""

    def __init__(self, action):
        self.action = int(action)

    def actions(self, obs_batch):
        # Mirrors network.actions() interface: takes a batch, returns a list of actions.
        return [self.action for _ in obs_batch]


class WatchLCC(View):
    """Visualizes the LCC mechanism under a fixed, lane-closing action."""

    def __init__(self, args):
        # Set SUMO seed before SUMO starts (read by sumo_env.set_params())
        if args.seed >= 0:
            os.environ["SUMO_EVAL_SEED"] = str(args.seed)
        elif "SUMO_EVAL_SEED" in os.environ:
            del os.environ["SUMO_EVAL_SEED"]

        # Policy must exist before run()/loop(); it needs no env, just stores the action.
        self.policy = ConstantPolicy(args.action)

        super().__init__(
            type(self).__name__.upper(),
            make_env(
                env=CustomEnvWrapper(CustomEnv("observe", gui_override=True)),
                max_episode_steps=args.max_s,
            ),
        )

        self.obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        self.ep = 0
        self.max_episodes = args.max_e

        print()
        print("WATCH LCC")
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]
        print()

    def setup(self):
        """Resets environment for a new episode."""
        # CustomEnvWrapper.reset() returns (obs, info) — unpack it (observe.py omits this).
        self.obs, _info = self.env.reset()
        self.ep_lane_closed_count = 0

    def close(self):
        """Closes the environment."""
        self.env.close()

    def loop(self):
        """Applies the constant action and reports per-cycle lane state."""
        action = self.policy.actions([self.obs.tolist()])[0]

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if info.get("lane_closed", 0) == 1:
            self.ep_lane_closed_count += 1

        print(
            f"t={info.get('sim_time', 0):>7.1f}s  "
            f"green={info.get('chosen_green_time_sec', 0):>4.1f}s  "
            f"lane_closed={info.get('lane_closed', 0)}  "
            f"reward={reward:>6.2f}"
        )

        if done:
            self.ep += 1
            print(
                f"\nEpisode {self.ep} done — lane closed {self.ep_lane_closed_count} cycles\n"
            )

            if bool(self.max_episodes) and self.ep >= self.max_episodes:
                self.env.close()
                exit()

            self.setup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WATCH LCC")

    parser.add_argument(
        "-action",
        type=int,
        default=8,
        help="Constant action index (8-15 close the lane)",
    )
    parser.add_argument(
        "-seed", type=int, default=-1, help="SUMO random seed (-1 = random)"
    )
    parser.add_argument(
        "-max_e", type=int, default=1, help="Episodes before exit (0 = infinite)"
    )
    parser.add_argument(
        "-max_s", type=int, default=0, help="Max steps per episode if > 0, else inf"
    )

    WatchLCC(parser.parse_args()).run()
