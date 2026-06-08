"""Smooth inspection of a trained model for demonstrations.

Reuses observe.py wholesale (model loading, network, loop, stats, logging) but runs SUMO at
a fine step-length (default 0.1s) so vehicle motion is smooth enough to show a jury. The
observation pipeline is step-length invariant (queue and speed fixes in the controller), so a
model trained at 1.0s behaves faithfully here.

Run:
    uv run python diagnostics/observe_smooth.py -d save/1ramp_1x3/<variant>/<model>.pack -seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import Observe


class ObserveSmooth(Observe):
    """Observe, but resilient across multiple episodes."""

    def setup(self):
        # CustomEnvWrapper.reset() returns (obs, info) — unpack it. observe.py omits this,
        # which crashes on the 2nd episode; fixed here so multi-episode demos work.
        self.obs, _info = self.env.reset()
        self.ep_green_times = []
        self.ep_lane_closed_count = 0
        self.ep_rewards = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSERVE SMOOTH")

    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser.add_argument("-d", type=str, default="", help="Directory", required=True)
    parser.add_argument("-gpu", type=str, default="0", help="GPU #")
    parser.add_argument(
        "-seed", type=int, default=-1, help="SUMO random seed (-1 = random)"
    )
    parser.add_argument(
        "-step_length",
        type=float,
        default=0.1,
        help="SUMO step-length for smooth playback (overrides the sumocfg)",
    )
    parser.add_argument(
        "-max_s", type=int, default=0, help="Max steps per episode if > 0, else inf"
    )
    parser.add_argument(
        "-max_e", type=int, default=0, help="Max episodes if > 0, else inf"
    )
    parser.add_argument(
        "-log", type=str2bool, default=False, help="Log csv to ./logs/test/"
    )
    parser.add_argument(
        "-log_s", type=int, default=0, help="Log step if > 0, else episode"
    )
    parser.add_argument(
        "-log_dir", type=str, default="./logs/test/", help="Log directory"
    )

    args = parser.parse_args()

    # Read by sumo_env.set_params() before SUMO starts → injects --step-length.
    os.environ["SUMO_STEP_LENGTH"] = str(args.step_length)

    ObserveSmooth(args).run()
