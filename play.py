import argparse

from dqn import CustomEnvWrapper, make_env

from env import CustomEnv, View


class Play(View):
    """Interactive playback of environment."""

    def __init__(self, args):
        """Initializes playback environment."""
        super().__init__(
            type(self).__name__.upper(),
            make_env(
                env=CustomEnvWrapper(
                    CustomEnv(type(self).__name__.lower(), p=args.player)
                ),
                max_episode_steps=args.max_s,
            ),
        )

        self.ep = 0

        print()
        print("PLAY")
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_episodes = args.max_e
        self.log = (args.log, args.log_s, args.log_dir + args.player)

    def setup(self):
        """Resets environment for new episode."""
        _ = self.env.reset()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def loop(self):
        """Execute single step in playback loop."""
        # Retrieve action from human input
        action = self.get_play_action()

        # Execute action in environment
        _, _, terminated, truncated, info = self.env.step(action)
        # Check for episode termination
        done = terminated or truncated

        # Log episode statistics
        self.env.log_info_writer(info, done, *self.log)

        # to fix additional episode at the end
        if done:
            self.ep += 1

            print()
            print("Episode :", self.ep)
            # You can still print the info from the episode that just finished
            [print(k, ":", info[k]) for k in info]

            # Check if we should exit BEFORE resetting the environment
            if bool(self.max_episodes) and self.ep >= self.max_episodes:
                # The environment (and SUMO) from the completed run is still open.
                # We can now safely close it and exit the script.
                self.env.close()  # Gracefully close the TraCI connection
                exit()

            # If we are NOT exiting, only then do we reset for the next episode.
            self.setup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLAY")
    str2bool = lambda v: v.lower() in ("yes", "y", "true", "t", "1")
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
    parser.add_argument("-player", type=str, default="player", help="Player")

    Play(parser.parse_args()).run()
