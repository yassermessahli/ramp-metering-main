import gymnasium as gym


def _unpack_step(step_result):
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info
    return step_result


class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        """Return only every `repeat`-th frame"""
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        """Repeat action, sum reward over last observations."""
        total_reward = 0.0
        done = False
        for _i in range(self._repeat):
            obs, reward, done, info = _unpack_step(self.env.step(action))
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MaxEpisodeStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = _unpack_step(self.env.step(ac))
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
