import gymnasium as gym
import ale_py

from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing

class Environments:
    def __init__(self, n_envs=1, render_mode=None):
        self.render_mode = render_mode
        self.envs = [self.new_env() for _ in range(n_envs)]
        self.obs = [None for _ in range(n_envs)]
        self.total_reward = [0 for _ in range(n_envs)]
        self.terminated = [None for _ in range(n_envs)]
        self.truncated = [None for _ in range(n_envs)]
        self.info = [None for _ in range(n_envs)]
        self.done = [None for _ in range(n_envs)]
        self.n_envs = n_envs

        for env_id in range(n_envs):
            self.reset_env(env_id)

    def len(self):
        return self.n_envs

    def reset_env(self, env_id):
        self.total_reward[env_id] = 0
        self.envs[env_id].reset()
        self.terminated[env_id] = None
        self.truncated[env_id] = None
        self.info[env_id] = None
        self.done[env_id] = None

        for _ in range(10):
            self.obs[env_id], reward, _, _, _ = self.envs[env_id].step(1)
            self.total_reward[env_id] += reward

    def step(self, env_id, action):
        obs, reward, terminated, truncated, info = self.envs[env_id].step(action)
        done = terminated or truncated
        self.obs[env_id] = obs
        self.total_reward[env_id] += reward
        self.terminated[env_id] = terminated
        self.truncated[env_id] = truncated
        self.info[env_id] = info

        return obs, reward, terminated, truncated, info, done

    def new_env(self):
        env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=self.render_mode)
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)
        return env
