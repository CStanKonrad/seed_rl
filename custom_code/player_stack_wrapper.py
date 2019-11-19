import gym
import numpy as np


class PlayerStackWrapper(gym.Env):
    """Supports only extracted observations (stackedx4)"""

    def __init__(self, env, env_config):
        self.env = env

        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        # observation shape will be (_, _, 4*frame_size)
        self.players = env.observation_space.shape[0]
        self.frame_size = 2 + 1 + 1
        obs_shape = np.array(env.observation_space.shape[1:])
        obs_shape[len(obs_shape) - 1] = self.frame_size * 4

        print('!!!PlayerStackWrapper: Squashed observations to!!!', obs_shape, env.action_space)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = env.action_space

    def _convert_obs(self, observation):
        conv_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        for i in range(conv_obs.shape[2]):
            j = i // self.frame_size
            layer_id = i % self.frame_size

            if layer_id == 0:
                layer = observation[0, ..., j * 4]
            elif layer_id == 1:
                layer = observation[0, ..., j * 4 + 1]
            elif layer_id == 3:
                layer = observation[0, ..., j * 4 + 3]
                for player_id in range(self.players):
                    layer = np.maximum(layer, observation[player_id, ..., j * 4 + 3])
            else:
                layer = observation[0, ..., j * 4 + 2]

            conv_obs[..., i] = layer
            #print(np.where(conv_obs > 0))

        return conv_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self._convert_obs(observation)
        reward = np.max(reward)

        return observation, reward, done, info

    def reset(self):
        print('psw reset')
        observation = self.env.reset()
        observation = self._convert_obs(observation)
        return observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

