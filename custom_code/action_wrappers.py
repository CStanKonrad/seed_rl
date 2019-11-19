import numpy as np
import gym

def encoded_player_position(observation):
  team = 'left_team' if observation['is_left'] == True else 'right_team'
  player_id = observation['active']
  player_position = observation[team][player_id]
  return np.floor(player_position[0] * 1000.0) + player_position[1]

class ActionOrder(gym.Wrapper):
  def __init__(self, env, env_config, piority_fn=encoded_player_position):
    gym.Wrapper.__init__(self, env)
    self._last_observation = None
    self._piority_fn = piority_fn

  def permute(self, action):
    action = np.array(action)
    permutation = np.argsort(list(map(self._piority_fn, self._last_observation)))
    permuted_action = np.zeros(action.shape, dtype=action.dtype)
    permuted_action[permutation] = action[...]
    # print('got:', action, permutation, 'returns:', permuted_action)
    return permuted_action

  def step(self, action):
    observation, reward, done, info = self.env.step(self.permute(action))
    self._last_observation = observation
    return observation, reward, done, info

  def reset(self):
    observation = self.env.reset()
    self._last_observation = observation
    return observation
