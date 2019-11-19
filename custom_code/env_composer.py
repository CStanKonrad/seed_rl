from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import observation_preprocessing
from gfootball.env import wrappers

from seed_rl.custom_code.action_wrappers import ActionOrder
from seed_rl.custom_code.player_stack_wrapper import PlayerStackWrapper
from seed_rl.custom_code.old_wrapper import WrapperEnv
import collections
import gym
import numpy as np

class FrameStack(gym.Wrapper):
  """Stack k last observations."""

  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.obs = collections.deque([], maxlen=k)
    low = env.observation_space.low
    high = env.observation_space.high
    low = np.concatenate([low] * k, axis=-1)
    high = np.concatenate([high] * k, axis=-1)
    self.observation_space = gym.spaces.Box(
        low=low, high=high, dtype=env.observation_space.dtype)

  def reset(self):
    observation = self.env.reset()
    self.obs.extend([observation] * self.obs.maxlen)
    return self._get_observation()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.obs.append(observation)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return np.concatenate(list(self.obs), axis=-1)

def dump_wrapper(env, config):
  if config['dump_frequency'] > 1:
    return wrappers.PeriodicDumpWriter(env, config['dump_frequency'])
  else:
    return env

def checkpoint_wrapper(env, config):
  assert 'scoring' in config['rewards'].split(',')
  if 'checkpoints' in config['rewards'].split(','):
    return wrappers.CheckpointRewardWrapper(env)
  else:
    return env

def single_agent_wrapper(env, config):
  if (config['number_of_left_players_agent_controls'] +
      config['number_of_right_players_agent_controls'] == 1):
     env = wrappers.SingleAgentObservationWrapper(env)
     env = wrappers.SingleAgentRewardWrapper(env)
     return env
  else:
    return env

KNOWN_WRAPPERS = {
  'periodic_dump': dump_wrapper,
  'checkpoint_score': checkpoint_wrapper,
  'single_agent': single_agent_wrapper,
  'obs_extract': lambda env, config: wrappers.SMMWrapper(env, config['channel_dimensions']),
  'obs_stack': lambda env, config: FrameStack(env, config['stacked_frames']),
  'action_order': ActionOrder,
  'psw': PlayerStackWrapper,
  'old_sw': WrapperEnv
}

def compose_environment(env_config, wrappers):
  def extract_from_dict(dictionary, keys):
    return {new_k: dictionary[k] for (new_k, k) in keys}

  players = [('agent:left_players=%d,right_players=%d' % (
    env_config['number_of_left_players_agent_controls'],
    env_config['number_of_right_players_agent_controls']))]
  if env_config['extra_players'] is not None:
    players.extend(env_config['extra_players'])
  env_config['players'] = players
  c = config.Config(extract_from_dict(env_config,
                             [('enable_sides_swap', 'enable_sides_swap'),
                              ('dump_full_episodes', 'enable_full_episode_videos'),
                              ('dump_scores', 'enable_goal_videos'),
                              ('level', 'env_name'),
                              ('players', 'players'),
                              ('render', 'render'),
                              ('tracesdir', 'logdir'),
                              ('write_video', 'write_video')]))
  env = football_env.FootballEnv(c)

  for w in wrappers:
    env = w(env, env_config)

  return env

def config_compose_environment(config):
  wrappers = []
  for w in config['wrappers'].split(','):
    wrappers.append(KNOWN_WRAPPERS[w])

  return compose_environment(config, wrappers)

# def sample_composed_environment():
#   return compose_environment(DEFAULT_EXTENDED_CONFIG, [
#     dump_wrapper,
#     checkpoint_wrapper,
#     ActionOrder,
#     lambda env, config: wrappers.SMMWrapper(env, config['channel_dimensions']),
#     single_agent_wrapper,
#     lambda env, config: FrameStack(env, config['stacked_frames']),
#     PlayerStackWrapper],
#   )
