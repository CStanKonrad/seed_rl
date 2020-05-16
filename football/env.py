# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Football env factory."""

from absl import flags
from absl import logging

import gym
from seed_rl.football import observation

import json

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string('env_config',
                    '',
                    'json with env config')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')

flags.DEFINE_string('observation_compressor', 'Auto',
                    "Function/Wrapper used for compressing observation"
                    "note that network should decompress observation")


def auto_deduce_compression(env):
  rank = len(env.observation_space.shape)
  if rank == 2:
    logging.info('BEWARE: auto_deduce_compression: Id (None)')
    return env
  elif rank == 4:
    logging.info('BEWARE: auto_deduce_compression: PacketBitsObservation')
    return observation.PackedBitsObservation(env)
  else:
    raise Exception('Cannot deduce compresor from env observation space:' + \
                    str(env.observation_space))

KNOWN_OBSERVATION_COMPRESSORS = {
  'PackedBitsObservation': observation.PackedBitsObservation,
  'Id': lambda x: x,
  'Auto': auto_deduce_compression
}


def create_environment(_, env_logdir='', actor_id=None):
  """Returns a gym Football environment."""
  logging.info('Creating environment: %s', FLAGS.env_config)
  config = json.loads(FLAGS.env_config)
  if env_logdir != '' and actor_id is not None:
    logging.info('Environment will get base_logdir: %s and actor_id %i', env_logdir, actor_id)
    config['base_logdir'] = env_logdir
    config['actor_id'] = actor_id
  else:
    config['base_logdir'] = None
    config['actor_id'] = None

  compresor = KNOWN_OBSERVATION_COMPRESSORS[FLAGS.observation_compressor]
  env = gym.make('gfootball_zpp:gfootball-custom-v1', **config)
  env = compresor(env)
  return env
