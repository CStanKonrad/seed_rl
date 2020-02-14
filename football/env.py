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
flags.DEFINE_string('env_config', '', 'json with env config')


def create_environment(_, env_logdir=''):
  """Returns a gym Football environment."""
  logging.info('Creating environment: %s', FLAGS.env_config)
  config = json.loads(FLAGS.env_config)
  if env_logdir != '':
    logging.info('Environment asked to provide logs to: %s', env_logdir)
    config['logdir'] = env_logdir
  return observation.PackedBitsObservation(
    gym.make('gfootball_zpp:gfootball-custom-v1', **config))
