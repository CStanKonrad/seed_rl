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


"""VTrace (IMPALA) learner for Google Research Football.
"""

from absl import app
from absl import flags
from absl import logging

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.football import env
from seed_rl.football import networks
from seed_rl.football.networks.gfootball import create_network as GFootball
from seed_rl.football.networks.gfootball_lstm import create_network as GFootballLSTM
from seed_rl.football.networks.gfootball_lite import create_network as GFootballLite
from seed_rl.football.networks.vtrace_mlp_and_lstm import create_network as VtraceMLPandLSTM
from seed_rl.football.networks.gfootball_flex import create_network as GFootballFlex
from seed_rl.football.networks.gfootball_flex20 import create_network as GFootballFlex20
import tensorflow as tf



FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')

KNOWN_NETWORKS = {
  'GFootball': GFootball,
  'GFootballLSTM': GFootballLSTM,
  'GFootballLite': GFootballLite,
  'GFootballFlex': GFootballFlex,
  'GFootballFlex20': GFootballFlex20,
  'VtraceMLPandLSTM': VtraceMLPandLSTM
}


def create_agent(action_space, env_observation_space,
                 parametric_action_distribution, extended_network_config={}):
  network_config = extended_network_config.copy()
  network_config['action_space'] = action_space
  network_config['env_observation_space'] = env_observation_space
  network_config['parametric_action_distribution'] = parametric_action_distribution

  if 'network_name' in network_config:
    network_name = network_config['network_name']
  else:
    network_name = 'GFootball'
    logging.warning('WARNING: NO NETWORK NAME PROVIDED, DEFAULT WILL BE USED')

  logging.info('Creating network %s with parameters: %s', network_name, str(network_config))

  return KNOWN_NETWORKS[network_name](network_config)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(env.create_environment)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(env.create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
