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

# This is a modified version of the original file

r"""SEED actor."""

import os

from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf

from inspect import signature

FLAGS = flags.FLAGS

flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_actors_with_summaries', 1,
                     'Number of actors that will log debug/profiling TF '
                     'summaries.')
flags.DEFINE_bool('render', False,
                  'Whether the first actor should render the environment.')


def are_summaries_enabled():
  return FLAGS.task < FLAGS.num_actors_with_summaries


def is_rendering_enabled():
  return FLAGS.render and FLAGS.task == 0


def actor_loop(create_env_fn):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
  """
  logging.info('Starting actor loop')
  main_logdir = os.path.join(FLAGS.logdir, 'actors', 'actor_{}'.format(FLAGS.task))
  env_logdir = main_logdir
  if are_summaries_enabled():
    summary_writer = tf.summary.create_file_writer(
        main_logdir,
        flush_millis=20000, max_queue=1000)
    timer_cls = profiling.ExportingTimer
  else:
    summary_writer = tf.summary.create_noop_writer()
    timer_cls = utils.nullcontext

  actor_step = 0
  with summary_writer.as_default():
    while True:
      try:
        # Client to communicate with the learner.
        client = grpc.Client(FLAGS.server_address)

        # Checks whenever env can provide additional logs
        create_env_fn_params = signature(create_env_fn).parameters
        if 'env_logdir' in create_env_fn_params and 'actor_id' in create_env_fn_params:
          env = create_env_fn(FLAGS.task, env_logdir=env_logdir, actor_id=FLAGS.task)
        else:
          env = create_env_fn(FLAGS.task)

        # Unique ID to identify a specific run of an actor.
        run_id = np.random.randint(np.iinfo(np.int64).max)
        observation = env.reset()
        zero_reward = utils.get_initial_reward(env)
        reward = tf.identity(zero_reward)
        raw_reward = 0.0
        done = False

        episode_step = 0
        episode_return = tf.identity(zero_reward)
        episode_raw_return = 0

        while True:
          tf.summary.experimental.set_step(actor_step)
          env_output = utils.EnvOutput(reward, done, observation)
          with timer_cls('actor/elapsed_inference_s', 1000):
            action = client.inference(
                FLAGS.task, run_id, env_output, raw_reward)
          with timer_cls('actor/elapsed_env_step_s', 1000):
            observation, reward, done, info = env.step(action.numpy())
          if is_rendering_enabled():
            env.render()
          episode_step += 1
          reward = utils.convert_reward(reward)
          episode_return += reward
          raw_reward = float((info or {}).get('score_reward', reward))
          episode_raw_return += raw_reward
          if done:
            logging.info('Return: %s Raw return: %f Steps: %i', episode_return.numpy(),
                         episode_raw_return, episode_step)
            with timer_cls('actor/elapsed_env_reset_s', 10):
              observation = env.reset()
              episode_step = 0
              episode_return = tf.identity(zero_reward)
              episode_raw_return = 0
            if is_rendering_enabled():
              env.render()
          actor_step += 1
      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        logging.exception(e)
        env.close()
