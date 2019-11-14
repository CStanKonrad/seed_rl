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

"""Common losses."""

import tensorflow as tf


def baseline(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))


def entropy(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def policy_gradient(logits, actions, advantages):
  if len(logits.shape) != 4:
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
  else:
    logits = tf.transpose(logits, perm=[2, 0, 1, 3])
    actions = tf.transpose(actions, perm=[2, 0, 1])
    results = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits[i], labels=actions[i]) for i in ange(actions.shape[0])]
    cross_entropy = tf.reduce_mean(results)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)
