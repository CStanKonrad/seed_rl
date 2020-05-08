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

"""SEED agent using Keras."""

from seed_rl.football import observation
import tensorflow as tf
from seed_rl.common import utils
import numpy as np

from .base_vtrace_network import AgentOutput, BaseVTraceNetwork


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same',
                                        kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(
      pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
      tf.keras.layers.Conv2D(
        num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i,
        kernel_initializer='lecun_normal')
      for i in range(num_blocks)
    ]
    self._res_convs1 = [
      tf.keras.layers.Conv2D(
        num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i,
        kernel_initializer='lecun_normal')
      for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


def make_logits(layer_fn, action_specs):
  return [layer_fn(n, 'policy_logits') for n in action_specs]


def make_heads(action_specs, heads_specs):
  def make_head(action_spec, head_spec):
    if len(head_spec['conv_sizes']) != 0:
      conv_stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in head_spec['conv_sizes']
      ]
    else:
      conv_stacks = None

    mlp_layers = [tf.keras.layers.Dense(
      size, "relu", kernel_initializer="lecun_normal") for size in head_spec['mlp_sizes']]

    action_layer = tf.keras.layers.Dense(action_spec, name='policy_logits', kernel_initializer='lecun_normal')

    seq_layers = tf.keras.Sequential(mlp_layers + [action_layer])

    def apply_head(data):
      if conv_stacks is not None:
        for stack in conv_stacks:
          data = stack(data)

        data = tf.nn.relu(data)
        data = tf.keras.layers.Flatten()(data)

      return seq_layers(data)
    
    return apply_head

  result = []
  if len(heads_specs) == 1:
    heads_specs = heads_specs * len(action_specs)
  for action_s, head_s in zip(action_specs, heads_specs):
    print("making head:")
    print(action_s, ":", head_s)
    result.append(make_head(action_s, head_s))
  return result

def apply_net(action_specs, policy_logits, core_output):
  n_actions = len(action_specs)
  arr = [policy_logits[i](core_output) for i in range(n_actions)]
  arr = tf.stack(arr)
  arr = tf.transpose(arr, perm=[1, 0, 2])
  return arr


def post_process_logits(action_specs, policy_logits):
  all_logits = np.sum(action_specs)
  new_shape = policy_logits.shape[:-2] + [all_logits]
  return tf.reshape(policy_logits, new_shape)


def choose_action(action_specs, policy_logits, sample=True):
  n_actions = len(action_specs)
  policy_logits = tf.transpose(policy_logits, perm=[1, 0, 2])

  if not sample:
    new_action = tf.stack([
      tf.math.argmax(
        policy_logits[i], -1, output_type=tf.int64) for i in range(n_actions)])
  else:
    new_action = tf.stack([tf.squeeze(
      tf.random.categorical(
        policy_logits[i], 1, dtype=tf.int64), 1) for i in range(n_actions)])

  new_action = tf.transpose(new_action, perm=[1, 0])
  return new_action


def make_conv_to_linear(conv_to_linear):
  if len(conv_to_linear) != 0:
    return tf.keras.Sequential([tf.keras.layers.Dense(size, 'relu', kernel_initializer='lecun_normal') for size in conv_to_linear])
  else:
    return None

class GFootball(BaseVTraceNetwork):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """

  def __init__(self, action_specs, mlp_sizes, lstm_sizes, heads_specs, baseline_specs, sample_actions, conv_to_linear):
    super(GFootball, self).__init__(name='gfootball')

    self._config = {'sample_actions': sample_actions}

    # Parameters and layers for unroll.

    self._action_specs = action_specs

    self._separate_baseline = baseline_specs['separate']

    self._baseline_grad_stop = baseline_specs['grad_stop']

    # Parameters and layers for _torso.
    self._stacks = [
      _Stack(num_ch, num_blocks)
      for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2), (32, 2)]
    ]

    self._self_conv_to_linear = make_conv_to_linear(conv_to_linear)

    self._core = tf.keras.layers.StackedRNNCells(
      [tf.keras.layers.LSTMCell(size) for size in lstm_sizes])
    self._mlp_after_lstm = tf.keras.Sequential([tf.keras.layers.Dense(
      size, "relu", kernel_initializer="lecun_normal") for size in mlp_sizes])

    # Layers for _head.
    self._policy_logits = make_heads(self._action_specs, heads_specs)

    self._baseline = tf.keras.Sequential(
      [tf.keras.layers.Dense(size, "relu", kernel_initializer="lecun_normal") for size in baseline_specs['mlp_sizes']] + \
      [tf.keras.layers.Dense(1, name='baseline', kernel_initializer='lecun_normal')]
    )

    if self._separate_baseline:
      self._baseline_stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2), (32, 2)]
      ]
      self._baseline_conv_to_linear = make_conv_to_linear(conv_to_linear)


  @tf.function
  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _torso(self, unused_prev_action, env_output):
    _, _, frame = env_output

    frame = observation.unpackbits(frame)
    frame /= 255

    conv_out = frame

    for stack in self._stacks:
      conv_out = stack(conv_out)

    if self._conv_to_linear:
      conv_out = tf.nn.relu(conv_out)
      conv_out = tf.keras.layers.Flatten()(conv_out)

      conv_out = self._conv_to_linear(conv_out)

    if self._separate_baseline:
      conv_out_baseline = frame

      for stack in self._baseline_stacks:
        conv_out_baseline = stack(conv_out_baseline)

      if self._baseline_conv_to_linear:
        conv_out_baseline = tf.nn.relu(conv_out_baseline)
        conv_out_baseline = tf.keras.layers.Flatten()(conv_out_baseline)

        conv_out_baseline = self._baseline_conv_to_linear(conv_out_baseline)
    else:
      conv_out_baseline = conv_out

    return conv_out, conv_out_baseline

  def _head(self, core_output_tuple):
    core_output_policy, core_output_baseline = core_output_tuple
    policy_logits = apply_net(
      self._action_specs,
      self._policy_logits,
      core_output_policy)

    if self._baseline_grad_stop:
      baseline = tf.squeeze(self._baseline(tf.stop_gradient(core_output_baseline)), axis=-1)
    else:
      baseline = tf.squeeze(self._baseline(core_output_baseline), axis=-1)

    # Sample an action from the policy.
    new_action = choose_action(self._action_specs, policy_logits, self._config['sample_actions'])

    return AgentOutput(new_action, post_process_logits(self._action_specs, policy_logits), baseline)

  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll,
               is_training, postprocess_action):
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    _, done, frame = env_outputs
    torso_outputs_policy, torso_outputs_baseline = utils.batch_apply(
      self._torso, (prev_actions, env_outputs))

    initial_core_state = self._core.get_initial_state(
      batch_size=tf.shape(torso_outputs_policy)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs_policy), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
        lambda x, y, d=d: tf.where(
          tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
        initial_core_state,
        core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output = self._mlp_after_lstm(core_output)
      core_output_list.append(core_output)
    outputs = tf.stack(core_output_list)
    if self._separate_baseline:
      outputs = (outputs, torso_outputs_baseline)
    else:
      outputs = (outputs, outputs)
    return utils.batch_apply(self._head, (outputs,)), core_state


def create_network(network_config):
  net = GFootball(network_config['action_space'].nvec,
                  mlp_sizes=network_config['mlp_sizes'],
                  lstm_sizes=network_config['lstm_sizes'],
                  heads_specs=network_config['heads_specs'],
                  baseline_specs=network_config['baseline_specs'],
                  sample_actions=network_config['sample_actions'],
                  conv_to_linear=network_config['conv_to_linear'])
  return net
