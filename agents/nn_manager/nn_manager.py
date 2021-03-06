import gym
import tensorflow as tf
import time
from absl import logging
from seed_rl.common.utils import EnvOutput

import collections
from seed_rl.common.parametric_distribution import ParametricDistribution, get_parametric_distribution_for_action_space
import numpy as np
import json
import os
import re

import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


def extract_number_from_txt(txt):
  m = re.search("[0-9]+", txt)
  return int(txt[m.start():m.end()])


def prefix_permute(tensor, prefix_permutation):
  permutation = list(range(len(tensor.shape)))
  prefix_len = len(prefix_permutation)
  permutation[0:prefix_len] = prefix_permutation
  return tf.transpose(tensor, perm=permutation)


def group_tensors(tensor_list, grouping):
  output = []
  index = 0
  for g in grouping:
    output.append(tf.stack(tensor_list[index:(index + g)]))
    index += g
  return output


def group_log_probs(log_probs, grouping):
  output = []
  index = 0
  for g in grouping:
    output.append(tf.reduce_sum(log_probs[..., index:(index + g)], axis=-1))
    index += g
  return tf.stack(output, axis=-1)

# based on ParametricDistribuion from https://github.com/CStanKonrad/seed_rl/blob/current_setup/common/parametric_distribution.py
class NNMDistributionWrapper(ParametricDistribution):
  def __init__(self, original_distribution, action_log_probs_grouping_fn):
    self._original_distribution = original_distribution
    self._action_log_probs_grouping_fn = action_log_probs_grouping_fn
    self._entropy_grouping_fn = action_log_probs_grouping_fn

  def create_dist(self, parameters):
    return self._original_distribution.create_dist(parameters)

  @property
  def param_size(self):
    return self._original_distribution.param_size

  @property
  def reparametrizable(self):
    return self._original_distribution.reparametrizable

  def postprocess(self, event):
    return self._original_distribution.postprocess(event)

  def sample(self, parameters):
    return self._original_distribution.sample(parameters)

  def log_prob(self, parameters, actions):
    """Compute the log probability of actions."""
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)
    log_probs -= self._original_distribution._postprocessor.forward_log_det_jacobian(
      tf.cast(actions, tf.float32), event_ndims=0)

    if self._original_distribution._event_ndims == 1:
      log_probs = self._action_log_probs_grouping_fn(log_probs)

    return log_probs

  def entropy(self, parameters):
    """Return the entropy of the given distribution."""
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    entropy += self._original_distribution._postprocessor.forward_log_det_jacobian(
      tf.cast(dist.sample(), tf.float32), event_ndims=0)
    if self._original_distribution._event_ndims == 1:
      entropy = self._entropy_grouping_fn(entropy)
    return entropy


def split_action_space(action_space, split_data):
  if isinstance(action_space, gym.spaces.Discrete):
    return [action_space]
  elif isinstance(action_space, gym.spaces.MultiDiscrete):
    output = []
    for s in split_data:
      new_as = gym.spaces.MultiDiscrete(action_space.nvec[s[0]:s[1]])
      output.append(new_as)
    return output
  else:
    assert False, 'Unsupported action space'


def get_distributions_for_action_spaces(action_space_list):
  return [get_parametric_distribution_for_action_space(action_space) for action_space in action_space_list]


def decode_string_config(config):
  if isinstance(config, dict):
    return config
  else:
    return json.loads(config)


def support_legacy_config(config):
  config = config.copy()
  network_actions_spec = config['network_actions_spec']

  lenghts = list(map(len, network_actions_spec))
  min_lenght = np.min(lenghts)
  max_length = np.max(lenghts)
  if min_lenght != 2 or max_length != 2 or network_actions_spec[0][0] == network_actions_spec[0][1]:
    new_network_actions_spec = [[0, len(l)] for l in network_actions_spec]
    config['network_actions_spec'] = new_network_actions_spec
  return config


class NNManager(tf.Module):
  def __init__(self, create_agent_fn, env_output_specs, action_space, logdir, save_checkpoint_secs, config,
               observation_space):
    super(NNManager, self).__init__(name=None)

    config = decode_string_config(config)
    config = support_legacy_config(config)

    self._original_observation_space = observation_space

    self._original_action_space = action_space

    # Specifies which network should produce actions for i'th observation.
    # For example [0, 0, 1] means that network number 0 should be used
    # to produce actions for the first and the second observation
    # and network number 1 for the third observation.
    self._observation_to_network_mapping = config['observation_to_network_mapping']

    self._network_action_spec = config['network_actions_spec']
    self._network_action_space = split_action_space(action_space, self._network_action_spec)
    self._network_action_distribution = get_distributions_for_action_spaces(self._network_action_space)
    self._num_networks = len(self._network_action_distribution)

    # Defines if gradients should be computed for i'th network.
    self._network_learning = config['network_learning']

    self._network_config = config['network_config']
    if (len(self._network_config) == 1) and (self._num_networks != 1):
      self._network_config = [self._network_config[0]] * self._num_networks

    self._network = [create_agent_fn(self._network_action_space[i], (), self._network_action_distribution[i],
                                     extended_network_config=self._network_config[i]) for i in
                     range(self._num_networks)]

    # by default iteration number is picked from  optimizer of the first network
    # if the first network is not updated then NNManager handles iterations manually
    handle_iterations_manually = config['handle_iterations_manually'] if 'handle_iterations_manually' in config else False
    self._handle_iterations_manually = (not self._network_learning[0]) or handle_iterations_manually
    self._iterations = tf.Variable(0, dtype=tf.int64)

    self._logdir = logdir
    self._save_checkpoint_secs = save_checkpoint_secs

    self._ckpt = None
    self._manager = None
    self._last_ckpt_time = [0] * self._num_networks
    self._last_manager_save_time = 0

    self._networks_trainable_variables = None

    self._optimizers = None
    self._learning_rate_fn = None

    self._single_agent = len(env_output_specs.reward.shape) == 0

    logging.info('Starting manager', )
    logging.info('NNManager: networks : %s', str(self._network))
    logging.info('NNManager: networks actions specs : %s', str(self._network_action_spec))
    logging.info('NNManager: networks actions spaces : %s', str(self._network_action_space))
    logging.info('NNManager: networks actions distributions : %s', str(self._network_action_distribution))
    logging.info('NNManager: networks lerning : %s', str(self._network_learning))
    logging.info('NNManager: networks configs : %s', str(self._network_config))
    logging.info('NNManager: single agent : %s', str(self._single_agent))


  def get_number_of_agents(self):
    return len(self._observation_to_network_mapping)

  @tf.function
  def get_observation_shape(self):
    return tf.convert_to_tensor(self._original_observation_space.shape)

  def get_action_space_distribution(self):
    original_distribution = get_parametric_distribution_for_action_space(self._original_action_space)

    def action_log_probs_grouping(log_probs):
      if self._single_agent:
        return tf.reduce_sum(log_probs, axis=-1)
      else:
        return group_log_probs(log_probs, self.get_action_groups())

    return NNMDistributionWrapper(original_distribution, action_log_probs_grouping)

  def create_trainable_variables(self):
    self._networks_trainable_variables = []
    for i in range(self._num_networks):
      if self._network_learning[i]:
        self._networks_trainable_variables.extend(self._network[i].trainable_variables)
    self._networks_trainable_variables = tuple(self._networks_trainable_variables)

  def get_networks_trainable_variables(self):
    return self._networks_trainable_variables

  def get_action_groups(self):
    groups = []
    for i in self._observation_to_network_mapping:
      groups.append(self._network_action_spec[i][1] - self._network_action_spec[i][0])
    return groups

  def create_optimizers(self, create_optimizer_fn, final_iteration):
    self._optimizers = []
    self._learning_rate_fn = []

    for i in range(self._num_networks):
      optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)
      optimizer._create_hypers()
      optimizer._create_slots(self._network[i].trainable_variables)

      optimizer.iterations # create optimizer iterations variable

      self._optimizers.append(optimizer)
      self._learning_rate_fn.append(learning_rate_fn)

    if not self._handle_iterations_manually:
      self._iterations = self._optimizers[0].iterations

  def iterations(self):
    return self._iterations

  def optimize(self, temp_grads):
    index = 0
    for i in range(self._num_networks):
      if self._network_learning[i]:
        num_variables = len(self._network[i].trainable_variables)
        self._optimizers[i].apply_gradients(
          zip(temp_grads[index:(index + num_variables)], self._network[i].trainable_variables))
        index += num_variables

  def post_optimize(self):
    if self._handle_iterations_manually:
      self._iterations.assign(self._iterations + 1)

  @tf.function
  def initial_state(self, batch_size):
    result_state = []
    for net_num in self._observation_to_network_mapping:
      result_state.append(self._network[net_num].initial_state(batch_size))
    return result_state

  def make_checkpoints(self):
    self._ckpt = [tf.train.Checkpoint(agent=self._network[i], optimizer=self._optimizers[i]) for i in
                  range(self._num_networks)]
    self._manager = [
      tf.train.CheckpointManager(self._ckpt[i], os.path.join(self._logdir, 'ckpt', str(i)), max_to_keep=1,
                                 keep_checkpoint_every_n_hours=2) for i in range(self._num_networks)]
    current_time = time.time()
    for i in range(self._num_networks):
      self._last_ckpt_time[i] = 0  # Force checkpointing of the initial model.
      if self._manager[i].latest_checkpoint:
        logging.info('Restoring checkpoint: %s for network %i', self._manager[i].latest_checkpoint, i)
        self._ckpt[i].restore(self._manager[i].latest_checkpoint).assert_consumed()
        self._last_ckpt_time[i] = int(current_time)

  def _save_nn_manager(self):
    time_stamp = time.time()

    save_dir = os.path.join(self._logdir, 'model', 'nn_manager')
    tf.io.gfile.makedirs(save_dir)

    file_list = list(map(extract_number_from_txt, tf.io.gfile.listdir(save_dir)))
    file_list.append(-1)
    max_file_number = max(file_list)

    current_file_number = max_file_number + 1
    tf.saved_model.save(self, os.path.join(save_dir, str(current_file_number)))

    self._last_manager_save_time = time_stamp

  def _save_model_data_for_network(self, network_id):
    time_stamp = time.time()
    self._manager[network_id].save()
    model_file = os.path.split(self._manager[network_id].latest_checkpoint)[1][len('ckpt-'):]
    tf.saved_model.save(self._network[network_id], os.path.join(self._logdir, 'model', str(network_id),
      model_file))
    self._last_ckpt_time[network_id] = int(time_stamp)

  def manage_models_data(self):
    current_time = time.time()
    for i in range(self._num_networks):
      if (self._network_learning[i]) and (current_time - self._last_ckpt_time[i] >= self._save_checkpoint_secs):
        self._save_model_data_for_network(i)

    if current_time - self._last_manager_save_time > self._save_checkpoint_secs:
      self._save_nn_manager()

  def save(self):
    for i in range(self._num_networks):
      if self._network_learning[i]:
        self._save_model_data_for_network(i)
    self._save_nn_manager()

  def adjust_discounts(self, discounts):
    if not self._single_agent:
      discounts = tf.expand_dims(discounts, -1)
      rep_m = [1] * len(discounts.shape)
      rep_m[-1] = self.get_number_of_agents()
      discounts = tf.tile(discounts, rep_m)

    return discounts

  def vtrace_adjust_loss(self, loss):
    # total loss was divided by number of agents during tf.reduce_mean ops
    return loss

  def _prepare_input(self, input_, unroll):
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)

    if self._single_agent:
      return [input_]
    else:
      prev_actions, env_outputs = input_

      prev_actions = prefix_permute(prev_actions, [2, 0, 1])
      prev_actions = group_tensors(prev_actions, self.get_action_groups())
      prev_actions = tf.nest.map_structure(lambda t: tf.transpose(t, perm=[1, 2, 0]), prev_actions)

      done = env_outputs.done

      num_observations = env_outputs.observation.shape[2]
      assert num_observations == self.get_number_of_agents()

      input_ = []
      for i in range(num_observations):
        input_.append((prev_actions[i], EnvOutput(env_outputs.reward[:, :, i], done, env_outputs.observation[:, :, i])))


      return input_

  def _prepare_call_output(self, new_action, policy_logits, baseline, unroll):

    new_action = tf.concat(new_action, axis=2)
    policy_logits = tf.concat(policy_logits, axis=2)
    baseline = tf.stack(baseline)
    if self._single_agent:
      baseline = tf.squeeze(baseline, axis=0)
    else:
      baseline = prefix_permute(baseline, [1, 2, 0])

    output = AgentOutput(new_action, policy_logits, baseline)

    if not unroll:
      # Remove time dimension.
      output = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), output)
    return output

  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False, is_training=False,
               postprocess_action=True):
    input_ = self._prepare_input((prev_actions, env_outputs), unroll)

    new_action = []
    policy_logits = []
    baseline = []
    num_agents = self.get_number_of_agents()
    new_core_state = [None] * num_agents
    for i in range(num_agents):
      net_num = self._observation_to_network_mapping[i]

      o, s = self._network[net_num](*input_[i], core_state[i], unroll=True,
                                    is_training=is_training,
                                    postprocess_action=postprocess_action)

      new_core_state[i] = s

      new_action.append(o.action)
      policy_logits.append(o.policy_logits)
      baseline.append(o.baseline)


    output = self._prepare_call_output(new_action, policy_logits, baseline, unroll)

    return output, new_core_state

  def write_summaries(self):
    """ This method should be called with tf summary writer """
    for net_id in range(self._num_networks):
      tf.summary.scalar('nn_manager/network_{}/optimizer/iterations'.format(net_id),
                        self._optimizers[net_id].iterations)
      tf.summary.scalar('nn_manager/network_{}/optimizer/learning_rate'.format(net_id),
                        self._learning_rate_fn[net_id](self._optimizers[net_id].iterations))
