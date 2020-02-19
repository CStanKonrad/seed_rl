import gym
import tensorflow as tf
import time
from absl import logging
from seed_rl.common.utils import EnvOutput
# from seed_rl.football import observation as observation_processor
import collections
from seed_rl.common.parametric_distribution import ParametricDistribution, get_parametric_distribution_for_action_space
import numpy as np
import json

import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


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
    # logging.info('Log probs ungrouped %s', str(log_probs))
    if self._original_distribution._event_ndims == 1:
      log_probs = self._action_log_probs_grouping_fn(log_probs)
    # logging.info('Log probs grouped %s', str(log_probs))
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


class NNManager():
  def __init__(self, create_agent_fn, env_output_specs, action_space, logdir, save_checkpoint_secs, config,
               observation_space=()):
    config = decode_string_config(config)
    config = support_legacy_config(config)

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

    self._network = [create_agent_fn(self._network_action_space[i], (), self._network_action_distribution[i]) for i in
                     range(self._num_networks)]  # todo change () to proper observation_space priority: low

    for i in range(self._num_networks):
      if hasattr(self._network[i], 'change_config'):
        self._network[i].change_config(self._network_config[i])

    # by default iteration number is picked from  optimizer of the first network
    # if the first network is not updated then NNManager handles iterations manually
    self._handle_iterations_manually = not self._network_learning[0]
    self._iterations = tf.Variable(0, dtype=tf.int64)

    self._logdir = logdir
    self._save_checkpoint_secs = save_checkpoint_secs

    self._ckpt = None
    self._manager = None
    self._last_ckpt_time = [0] * self._num_networks

    self.trainable_variables = None

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

  def get_action_space_distribution(self):
    original_distribution = get_parametric_distribution_for_action_space(self._original_action_space)

    def action_log_probs_grouping(log_probs):
      if self._single_agent:
        return tf.reduce_sum(log_probs, axis=-1)
      else:
        return group_log_probs(log_probs, self.get_action_groups())

    return NNMDistributionWrapper(original_distribution, action_log_probs_grouping)

  def create_trainable_variables(self):
    self.trainable_variables = []
    for i in range(self._num_networks):
      if self._network_learning[i]:
        self.trainable_variables.extend(self._network[i].trainable_variables)
    self.trainable_variables = tuple(self.trainable_variables)

  def get_action_groups(self):  # todo asserts
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

  def initial_state(self, batch_size):  # todo check
    result_state = []
    for net_num in self._observation_to_network_mapping:
      result_state.append(self._network[net_num].initial_state(batch_size))
    return result_state

  def make_checkpoints(self):
    self._ckpt = [tf.train.Checkpoint(agent=self._network[i], optimizer=self._optimizers[i]) for i in
                  range(self._num_networks)]
    self._manager = [tf.train.CheckpointManager(self._ckpt[i], self._logdir + f"/ckpt/{i}", max_to_keep=1,
                                                keep_checkpoint_every_n_hours=2) for i in range(self._num_networks)]
    current_time = time.time()
    for i in range(self._num_networks):
      self._last_ckpt_time[i] = 0  # Force checkpointing of the initial model.
      if self._manager[i].latest_checkpoint:
        logging.info('Restoring checkpoint: %s for network %i', self._manager[i].latest_checkpoint, i)
        self._ckpt[i].restore(self._manager[i].latest_checkpoint)  # .assert_consumed()
        self._last_ckpt_time[i] = int(current_time)

  def _save_checkpoints_for(self, network_id):
    time_stamp = time.time()
    self._manager[network_id].save()
    self._last_ckpt_time[network_id] = int(time_stamp)

  def manage_checkpoints(self):
    current_time = time.time()
    for i in range(self._num_networks):
      if (self._network_learning[i]) and (current_time - self._last_ckpt_time[i] >= self._save_checkpoint_secs):
        self._save_checkpoints_for(i)

  def save_checkpoints(self):
    for i in range(self._num_networks):
      if self._network_learning[i]:
        self._save_checkpoints_for(i)

  def adjust_discounts(self, discounts):
    # logging.info('discounts before %s', str(discounts))
    if not self._single_agent:
      discounts = tf.expand_dims(discounts, -1)
      rep_m = [1] * len(discounts.shape)
      rep_m[-1] = len(self._observation_to_network_mapping)
      discounts = tf.tile(discounts, rep_m)

    # logging.info('discounts after %s', str(discounts))
    return discounts

  def vtrace_adjust_loss(self, loss):
    # total loss was divided by number of agents during tf.reduce_mean ops
    return loss

  def _prepare_input(self, input_, unroll):
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)

    # logging.info('Preparing input %s', str(input_))
    if self._single_agent:
      return [input_]
    else:
      prev_actions, env_outputs = input_

      # logging.info('Called with prev_action before mangle %s', str(prev_actions))
      prev_actions = prefix_permute(prev_actions, [2, 0, 1])
      prev_actions = group_tensors(prev_actions, self.get_action_groups())
      prev_actions = tf.nest.map_structure(lambda t: tf.transpose(t, perm=[1, 2, 0]), prev_actions)
      # logging.info('Called with prev_action after mangle %s', str(prev_actions))

      def prepare_observation(observation):
        return prefix_permute(observation, [2, 0, 1])

      permuted_observation = tf.xla.experimental.compile(prepare_observation, [env_outputs.observation])[
        0] if tf.test.is_gpu_available() else prepare_observation(env_outputs.observation)

      permuted_reward = prefix_permute(env_outputs.reward, [2, 0, 1])

      done = env_outputs.done

      num_observations = permuted_observation.shape[0]
      assert num_observations == len(self._observation_to_network_mapping)

      input_ = []
      for i in range(num_observations):
        input_.append((prev_actions[i], EnvOutput(permuted_reward[i], done, permuted_observation[i])))

      # logging.info('Processed input %s', str(input_))

      return input_

  def _prepare_call_output(self, new_action, policy_logits, baseline, unroll):

    new_action = tf.concat(new_action, axis=0)
    policy_logits = tf.concat(policy_logits, axis=0)
    baseline = tf.stack(baseline)
    if self._single_agent:
      baseline = tf.squeeze(baseline, axis=0)
    else:
      new_action = prefix_permute(new_action, [1, 2, 0])
      policy_logits = prefix_permute(policy_logits, [1, 2, 0])
      baseline = prefix_permute(baseline, [1, 2, 0])

    output = AgentOutput(new_action, policy_logits, baseline)

    # logging.info('Ends with after mangle output %s', str(output))

    if not unroll:
      # Remove time dimension.
      output = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), output)
    return output

  def __call__(self, input_, core_state, unroll=False, is_training=False):
    input_ = self._prepare_input(input_, unroll)

    new_action = []
    policy_logits = []
    baseline = []
    num_agents = self.get_number_of_agents()
    new_core_state = [None] * num_agents
    for i in range(num_agents):
      net_num = self._observation_to_network_mapping[i]
      # logging.info('for %d net %d', i, net_num)

      o, s = self._network[net_num](input_[i], core_state[i], unroll=True,
                                    is_training=is_training)
      # logging.info('o %s', str(o))
      new_core_state[i] = s

      new_action.append(prefix_permute(o.action, [2, 0, 1]))
      policy_logits.append(prefix_permute(o.policy_logits, [2, 0, 1]))  # todo think
      baseline.append(o.baseline)

    # logging.info('Ends with before mangle new_actions %s', str(new_action))
    # logging.info('Ends with before mangle policy_logits %s', str(policy_logits))

    output = self._prepare_call_output(new_action, policy_logits, baseline, unroll)

    return output, new_core_state
