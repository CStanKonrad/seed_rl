import gym
import tensorflow as tf
import time
from absl import logging
from seed_rl.common.utils import EnvOutput
import collections


AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')

def _prefix_permute(tensor, prefix_permutation):
  permutation = list(range(len(tensor.shape)))
  prefix_len = len(prefix_permutation)
  permutation[0:prefix_len] = prefix_permutation
  return tf.transpose(tensor, perm=permutation)

class NNManager():
  def __init__(self, create_agent_fn, env_output_specs, action_space, logdir, save_checkpoint_secs, config):
    self._num_networks = len(config['networks_actions'])
    self._networks_actions = config['networks_actions']
    self._mapper_function = lambda x: x  #eval(config['mapper_function'])
    self._logdir = logdir
    self._save_checkpoint_secs = save_checkpoint_secs

    self._network = [create_agent_fn(env_output_specs, self._networks_actions[i]) for i in range(self._num_networks)]
    self._ckpt = None
    self._manager = None
    self._last_ckpt_time = [0] * self._num_networks
    #self._inference_device = inference_device

    self.trainable_variables = None
    self._optimizers = None
    self._learning_rate_fn = None
    self._temp_grads = None

    logging.info('Starting manager', )
    logging.info('NNManager: networks : %s', str(self._network))
    logging.info('NNManager: networks actions : %s', str(self._networks_actions))


  def create_optimizers(self, create_optimizer_fn, final_iteration):
    self._optimizers = []
    self._learning_rate_fn = []
    self._temp_grads = []
    for i in range(self._num_networks):
      optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)
      optimizer._create_hypers()
      optimizer._create_slots(self._network[i].trainable_variables)

      temp_grad = [
        tf.Variable(tf.zeros_like(v), trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ)
        for v in self._network[i].trainable_variables
      ]
      self._optimizers.append(optimizer)
      self._learning_rate_fn.append(learning_rate_fn)
      self._temp_grads.append(temp_grad)

  def iterations(self):
    return self._optimizers[0].iterations

  def optimize(self, unroll_specs, decode, data, compute_loss, training_strategy, strategy):
    def compute_gradients(args):
      args = tf.nest.pack_sequence_as(unroll_specs, decode(args, data))
      with tf.GradientTape() as tape:
        loss = compute_loss(self, *args)

      for i in range(self._num_networks):
        grads = tape.gradient(loss, self._network[i].trainable_variables)
        for t, g in zip(self._temp_grads[i], grads):
          t.assign(g)
      return loss

    loss = training_strategy.experimental_run_v2(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      for i in range(self._num_networks):
        self._optimizer[i].apply_gradients(zip(self._temp_grads[i], self._network[i].trainable_variables))

    strategy.experimental_run_v2(apply_gradients, (loss,))


  def create_variables(self):
    self.trainable_variables = []
    for i in range(self._num_networks):
      for var in self._network[i].trainable_variables:
        self.trainable_variables.append(var)
    self.trainable_variables = tuple(self.trainable_variables)

  def initial_state(self, batch_size):
    return ()

  def make_checkpoints(self):
    self._ckpt = [tf.train.Checkpoint(agent=self._network[i], optimizer=self._optimizers[i]) for i in range(self._num_networks)]
    self._manager = [tf.train.CheckpointManager(self._ckpt[i], self._logdir + f"/cpkt/{i}", max_to_keep=1,
                                                keep_checkpoint_every_n_hours=6) for i in range(self._num_networks)]
    for i in range(self._num_networks):
      self._last_ckpt_time[i] = 0  # Force checkpointing of the initial model.
      if self._manager[i].latest_checkpoint:
        logging.info('Restoring checkpoint: %s', self._manager[i].latest_checkpoint)
        self._ckpt[i].restore(self._manager[i].latest_checkpoint).assert_consumed()
        self._last_ckpt_time[i] = time.time()

  def manage_checkpoints(self):
    current_time = time.time()
    for i in range(self._num_networks):
      if current_time - self._last_ckpt_time[i] >= self._save_checkpoint_secs:
        self._manager[i].save()
        self._last_ckpt_time[i] = current_time



  def __call__(self, input_, core_state, unroll=False, inference=False):
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)

    logging.info('Called with input %s', str(input_))
    prev_actions, env_outputs = input_

    logging.info('Called with prev_action before t %s', str(prev_actions))
    prev_actions = _prefix_permute(prev_actions, [2, 0, 1])
    logging.info('Called with prev_action after t %s', str(prev_actions))

    logging.info('Called with env_out before t %s', str(env_outputs))
    permuted_observation = _prefix_permute(env_outputs.observation, [2, 0, 1])
    permuted_reward = _prefix_permute(env_outputs.reward, [2, 0, 1])
    done = env_outputs.done
    logging.info('Called with env_out after t %s', str(permuted_observation))

    num_players = permuted_observation.shape[0]

    input_ = []
    for i in range(num_players):
      input_.append((prev_actions[i], EnvOutput(permuted_reward[i], done, permuted_observation[i])))

    logging.info('Processed input %s', str(input_))

    new_action = []
    policy_logits = []
    baseline = []
    for i in range(num_players):
      net_num = self._mapper_function(i)
      logging.info('for %d net %d', i, net_num)
      o, s = self._network[net_num](input_[i], core_state, True)#todo change
      logging.info('o %s', str(o))
      new_action.append(o.action)
      policy_logits.append(o.policy_logits)
      baseline.append(o.baseline)

    new_action = tf.stack(new_action)
    policy_logits = tf.stack(policy_logits)
    baseline = tf.stack(baseline)

    new_action = _prefix_permute(new_action, [1, 2, 0])
    policy_logits = _prefix_permute(policy_logits, [1, 2, 0])
    baseline = _prefix_permute(baseline, [1, 2, 0])


    outputs = AgentOutput(new_action, policy_logits, baseline)


    logging.info('Ends with after %s', str(outputs))

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state
