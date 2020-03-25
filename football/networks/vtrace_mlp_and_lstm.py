from seed_rl.agents.vtrace.networks import MLPandLSTM

def create_network(network_config):
  net = MLPandLSTM(parametric_action_distribution=network_config['parametric_action_distribution'],
                   mlp_sizes=network_config['mlp_sizes'],
                   lstm_sizes=network_config['lstm_sizes'])
  return net
