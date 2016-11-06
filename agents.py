import numpy as np
import replay_memory

class RandomAgent(object):

  def action_given_state(self, state):
    return map(float, (np.random.random(size=2)*2)-1)

  def train(self, state_action_rewards):
#    for _s, a, r in state_action_rewards:
#      print "a", a, "r", r
    pass


class NafAgent(object):
  def __init__(self, opts):
    render_shape = (opts.height, opts.width, 3)
    self.replay_memory = replay_memory.ReplayMemory(buffer_size=20, 
                                                    state_shape=render_shape,
                                                    action_dim=2,
                                                    load_factor=1.2)

  def action_given_state(self, state):
    return map(float, (np.random.random(size=2)*2)-1)

  def train(self, state_action_rewards):
    self.replay_memory.add_episode(state_action_rewards)    
    print self.replay_memory.batch(16)

