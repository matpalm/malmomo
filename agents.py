from collections import *
import models
import numpy as np
import replay_memory
import sys
import tensorflow as tf
import time
import util

def add_opts(parser):
  parser.add_argument('--batch-size', type=int, default=128, help="training batch size")
#  parser.add_argument('--batches-per-step', type=int, default=5,
#                      help="number of batches to train per step")
  parser.add_argument('--replay-memory-size', type=int, default=10000,
                      help="max size of replay memory")
  parser.add_argument('--replay-memory-burn-in', type=int, default=100,
                      help="dont train from replay memory until it reaches this size")

class RandomAgent(object):
  def __init__(self, opts):
    self.stats_ = Counter()

  def action_given(self, state):
    return map(float, (np.random.random(size=2)*2)-1)

  def train(self, state_action_rewards):
    self.stats_['runs'] += 1
    if state_action_rewards[-1][2] != 0:
      self.stats_['was_successful'] += 1

  def stats(self):
    return self.stats_

class NafAgent(object):
  def __init__(self, opts):
    self.opts = opts

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 #opts.gpu_mem_fraction
    self.sess = tf.Session(config=config)

    render_shape = (opts.height, opts.width, 3)
    self.replay_memory = replay_memory.ReplayMemory(buffer_size=opts.replay_memory_size, 
                                                    state_shape=render_shape,
                                                    action_dim=2,
                                                    load_factor=1.2)

    # s1 and s2 placeholders
    batched_state_shape = [None] + list(render_shape)
    s1 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
    s2 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)

    # initialise base models for value & naf networks. value subportion of net is
    # explicitly created seperate because it has a target network note: in the case of
    # --share-input-state-representation the input state network of the value_net will
    # be reused by the naf.l_value and naf.output_actions net
    self.value_net = models.ValueNetwork("value", s1, opts)
    self.target_value_net = models.ValueNetwork("target_value", s2, opts)
    self.network = models.NafNetwork("naf", s1, s2,
                                     self.value_net, self.target_value_net,
                                     action_dim=2, opts=opts)
    
    with self.sess.as_default():
      tf.get_default_session().run(tf.initialize_all_variables())
      # TODO: reinclude saver_util stuff
      for v in tf.all_variables():
        print >>sys.stderr, v.name, util.shape_and_product_of(v)
      # TODO: opt for update rate
      self.target_value_net.set_as_target_network_for(self.value_net, 0.01)

  def action_given(self, state):
    with self.sess.as_default():
      return self.network.action_given(state, add_noise=True)

  def train(self, state_action_rewards):
    start = time.time()
    self.replay_memory.add_episode(state_action_rewards)    
    print "replay_memory.add_episode", time.time()-start

    if self.replay_memory.size() > self.opts.replay_memory_burn_in:
      with self.sess.as_default():
        start = time.time()
        batch = self.replay_memory.batch(self.opts.batch_size)
        print "fetch batch", time.time()-start
        start = time.time()
        self.network.train(batch)
        print "train", time.time()-start
        self.network.target_value_net.update_weights()

  def stats(self):
    return self.replay_memory.stats
