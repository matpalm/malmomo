import ckpt_util
from collections import *
import event_log
import models
import numpy as np
import replay_memory
import sys
import tensorflow as tf
import time
import util

def add_opts(parser):
  parser.add_argument('--gpu-mem-fraction', type=float, default=0.5,
                      help="fraction of gpu mem to allocate")

class RandomAgent(object):
  def __init__(self, opts):
    self.stats_ = Counter()

  def action_given(self, state, is_eval):
    turn = (np.random.random()*2)-1    # (-1,1) for turn
    move = (np.random.random()*2)-0.5  # (-0.5,1.5) for move, i.e. favor moving forward
    return turn, move

  def end_of_episode(self):
    pass


class NafAgent(object):
  def __init__(self, opts):
#    self.opts = opts

    self.network = models.NafNetwork("naf", action_dim=2, opts=opts)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = opts.gpu_mem_fraction
    self.sess = tf.Session(config=config)

    with self.sess.as_default():
      # setup saver to load first set of ckpts. block until some are available
      self.loader = ckpt_util.AgentCkptLoader(self.sess, opts.ckpt_dir)
      self.loader.blocking_load_ckpt()
      # dump info on vars
      for v in tf.all_variables():
        if '/biases:' not in v.name:
          print >>sys.stderr, v.name, util.shape_and_product_of(v)

  def action_given(self, state, is_eval):
    with self.sess.as_default():
      return self.network.action_given(state, add_noise=(not is_eval))

  def end_of_episode(self):
    self.loader.reload_if_new_ckpt()
