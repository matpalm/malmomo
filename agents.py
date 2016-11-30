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
#  parser.add_argument('--batch-size', type=int, default=128, help="training batch size")
#  parser.add_argument('--batches-per-step', type=int, default=5,
#                      help="number of batches to train per step")
#  parser.add_argument('--event-log-in', type=str, default=None,
#                      help="if set replay these event files into replay memory (comma"
#                           " separated list")
#  parser.add_argument('--event-log-in-num', type=int, default=None,
#                      help="if set only read this many events from event-logs-in")
#  parser.add_argument('--dont-store-new-memories', action='store_true',
#                      help="if set do not store new memories.")
#  parser.add_argument('--ckpt-reload-freq', type=int, default=30,
#                      help="freq (sec) to reload ckpts from trainer")
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
    self.opts = opts

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = opts.gpu_mem_fraction
    self.sess = tf.Session(config=config)

#    render_shape = (opts.height, opts.width, 3)
#    self.replay_memory = replay_memory.ReplayMemory(opts=opts,
#                                                    state_shape=render_shape,
#                                                    action_dim=2,
#                                                    load_factor=1.1)
#    if opts.event_log_in:
#      self.replay_memory.reset_from_event_logs(opts.event_log_in,
#                                               opts.event_log_in_num)

    self.network = models.NafNetwork("naf", action_dim=2, opts=opts)

    with self.sess.as_default():
      # setup saver to load first set of ckpts. block until some are available
      self.loader = ckpt_util.AgentCkptLoader(self.sess, opts.ckpt_dir)
      self.loader.blocking_load_ckpt()
      # dump info on vars
      for v in tf.all_variables():
        if '/biases:' not in v.name:
          print >>sys.stderr, v.name, util.shape_and_product_of(v)
      # setup target network
      #self.network.post_init_setup()  NOT FOR EVAL ROLLOUTS

  def action_given(self, state, is_eval):
    with self.sess.as_default():
      return self.network.action_given(state, add_noise=(not is_eval))

  def end_of_episode(self):
    self.loader.reload_if_new_ckpt()

#  def add_episode(self, episode):
#    # add to replay memory
#    if not self.opts.dont_store_new_memories:
#      self.replay_memory.add_episode(episode)
#
#    # do some number of training steps
#    if self.replay_memory.burnt_in():
#      with self.sess.as_default():
#        losses = []
#        for _ in xrange(self.opts.batches_per_step):
#          batch = self.replay_memory.batch(self.opts.batch_size)
#          loss = self.network.train(batch)
#          losses.append(loss)
#          self.network.target_value_net.update_weights()
#        print "losses\t" + "\t".join(map(str, np.percentile(losses, np.linspace(0, 100, 11))))
#      if self.saver_util is not None:
#        self.saver_util.save_if_required()

#  def stats(self):
#    return self.replay_memory.stats
