#!/usr/bin/env python
import argparse
import ckpt_util
from multiprocessing import Process, Queue
from concurrent import futures
import grpc
import models
import model_pb2
import replay_memory as rm
import sys
import tensorflow as tf
import time
import numpy as np
import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', type=int, default=160, help="render width")
parser.add_argument('--height', type=int, default=120, help="render height")
parser.add_argument('--batch-size', type=int, default=128, help="training batch size")
parser.add_argument('--batches-per-step', type=int, default=5,
                    help="number of batches to train per step")
parser.add_argument('--event-log-in', type=str, default=None,
                    help="if set replay these event files into replay memory (comma"
                         " separated list")
parser.add_argument('--event-log-in-num', type=int, default=None,
                    help="if set only read this many events from event-logs-in")
#parser.add_argument('--ckpt-dir', type=str, default=None,
#                    help="if set save ckpts to this dir")
#parser.add_argument('--ckpt-save-freq', type=int, default=60,
#                    help="freq (sec) to save ckpts to agents to reload from")
parser.add_argument('--gpu-mem-fraction', type=float, default=0.5,
                    help="fraction of gpu mem to allocate")

rm.add_opts(parser)
ckpt_util.add_opts(parser)
models.add_opts(parser)
util.add_opts(parser)
opts = parser.parse_args()
print >>sys.stderr, "OPTS", opts


class EnqueueServer(model_pb2.ModelServicer):
  """ Enqueues calls to new episode."""
  def __init__(self, q):
    self.q = q
  def AddEpisode(self, request, context):
    self.q.put(request)
    return model_pb2.Empty()

def run_enqueue_server(episodes):
  grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
  model_pb2.add_ModelServicer_to_server(EnqueueServer(episodes), grpc_server)
  grpc_server.add_insecure_port('[::]:20045')
  print ">start_episode_queuer"
  grpc_server.start()
  while True:
    time.sleep(10)

def run_trainer(episodes, opts):
  # init replay memory
  render_shape = (opts.height, opts.width, 3)
  replay_memory = rm.ReplayMemory(opts=opts,
                                  state_shape=render_shape,
                                  action_dim=2,
                                  load_factor=1.1)
  if opts.event_log_in:
    replay_memory.reset_from_event_logs(opts.event_log_in,
                                        opts.event_log_in_num)

  # init network for training
  config = tf.ConfigProto()
  #config.gpu_options.allow_growth = True
  #config.log_device_placement = True
  config.gpu_options.per_process_gpu_memory_fraction = opts.gpu_mem_fraction
  sess = tf.Session(config=config)

  network = models.NafNetwork("naf", action_dim=2, opts=opts)

  with sess.as_default():
    # setup saver util and either load saved ckpt or init variables
    saver = ckpt_util.TrainerCkptSaver(sess, opts.ckpt_dir, opts.ckpt_save_freq)
    for v in tf.all_variables():
      if '/biases:' not in v.name:
        print >>sys.stderr, v.name, util.shape_and_product_of(v)
    # while true process episodes from run_agents
    while True:
      print util.dts(), ">waiting for episode"
      episode = episodes.get()
      print util.dts(), ">processing episode"
      replay_memory.add_episode(episode)
      print replay_memory.stats
      if replay_memory.burnt_in():
        losses = []
        for _ in xrange(opts.batches_per_step):
          print "BATCH"
          batch = replay_memory.batch(opts.batch_size)
          network.train(batch)
          losses.append(loss)
          network.target_value_net.update_weights()
          print "losses\t" + "\t".join(map(str, np.percentile(losses,
                                                              np.linspace(0, 100, 11))))
        saver.save_if_required()

if __name__ == '__main__':
  queued_episodes = Queue(10)

  enqueue_process = Process(target=run_enqueue_server, args=(queued_episodes,))
  enqueue_process.daemon = True
  enqueue_process.start()

  trainer_process = Process(target=run_trainer, args=(queued_episodes, opts))
  trainer_process.daemon = True
  trainer_process.start()

  while True:
    print util.dts(), "pending", queued_episodes.qsize()
    time.sleep(1)
