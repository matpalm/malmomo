#!/usr/bin/env python
import argparse
import ckpt_util
from multiprocessing import Process, Queue
from concurrent import futures
import json
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
parser.add_argument('--batches-per-new-episode', type=int, default=5,
                    help="number of batches to train per new episode")
parser.add_argument('--event-log-in', type=str, default=None,
                    help="if set replay these event files into replay memory (comma"
                         " separated list")
parser.add_argument('--event-log-in-num', type=int, default=None,
                    help="if set only read this many events from event-logs-in")
parser.add_argument('--gpu-mem-fraction', type=float, default=0.5,
                    help="fraction of gpu mem to allocate")

# TODO: event_log_in and in_num should move to run_replayer.py

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
  # TODO! MOVE TO RUN_REPLAYER
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
    network.setup_target_network()

    # while true process episodes from run_agents
    while True:
      start_time = time.time()
      episode = episodes.get()
      wait_time = time.time() - start_time

      start_time = time.time()
      replay_memory.add_episode(episode)
      losses = []
      if replay_memory.burnt_in():
        for _ in xrange(opts.batches_per_new_episode):
          batch = replay_memory.batch(opts.batch_size)
          loss = network.train(batch)
          network.target_value_net.update_target_weights()
          losses.append(float(loss))
        saver.save_if_required()
      process_time = time.time() - start_time

      print "STATS\t%s\t%s" % (util.dts(), json.dumps({"wait_time": wait_time,
                                                       "process_time": process_time,
                                                       "pending": episodes.qsize(),
                                                       "losses": sorted(losses),
                                                       "replay_memory": replay_memory.stats}))

if __name__ == '__main__':
  queued_episodes = Queue(5)

  enqueue_process = Process(target=run_enqueue_server, args=(queued_episodes,))
  enqueue_process.daemon = True
  enqueue_process.start()

  trainer_process = Process(target=run_trainer, args=(queued_episodes, opts))
  trainer_process.daemon = True
  trainer_process.start()

  while True:
    time.sleep(2)
