#!/usr/bin/env python
import argparse
import sys, json
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dirs', nargs="+",
                    help="space separated list of run dirs to process")
parser.add_argument('--max-episode', type=int, default=None,
                    help="if set exit when hitting this episode num")
parser.add_argument('--emit', type=str, default=None,
                    help="what to emit  {turn_moves,losses,rewards}")
opts = parser.parse_args()

class TurnMoveStats(object):
  def __init__(self, run_dir):
    self.collecting_ep = None
    self.moves = []
    self.turns = []
    self.filename = run_dir + "/eval.out"

  def emit_stats(self):
    print "\t".join(map(str, [self.filename, self.collecting_ep,
                              np.mean(self.moves), np.std(self.moves),
                              np.mean(self.turns), np.std(self.turns)]))
    self.move = []
    self.turns = []

  def process(self, line):
    if not line.startswith("ACTION"): return
    _action, data = line.split("\t")
    data = json.loads(data)
    if not data['eval']: return
    if data['episode'] != self.collecting_ep:
      if self.collecting_ep is not None:
        self.emit_stats()
      self.collecting_ep = data['episode']
    self.moves.append(data['move'])
    self.turns.append(data['turn'])
    return int(data['episode'])

  def end_of_file(self):
    self.emit_stats()


class LossStats(object):
  def __init__(self, run_dir):
    self.n = 0
    self.filename = run_dir + "/trainer.out"

  def process(self, line):
    if not line.startswith("STATS"): return
    _stats, _dts, data = line.strip().split("\t")
    data = json.loads(data)
    if 'losses' in data:
      mean_loss = np.mean(data['losses'])  # old version
    elif 'loss' in data:
      mean_loss = np.mean(data['loss']['mean'])
    else:
      # burning in still
      return
    mean_loss = min(mean_loss, 1e3)
    print "\t".join([self.filename, str(self.n), str(mean_loss)])
    self.n += 1  # TODO: use data['batches_trained']
    return self.n - 1  # TODO: this isnt actually the episode...

  def end_of_file(self):
    pass


class RewardStats(object):
  def __init__(self, run_dir):
    self.filename = run_dir + "/eval.out"

  def process(self, line):
    if not line.startswith("REWARD"): return
    _reward, data = line.split("\t")
    data = json.loads(data)
    if not data['eval']: return
    print "\t".join(map(str, [self.filename, data['episode'], data['reward']]))
    return int(data['episode'])

  def end_of_file(self):
    pass

def new_processor(run_dir):
  if opts.emit == "turn_moves":
    return TurnMoveStats(run_dir)
  elif opts.emit == "losses":
    return LossStats(run_dir)
  elif opts.emit == "rewards":
    return RewardStats(run_dir)
  else:
    raise Exception("unknown --emit")

for run_dir in opts.dirs:
  processor = new_processor(run_dir)
  for line in open(processor.filename, "r"):
    episode_id = processor.process(line)
    if opts.max_episode is not None and episode_id > opts.max_episode:
      break
  processor.end_of_file()
