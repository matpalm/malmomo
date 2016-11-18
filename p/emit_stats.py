#!/usr/bin/env python

import sys, json
import numpy as np

collecting_ep = None
moves = []
turns = []

def emit_stats(filename):
  print "\t".join(map(str, [filename, collecting_ep, np.mean(moves), np.std(moves), 
                            np.mean(turns), np.std(turns)]))

for filename in sys.argv[1:]:
  for line in open(filename, "r"):
    if not line.startswith("ACTION"): continue
    _action, data = line.split("\t")
    data = json.loads(data)
    if not data['eval']: continue
    if data['episode'] != collecting_ep:
      if collecting_ep != None:
        emit_stats(filename)
        move = []
        turns = []
      collecting_ep = data['episode']
    moves.append(data['move'])
    turns.append(data['turn'])
  emit_stats(filename)
