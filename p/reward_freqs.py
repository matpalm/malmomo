#!/usr/bin/env python
import json
import sys
import numpy as np

rewards = []
for line in sys.stdin:
  if not line.startswith("REWARD"): continue
  cols = line.strip().split("\t")
  data = json.loads(cols[1])
  rewards.append(data['reward'])

rmin, rmax = np.min(rewards), np.max(rewards)
rrange = rmax - rmin

num_bins = 20
histo = [0] * (num_bins+1)
for r in rewards:
  n = (r-rmin)/rrange
  rbin = int(n*num_bins)
  histo[rbin] += 1

for i, h in enumerate(histo):
  print rmin + (i*(rrange/num_bins)), "\t", h


#grep ^REWARD $1 | cut -f2 | jq '.reward' | sort | uniq -c | normalise.py  | sort -k3 -nr
