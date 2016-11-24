#!/usr/bin/env python
import sys, json
import numpy as np
for filename in sys.argv[1:]:
  n = 0
  for line in open(filename, "r"):
    if not line.startswith("losses"): continue
    losses = line.strip().split("\t")
    losses.pop(0)
    print "\t".join([filename, str(n), losses[5]])
    n += 1
#
#    _reward, data = line.split("\t")
#    data = json.loads(data)
#    if not data['eval']: continue
#    print "\t".join(map(str, [filename, data['episode'], data['reward']]))
