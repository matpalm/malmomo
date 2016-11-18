#!/usr/bin/env python
import sys, json
import numpy as np
for filename in sys.argv[1:]:
  for line in open(filename, "r"):
    if not line.startswith("REWARD"): continue
    _reward, data = line.split("\t")
    data = json.loads(data)
    if not data['eval']: continue    
    print "\t".join(map(str, [filename, data['episode'], data['reward']]))
