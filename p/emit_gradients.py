#!/usr/bin/env python
import sys, re
import numpy as np
from collections import Counter

grad_re = re.compile("^I tensorflow/core/kernels/logging_ops.cc:79\] gradient (.*)\/(.*):0 l2_norm \[(.*)\]")

num_emits = Counter()
for line in sys.stdin:
  m = grad_re.match(line)
  if m:
    name, clazz, norm = m.groups()
    assert clazz in ['weights', 'biases']
    print "\t".join([name, clazz, str(num_emits[name]), norm])
    num_emits[name] += 1
