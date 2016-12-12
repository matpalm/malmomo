#!/usr/bin/env python

# implementation of prioritised replay as described in
# Prioritized Experience Replay; Schaul, Quan, Antonoglou & Silver
# https://arxiv.org/abs/1511.05952v4

from collections import Counter
import math
import numpy as np
import random

class SumTree(object):
  """ SumTree datastructure for storing priority values in binary heap like structure for fast sampling """

  def __init__(self, num_elements):
    assert num_elements != 0 and ((num_elements & (num_elements - 1)) == 0), "num_elements must be a power of 2"
    self.num_elements = num_elements
    self.tree = np.zeros(self.num_elements * 2)  # tree is rooted at idx 1, not 0
    self.size = 0

  def update(self, idx, value):
#    print "sum_tree.update idx", idx, "value", value
    # note: we don't keep track of size and instead assume caller does.
    self.size = max(self.size, idx+1)
    idx += self.num_elements  # shift into range of lower row
    delta = float(value) - self.tree[idx]
    while idx != 0:
      self.tree[idx] += delta
      idx >>= 1

  def value_at(self, idx):
    return self.tree[idx + self.num_elements]

  def total(self):
    return self.tree[1]

  def value_ntiles(self, n=11):
    current_values = self.tree[self.num_elements : self.num_elements+self.size]
    return np.percentile(current_values, np.linspace(0, 100, n),
                         interpolation='nearest')

  def index_of(self, value):
    if self.total() == 0: raise ValueError("no inserts yet?")
    assert value >= 0
    assert value <= self.total()
    idx = 1
    while True:
      peek_left = self.tree[idx << 1]
      if value >= peek_left:
        # continue down on rhs
        value -= peek_left
        idx = (idx << 1) + 1
      else:
        # continue down on lhs
        idx <<= 1
      # stop when we're in the final row
      if idx >= self.num_elements:
        return idx - self.num_elements

  def sample(self, n):
    # do sampling WITHOUT replacement by explicitly taking samples out of tree and
    # then returning them after all samples are taken. ( tried simpler rejection
    # sampling but wasn't as effective )
#    print "sum_tree.sample"
    samples = []  # indexs and values
    for i in range(n):
      # sample and index and value
      idx = self.index_of(random.random() * self.total())
      value = self.value_at(idx)
      samples.append((idx, value))
      # remove temporarily from tree
      self.update(idx, 0)
    # replace entries in tree
#    print "sum_tree.replace"
    for idx, value in samples:
      self.update(idx, value)
    # return samples
#    print "sum_tree.sample returning", samples
    return samples

  def dump(self, additional_idx_data=None):
    print ">>dump"
    print "total", self.total()
    for idx, value in enumerate(self.tree):
      real_idx = idx - self.num_elements
      if real_idx < 0:
        real_idx = "."
        additional_data = "."
      elif additional_idx_data:
        additional_data = additional_idx_data[real_idx]
      print "\t".join(map(str, [idx, real_idx, additional_data, value]))
    print "<<dump"


class PrioExperienceReplay(object):
  def __init__(self, size, p_epsilon=1.0, p_alpha=0.6, p_beta=0.4, p_max=100.0):
    """ size: sum_tree max size
        p_epsilon: small value to add to every loss. larger values => more uniform sampling
        p_alpha: pow to raise loss to (after adding p_epsilon), 0=>uniform sampling. 1=>linear
        p_beta: importance sampling bias correction rescaling factor.
        p_max: max value p_i can take (after raising by p_alpha)
        use p_alpha & p_beta dfts from paper sec4
    """
    self.sum_tree = SumTree(size)
    self.p_epsilon = p_epsilon
    self.p_alpha = p_alpha
    self.p_beta = p_beta
    self.p_max = p_max
    self.num_times_sampled = Counter()

    # we use a special priority value for storing new experiences, one that is slightly larger
    # than the maximum value. this makes new experiences most likely to be sampled (though
    # not guaranteed) while always allowing us to cleanly identify them during sampling so we can
    # assign an importance reweighting of 1.0 (since we don't know anything about them yet...)
    # the alternate, allowing them to be reweighted down, would penalise them the first time
    # they were trained against, though maybe this is OK since next time they are sampled
    # they'd have proper weights...
    self.p_new_experience = self.p_max + 1

  def new_experience(self, idx):
    """
    add a completely new entry to replay memory. these entries take priority in next
    batch and are given an importance weight of 1.0
    """
#    print "prio_replay.new_experience", idx
    self.sum_tree.update(idx, self.p_new_experience)
    del self.num_times_sampled[idx]

  def sample(self, n):
    """ sample a set of n (idxs, importance_sampling weights) """
    print ">sample"

    # sample idxs and priorities
    idx_prios = self.sum_tree.sample(n)

    # calculate importance sampling weights
    unscaled_weights = []
    print "sum_tree.total()", self.sum_tree.total(), "sum_tree.size()", self.sum_tree.size, "p_beta", self.p_beta
    print "\t".join(["idx", "prio", "p_j", "w_j"])
    for idx, prio in idx_prios:
      # convert priority to prob_j by normalising based on total
      p_j = prio / self.sum_tree.total()
      # convert to importance weight
      w_j = math.pow(self.sum_tree.size * p_j, -self.p_beta)
      print "i_%d\t%f\t%f\t%f" % (idx, prio, p_j, w_j)
      unscaled_weights.append(w_j)
      self.num_times_sampled[idx] += 1

    # zip idxs with unscaled_weights and rescale them so max w_i in the batch has
    # weight 1.0. note: new experiences, based on special prio value, are given
    # an importance weight of 1.0
    weight_scaling_factor = 1.0 / max(unscaled_weights)
    idxs, weights = [], []
    for (idx, prio), unscaled_weight in zip(idx_prios, unscaled_weights):
      idxs.append(idx)
      weights.append(1.0 if prio == self.p_new_experience else unscaled_weight * weight_scaling_factor)

    print "<prio_replay.sample", zip(idxs, weights)
    return idxs, weights

  def update_priorities(self, idxs, losses):
    """ update a set of prios. called after a sampling with new losses """
    for idx, loss in zip(idxs, losses):
#      print "prio_replay.update_priorities idx", idx, "loss", loss
      prio = loss + self.p_epsilon         # always add a small minimum amount. higher => more uniform sampling.
      prio = math.pow(prio, self.p_alpha)  # rescale; 0=>uniform, i.e. ignore scale of p_i; 1=>linear
      prio = min(prio, self.p_max)         # clip at some max value
      self.sum_tree.update(idx, prio)
    print "<prio_replay.update_priorities -> dump"
    self.dump()

  def dump(self):
#    print ">sumtree dump"
#    self.sum_tree.dump(self.num_times_sampled)
    print ">most_common"
    print "idx\tfreq\tvalue"
    most_common = self.num_times_sampled.most_common(200)
    for idx, freq in most_common:
      print "\t".join(map(str, [idx, freq, self.sum_tree.value_at(idx)]))
#    print "value percentiles", map(float, per.sum_tree.value_ntiles(n=11))
