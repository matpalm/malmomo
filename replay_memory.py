import collections
import numpy as np

Batch = collections.namedtuple("Batch", "state_1 action reward terminal_mask state_2")

class ReplayMemory(object):
  def __init__(self, buffer_size, state_shape, action_dim, load_factor=1.5):
    self.buffer_size = buffer_size
    self.state_shape = state_shape
    self.insert = 0
    self.full = False

    # the elements of the replay memory. each event represents a row in the following
    # five matrices. state_[12]_idx index into states list
    self.state_1_idx = np.empty(buffer_size, dtype=np.int32)
    self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
    self.reward = np.empty((buffer_size, 1), dtype=np.float32)
    self.terminal_mask = np.empty((buffer_size, 1), dtype=np.float32)
    self.state_2_idx = np.empty(buffer_size, dtype=np.int32)

    # states themselves, since they can either be state_1 or state_2 in an event
    # are stored in a separate matrix. it is sized fractionally larger than the replay
    # memory since a rollout of length n contains n+1 states.
    self.state_buffer_size = int(buffer_size * load_factor)
    shape = [self.state_buffer_size] + list(state_shape)
    self.state = np.empty(shape, dtype=np.float16)

    # keep track of free slots in state buffer
    self.state_free_slots = list(range(self.state_buffer_size))

    # for the sake of debugging we fill in a zero filled state as first element
    # this is to be used in the case where the final (s1,a,r,s2) with terminal = True
    # means s2, passed through target net, isn't used.
    self.zero_state_idx = self.state_free_slots.pop(0)
    assert self.zero_state_idx == 0
    self.state[self.zero_state_idx] = np.zeros(state_shape)

    # some stats
    self.stats = collections.Counter()

  def reset_from_event_log(self, log_file):
    raise Exception("TODO")

  def add_episode(self, state_action_rewards):
    self.stats['>add_episode'] += 1
    assert len(state_action_rewards) > 0

    for n, (state, action, reward) in enumerate(state_action_rewards):
      # for first element need to explicitly store state
      if n == 0:
        state_1_idx = self.state_free_slots.pop(0)
        self.state[state_1_idx] = state
      # in all but not last element we need to peek for next state
      if n != len(state_action_rewards)-1:
        terminal = False
        state_2_idx = self.state_free_slots.pop(0)
        self.state[state_2_idx] = state_action_rewards[n+1][0]
      else:
        # store special zero state for null state_2 (see init for more info)
        terminal = True
        state_2_idx = self.zero_state_idx
      # add element to replay memory
      self._add(state_1_idx, action, reward, terminal, state_2_idx)
      # roll state_2_idx to become state_1_idx for next step
      state_1_idx = state_2_idx

    self.stats['free_slots'] = len(self.state_free_slots)

  def _add(self, s1_idx, a, r, t, s2_idx):
    print ">add s1_idx=%s, a=%s, r=%s, t=%s s2_idx=%s" % (s1_idx, a, r, t, s2_idx)

    self.stats['>add'] += 1
    assert s1_idx >= 0, s1_idx
    assert s1_idx < self.state_buffer_size, s1_idx
    assert s1_idx not in self.state_free_slots, s1_idx

    if self.full:
      # are are about to overwrite an existing entry.
      # we always free the state_1 slot we are about to clobber...
      self.state_free_slots.append(self.state_1_idx[self.insert])
#      print "full; so free slot", self.state_1_idx[self.insert]


      # NOT ANY MORE! 
      # and we free the state_2 slot also if the slot is a terminal event
      # (since that implies no other event uses this state_2 as a state_1)
#      self.stats['cache_evicted_s1'] += 1
#      if self.terminal_mask[self.insert] == 0:
#        self.state_free_slots.append(self.state_2_idx[self.insert])
#        print "also, since terminal, free", self.state_2_idx[self.insert]
#        self.stats['cache_evicted_s2'] += 1

    # add s1, a, r, s2
    self.state_1_idx[self.insert] = s1_idx
    self.action[self.insert] = a
    self.reward[self.insert] = r
    self.state_2_idx[self.insert] = s2_idx
    # if terminal we set terminal mask to 0.0 representing the masking of the righthand
    # side of the bellman equation
    self.terminal_mask[self.insert] = 0.0 if t else 1.0

    # move insert ptr forward
    self.insert += 1
    if self.insert >= self.buffer_size:
      self.insert = 0
      self.full = True

  def random_indexes(self, n=1):
    if self.full:
      return np.random.randint(0, self.buffer_size, n)
    elif self.insert == 0:  # empty
      return []
    else:
      return np.random.randint(0, self.insert, n)

  def batch(self, batch_size=None):
    self.stats['>batch'] += 1
    idxs = self.random_indexes(batch_size)
    return Batch(np.copy(self.state[self.state_1_idx[idxs]]),
                 np.copy(self.action[idxs]),
                 np.copy(self.reward[idxs]),
                 np.copy(self.terminal_mask[idxs]),
                 np.copy(self.state[self.state_2_idx[idxs]]))




