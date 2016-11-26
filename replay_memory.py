import collections
import event_log
import numpy as np
import sys
import time
import util

Batch = collections.namedtuple("Batch", "state_1 action reward terminal_mask state_2")

def add_opts(parser):
  parser.add_argument('--replay-memory-size', type=int, default=10000,
                      help="max size of replay memory")
  parser.add_argument('--replay-memory-burn-in', type=int, default=100,
                      help="dont train from replay memory until it reaches this size")
  parser.add_argument('--smooth-reward-factor', type=float, default=0.5,
                      help="if set use this value to smooth rewards")

class ReplayMemory(object):
  def __init__(self, opts, state_shape, action_dim, load_factor=1.5):
    self.buffer_size = opts.replay_memory_size
    self.burn_in = opts.replay_memory_burn_in
    self.state_shape = state_shape
    self.smooth_reward_factor = opts.smooth_reward_factor
    self.insert = 0
    self.full = False

    # the elements of the replay memory. each event represents a row in the following
    # five matrices. state_[12]_idx index into states list
    self.state_1_idx = np.empty(self.buffer_size, dtype=np.int32)
    self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
    self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
    self.terminal_mask = np.empty((self.buffer_size, 1), dtype=np.float32)
    self.state_2_idx = np.empty(self.buffer_size, dtype=np.int32)

    # states themselves, since they can either be state_1 or state_2 in an event
    # are stored in a separate matrix. it is sized fractionally larger than the replay
    # memory since a rollout of length n contains n+1 states.
    self.state_buffer_size = int(self.buffer_size * load_factor)
    shape = [self.state_buffer_size] + list(state_shape)
    self.state = np.empty(shape, dtype=np.uint8)  # (0, 255)

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

  def reset_from_event_logs(self, log_files, max_to_restore):
    num_episodes = 0
    num_events = 0
    start = time.time()
    for log_file in log_files.split(","):
      print "restoring from [%s]. num_episodes=%s num_events=%s" \
        % (log_file, num_episodes, num_events)
      elr = event_log.EventLogReader(log_file.strip())
      for episode in elr.entries():
        if num_episodes%100 == 0: print "...reset_from_event_log restored", num_episodes, self.stats
        if max_to_restore is not None and self.size() > max_to_restore: break
        num_episodes += 1
        num_events += len(episode.event)
        self.add_episode(episode)
        if self.full: break
    print "reset_from_event_log took", time.time()-start, "sec"\
          " num_episodes", num_episodes, "num_events", num_events
    sys.stdout.flush()

  def add_episode(self, episode):
    # potentially smooth rewards. this only currently works for the case of
    # only having a non-zero reward as last element..
    rewards = [e.reward for e in episode.event]
    if self.smooth_reward_factor > 0:
      rewards = util.smooth(rewards, self.smooth_reward_factor)

    self.stats['>add_episode'] += 1
    for n, event in enumerate(episode.event):
      if n == 0:
        # for first event need to explicitly store state
        state_1_idx = self.state_free_slots.pop(0)
        self.state[state_1_idx] = util.rgb_from_render(event.render)
      if n != len(episode.event)-1:
        # in all but last event we need to peek for next state
        terminal = False
        state_2_idx = self.state_free_slots.pop(0)
        self.state[state_2_idx] = util.rgb_from_render(episode.event[n+1].render)
      else:
        # for last event store zero state for null state_2 (see init for more info)
        terminal = True
        state_2_idx = self.zero_state_idx
      # add element to replay memory
      self._add(state_1_idx, event.action.value, rewards[n], terminal, state_2_idx)
      # roll state_2_idx to become state_1_idx for next step
      state_1_idx = state_2_idx

    self.stats['free_slots'] = len(self.state_free_slots)
    self.stats['size'] = self.size()

  def _add(self, s1_idx, a, r, t, s2_idx):
#    print ">add s1_idx=%s, a=%s, r=%s, t=%s s2_idx=%s" % (s1_idx, a, r, t, s2_idx)

    self.stats['>add'] += 1
    assert s1_idx >= 0, s1_idx
    assert s1_idx < self.state_buffer_size, s1_idx
    assert s1_idx not in self.state_free_slots, s1_idx

    if self.full:
      # are are about to overwrite an existing entry.
      # we always free the state_1 slot we are about to clobber...
      self.state_free_slots.append(self.state_1_idx[self.insert])
#      print "full; so free slot", self.state_1_idx[self.insert]

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

  def size(self):
    return self.buffer_size if self.full else self.insert

  def burnt_in(self):
    return self.size() > self.burn_in

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
    # TODO: add tensor.reverse for x flipping when processing 50% time
    return Batch(np.copy(self.state[self.state_1_idx[idxs]]),
                 np.copy(self.action[idxs]),
                 np.copy(self.reward[idxs]),
                 np.copy(self.terminal_mask[idxs]),
                 np.copy(self.state[self.state_2_idx[idxs]]))

  def dump_replay_memory_images_to_disk(self, directory):
    print ">dump_replay_memory_images_to_disk", directory
    util.make_dir(directory)
    for idx in range(self.state_buffer_size):
      if idx == 0:
        pass  # dummy zero state
      elif idx in self.state_free_slots:
        print "idx", idx, "in free slots; ignore"
      else:
        with open("%s/%05d.png" % (directory, idx), "wb") as f:
          f.write(util.rgb_to_png_bytes(self.state[idx]))
