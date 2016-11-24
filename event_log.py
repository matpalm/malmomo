#!/usr/bin/env python
import gzip
import matplotlib.pyplot as plt
import model_pb2
import numpy as np
import StringIO
import struct
import util

class EventLog(object):

  def __init__(self, path):
    self.log_file = open(path, "ab")
    self.episode_entry = None

  def add_episode(self, episode):
    for event in episode.event:
      util.ensure_render_is_png_encoded(event.render)
    buff = episode.SerializeToString()
    buff_len = struct.pack('=l', len(buff))
    self.log_file.write(buff_len)
    self.log_file.write(buff)
    self.log_file.flush()


class EventLogReader(object):

  def __init__(self, path):
    if path.endswith(".gz"):
      self.log_file = gzip.open(path, "rb")
    else:
      self.log_file = open(path, "rb")

  def entries(self):
    episode = model_pb2.Episode()
    while True:
      buff_len_bytes = self.log_file.read(4)
      if len(buff_len_bytes) == 0: return
      buff_len = struct.unpack('=l', buff_len_bytes)[0]
      buff = self.log_file.read(buff_len)
      episode.ParseFromString(buff)
      yield episode


if __name__ == "__main__":
  import argparse, os, sys, Image, ImageDraw
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--file', type=str, default=None)
  parser.add_argument('--echo', action='store_true', help="write event to stdout")
  parser.add_argument('--episodes', type=str, default=None,
                      help="if set only process these specific episodes (comma separated list)")
  parser.add_argument('--nth', type=int, default=None,
                      help="if set emit every nth episode")
  parser.add_argument('--img-output-dir', type=str, default=None,
                      help="if set output all renders to this DIR/e_NUM/s_NUM.png")
  opts = parser.parse_args()

  episode_whitelist = None
  if opts.episodes is not None:
    episode_whitelist = set(map(int, opts.episodes.split(",")))
  if opts.img_output_dir is not None:
    util.make_dir(opts.img_output_dir)

  total_num_read_episodes = 0
  total_num_read_events = 0

  elr = EventLogReader(opts.file)
  for episode_id, episode in enumerate(elr.entries()):
    if opts.nth is not None and episode_id % opts.nth != 0:
      continue
    if episode_whitelist is not None:
      if episode_id not in episode_whitelist:
        continue
      episode_whitelist.discard(episode_id)
    if opts.echo:
      print "-----", episode_id
      print episode
    total_num_read_episodes += 1
    total_num_read_events += len(episode.event)
    if opts.img_output_dir is not None:
      dir = "%s/ep_%05d" % (opts.img_output_dir, episode_id)
      util.make_dir(dir)
      print "writing to", dir
      for event_id, event in enumerate(episode.event):
        assert event.render.is_png_encoded, "only expect serialised pngs"
        img = Image.open(StringIO.StringIO(event.render.bytes))
#        img = img.resize((200, 200))
        filename = "%s/e%04d.png" % (dir, event_id)
        img.save(filename)
    if episode_whitelist is not None and len(episode_whitelist) == 0:
      break
  print >>sys.stderr, "read", total_num_read_episodes, "episodes for a total of", total_num_read_events, "events"
