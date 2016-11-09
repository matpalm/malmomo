#!/usr/bin/env python
import agents
import argparse
import event_log
import itertools
import json
import MalmoPython
import models
import model_pb2
import numpy as np
import os
from PIL import Image
import replay_memory
import sys
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

# TODO: no problem with slim import now so push all opts into module where used
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=int, default=None, help="output data to runs/N")
parser.add_argument('--width', type=int, default=160, help="render width")
parser.add_argument('--height', type=int, default=120, help="render height")
parser.add_argument('--episode-time-ms', type=int, default=10000,
                    help="episode timeout (ms)")
parser.add_argument('--agent', type=str, default="Naf", help="{Naf,Random}")
parser.add_argument('--event-log-out', type=str, default=None,
                    help="if set agent also write all episodes to this file")

agents.add_opts(parser)
models.add_opts(parser)
util.add_opts(parser)
replay_memory.add_opts(parser)
opts = parser.parse_args()
print >>sys.stderr, "OPTS", opts

# set up out malmo client
malmo = MalmoPython.AgentHost()
spec = open("classroom_basic.xml").read()
spec = spec.replace("__WIDTH__", str(opts.width))
spec = spec.replace("__HEIGHT__", str(opts.height))
spec = spec.replace("__EPISODE_TIME_MS__", str(opts.episode_time_ms))
mission = MalmoPython.MissionSpec(spec, True)
mission_record = MalmoPython.MissionRecordSpec()

# init our rl_agent
agent_cstr = eval("agents.%sAgent" % opts.agent)
agent = agent_cstr(opts)

event_log = event_log.EventLog(opts.event_log_out) if opts.event_log_out else None

for episode_idx in itertools.count(1):
  print >>sys.stderr, "EPISODE", episode_idx, util.dts()
#  episode_dir = "%s/e_%06d/" % (output_dir, episode)
#  u.make_dir(episode_dir)

  # start new mission; explicitly wait for first observation 
  # (not just world_state.has_mission_begun)
  mission_start = time.time()
  while True:
    try:
      malmo.startMission(mission, mission_record)
      break
    except RuntimeError as r:
      print >>sys.stderr, "failed to start mission", r
      time.sleep(1)
  world_state = malmo.getWorldState()
  while len(world_state.observations) == 0:
    print >>sys.stderr, "started, but no obs?"
    time.sleep(0.1)
    world_state = malmo.getWorldState()
  print "START_TIME", time.time()-mission_start

  # run until the mission has ended
  episode = model_pb2.Episode()
  while world_state.is_mission_running:
    # extract render and convert to numpy array (w, h, 3) with values scaled 0.0 -> 1.0
    if len(world_state.video_frames) == 0:
      print >>sys.stderr, "no vid frames? at step", len(episode.event)
      time.sleep(0.1)
      continue

    event = episode.event.add()

    frame = world_state.video_frames[0]
    # TODO: should be able to do this conversion directly, i.e. not via Image
    img = Image.frombytes('RGB', (frame.width, frame.height), str(frame.pixels))
    img = np.array(img, dtype=np.float16) / 255
    event.render.width = frame.width
    event.render.height = frame.height
    event.render.bytes = img.tostring()
    event.render.is_png_encoded = False

    # decide action given state and send to malmo
    # TODO: change to take model_pb2.Render directly and return model_pb2.Action
    turn, move = agent.action_given(img)
    print "ACTION\t%s" % json.dumps({"episode": episode_idx, "step": len(episode.event),
                                    "turn": turn, "move": move})
    malmo.sendCommand("turn %f" % turn)
    malmo.sendCommand("move %f" % move)
    event.action.value.extend([turn, move])

    # note: for now reward is always zero, except for the end...
    event.reward = 0.0

    # wait for a bit
    time.sleep(0.1)
    world_state = malmo.getWorldState()

  # we only get final reward at very end so clobber last state with this reward
  episode_reward = world_state.rewards[0].getValue()
  episode.event[-1].reward = episode_reward
  print "REWARD\t%s\t%s" % (episode_reward, len(episode.event))

  # end of episode
  agent.add_episode(episode)
  if event_log:
    event_log.add_episode(episode)
  print "agent stats\t", agent.stats()


  
