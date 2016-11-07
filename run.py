#!/usr/bin/env python
import agents
import argparse
import itertools
import json
import MalmoPython
import models
import numpy as np
import os
from PIL import Image
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

agents.add_opts(parser)
models.add_opts(parser)
util.add_opts(parser)
opts = parser.parse_args()

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

for episode_idx in itertools.count(1):
  print "EPISODE", episode_idx, util.dts()
#  episode_dir = "%s/e_%06d/" % (output_dir, episode)
#  u.make_dir(episode_dir)

  # start new mission; explicitly wait for first observation 
  # (not just world_state.has_mission_begun)
  while True:
    try:
      malmo.startMission(mission, mission_record)
      break
    except RuntimeError as r:
      print >>sys.stderr, "failed to start mission", r, episode_idx
      time.sleep(1)
  world_state = malmo.getWorldState()
  while len(world_state.observations) == 0:
    print >>sys.stderr, "started, but no obs?", episode_idx
    time.sleep(0.1)
    world_state = malmo.getWorldState()

  # run until the mission has ended
  episode = []
  while world_state.is_mission_running:
    # extract render and convert to numpy array (w, h, 3) with values scaled 0.0 -> 1.0
    render = world_state.video_frames[0]
    img = Image.frombytes('RGB', (render.width, render.height), str(render.pixels))
    img = np.array(img, dtype=np.float16) / 255

    # decide action given state and send to malmo
    turn, move = agent.action_given(img)
    print "ACTION %s" % json.dumps({"episode": episode_idx, "step": len(episode),
                                    "turn": turn, "move": move})
    malmo.sendCommand("turn %f" % turn)
    malmo.sendCommand("move %f" % move)

    # reward state, action & reward
    # note: for now reward is always zero, except for the end...
    episode.append((img, (turn, move), 0))

    # wait for a bit
    time.sleep(0.1)
    world_state = malmo.getWorldState()

  # we only get final reward at very end so clobber last state with
  # this reward
  episode_reward = world_state.rewards[0].getValue()
  last_state, last_action, _last_reward = episode[-1]
  episode[-1] = (last_state, last_action, episode_reward)
  print "REWARD %s" % json.dumps({"reward": episode_reward})

  # end of episode
  agent.add_episode(episode)
  print "agent stats", agent.stats()


  
