#!/usr/bin/env python
import agents
import argparse
import itertools
import json
import MalmoPython
import numpy as np
import os
from PIL import Image
import sys
import time
import util as u

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=int, default=None, help="output data to runs/N")
opts = parser.parse_args()

#output_dir = "runs/%d/" % opts.run

# set up out malmo client
malmo = MalmoPython.AgentHost()
mission = MalmoPython.MissionSpec(open("classroom_basic.xml").read(), True)
mission_record = MalmoPython.MissionRecordSpec()

# init out rl_agent
#agent = agents.RandomAgent()
agent = agents.NafAgent()

for episode in itertools.count(1):
  print "EPISODE", episode
#  episode_dir = "%s/e_%06d/" % (output_dir, episode)
#  u.make_dir(episode_dir)

  # start new mission; explicitly wait for first observation 
  # (not just world_state.has_mission_begun)
  while True:
    try:
      malmo.startMission(mission, mission_record)
      break
    except RuntimeError as r:
      print >>sys.stderr, "failed to start mission", r
      time.sleep(1)
  world_state = malmo.getWorldState()
  while len(world_state.observations) == 0:
    time.sleep(0.1)
    world_state = malmo.getWorldState()

  # run until the mission has ended
  step = 0
  state_action_rewards = []
  while world_state.is_mission_running:
    # extract render and convert to numpy array (w, h, 3) with values scaled 0.0 -> 1.0
    render = world_state.video_frames[0]
    img = Image.frombytes('RGB', (render.width, render.height), str(render.pixels))
    img = np.array(img, dtype=np.float16) / 255
#    img.save("%s/img_%06d.png" % (episode_dir, step))

    # decide action given state and send to malmo
    turn, move = agent.action_given_state(img)
    print "ACTION %s" % json.dumps({"episode": episode, "step": step,
                                    "turn": turn, "move": move})
    malmo.sendCommand("turn %f" % turn)
    malmo.sendCommand("move %f" % move)

    # reward state, action & reward
    # note: for now reward is always zero, except for the end...
    state_action_rewards.append((img, (turn, move), 0))

    # wait for a bit
    step += 1
    time.sleep(0.1)
    world_state = malmo.getWorldState()

  # we only get final reward at very end, so clobber last state with
  # this reward
  episode_reward = world_state.rewards[0].getValue()
  last_state, last_action, _last_reward = state_action_rewards[-1]
  state_action_rewards[-1] = (last_state, last_action, episode_reward)
  print "REWARD %s" % json.dumps({"reward": episode_reward})

  # end of episode
  agent.train(state_action_rewards)
  print agent.replay_memory.stats

print "done"
  
