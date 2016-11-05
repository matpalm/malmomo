#!/usr/bin/env python
from PIL import Image
import itertools
import json
import MalmoPython
import numpy as np
import os
import time
import util as u

output_dir = "runs/1/"

agent_host = MalmoPython.AgentHost()

mission = MalmoPython.MissionSpec(open("classroom_basic.xml").read(), True)
mission_record = MalmoPython.MissionRecordSpec()

for episode in itertools.count(1):
  print "EPISODE", episode

  episode_dir = "%s/e_%06d/" % (output_dir, episode)
  u.make_dir(episode_dir)

  # start new mission; explicitly wait for first observation 
  # (not just world_state.has_mission_begun)
  agent_host.startMission(mission, mission_record)
  world_state = agent_host.getWorldState()
  while len(world_state.observations) == 0:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

  # run until the mission has ended
  step = 0
  while world_state.is_mission_running:
    while len(world_state.observations) == 0:
      print "no observations?" 
      break # maybe bail on episode?

    # unpack general observation
    try:
      print "OBS", json.loads(world_state.observations[0].text)
    except:
      print "OBS failed ??"

    # write render out to disk
    render = world_state.video_frames[0]
    img = Image.frombytes('RGB', (render.width, render.height), str(render.pixels))
    img.save("%s/img_%06d.png" % (episode_dir, step))

    # decide a new move 
    turn, move = map(float, (np.random.random(size=2)*2)-1)
    print "ACTION %s" % json.dumps({"episode": episode, "step": step,
                                    "turn": turn, "move": move})
    agent_host.sendCommand("turn %f" % turn)
    agent_host.sendCommand("move %f" % move)

    # wait for a bit
    step += 1
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

  episode_reward = world_state.rewards[0].getValue()
  print "REWARD %s" % json.dumps({"reward": episode_reward})

  # end of episode
  # hackily let server reprep
  time.sleep(1)

print "done"
  
