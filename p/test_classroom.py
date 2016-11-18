#!/usr/bin/env python
import MalmoPython
import time

# set up out malmo client
malmo = MalmoPython.AgentHost()
spec = open("classroom_basic.xml").read()
spec = spec.replace("__WIDTH__", "640")
spec = spec.replace("__HEIGHT__", "480")
spec = spec.replace("__EPISODE_TIME_MS__", "10000000")
mission = MalmoPython.MissionSpec(spec, True)
mission_record = MalmoPython.MissionRecordSpec()

malmo.startMission(mission, mission_record)

while True:
  world_state = malmo.getWorldState()
  if len(world_state.rewards) > 0:
    print "len?", len(world_state.rewards)
    print world_state.rewards[0].getValue()
  time.sleep(1)

