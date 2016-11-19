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

# reopen stdout/stderr unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

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
parser.add_argument('--eval-freq', type=int, default=10,
                    help="do an eval (i.e. no noise) rollout every nth episodes")
parser.add_argument('--mission', type=str, default="classroom_1room.xml",
                    help="mission to run")
parser.add_argument('--overclock', action='store_true', help="run at x2 speed (needs more testing)")
parser.add_argument('--client-pool-size', type=int, default=1,
                    help="number of instances of launchClient.sh running")


agents.add_opts(parser)
models.add_opts(parser)
util.add_opts(parser)
replay_memory.add_opts(parser)
opts = parser.parse_args()
print >>sys.stderr, "OPTS", opts

def create_malmo_components():
  # setup client pool
  client_pool = MalmoPython.ClientPool()
  for i in range(opts.client_pool_size):
    client_pool.add( MalmoPython.ClientInfo( "127.0.0.1", 10000+i ) )
  # setup agent host
  malmo = MalmoPython.AgentHost()
  # can't do this without more complex caching of world state vid frames
  #malmo.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
  # define mission spec
  spec = open(opts.mission).read()
  spec = spec.replace("__WIDTH__", str(opts.width))
  spec = spec.replace("__HEIGHT__", str(opts.height))
  spec = spec.replace("__EPISODE_TIME_MS__", str(opts.episode_time_ms))
  spec = spec.replace("__MS_PER_TICK__", "25" if opts.overclock else "50")
  mission = MalmoPython.MissionSpec(spec, True)
  mission_record = MalmoPython.MissionRecordSpec()
  # return all
  return client_pool, malmo, mission, mission_record

client_pool, malmo, mission, mission_record = create_malmo_components()

# init our rl_agent
agent_cstr = eval("agents.%sAgent" % opts.agent)
agent = agent_cstr(opts)

event_log = event_log.EventLog(opts.event_log_out) if opts.event_log_out else None

for episode_idx in itertools.count(0):
  eval_episode = (episode_idx % opts.eval_freq == 0)
  print >>sys.stderr, "EPISODE", episode_idx, util.dts(), "eval =", eval_episode

  # start new mission; explicitly wait for first observation 
  # (not just world_state.has_mission_begun)
  mission_start = time.time()
  while True:
    try:
      malmo.startMission(mission, client_pool, mission_record, 0, "")
      break
    except RuntimeError as r:
      # have observed that getting stuck here doesn't recover, even if the servers
      # are restarted. try to recreate everything
      print >>sys.stderr, "failed to start mission", r
      print >>sys.stderr, "recreating malmo components..."
      time.sleep(1)
      client_pool, malmo, mission, mission_record = create_malmo_components()

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
      world_state = malmo.getWorldState()
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
    turn, move = agent.action_given(img, is_eval=eval_episode)
    malmo.sendCommand("turn %f" % turn)
    malmo.sendCommand("move %f" % move)
    event.action.value.extend([turn, move])

    # wait for a bit and refetch state
    time.sleep(0.05 if opts.overclock else 0.1)
    world_state = malmo.getWorldState()

    # check for any reward
    if world_state.rewards:
      assert len(world_state.rewards) == 1
      event.reward = world_state.rewards[0].getValue()
    else:
      event.reward = 0.0

    # dump debug
    print "ACTION\t%s" % json.dumps({"episode": episode_idx, "step": len(episode.event),
                                     "turn": turn, "move": move, "eval": eval_episode,
                                     "reward": event.reward})

  # report final reward for episode
  print "REWARD\t%s" % json.dumps({"episode": episode_idx, 
                                   "reward": sum([e.reward for e in episode.event]),
                                   "steps": len(episode.event), "eval": eval_episode})

  # end of episode
  agent.add_episode(episode)
  if event_log:
    event_log.add_episode(episode)
  print "agent stats\t", agent.stats()
  sys.stdout.flush()

  
