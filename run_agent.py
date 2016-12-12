#!/usr/bin/env python
import agents
import argparse
import ckpt_util
import event_log
import grpc
import itertools
import json
import MalmoPython
import models
import model_pb2
import numpy as np
import os
from PIL import Image
import specs
import sys
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

# reopen stdout/stderr unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=int, default=None, help="output data to runs/N")
parser.add_argument('--width', type=int, default=160, help="render width")
parser.add_argument('--height', type=int, default=120, help="render height")
parser.add_argument('--episode-time-sec', type=int, default=30,
                    help="episode timeout (seconds)")
parser.add_argument('--no-reward-default', type=int, default=-1,
                    help="the dft reward to give when environment doesn't")
parser.add_argument('--agent', type=str, default="Naf", help="{Naf,Random}")
parser.add_argument('--event-log-out', type=str, default=None,
                    help="if set agent also write all episodes to this file")
parser.add_argument('--mission', type=int, default=1,
                    help="which mission to run (see specs.py)")
parser.add_argument('--overclock-rate', type=int, default=4,
                    help="overclock multiplier")
parser.add_argument('--eval', action='store_true',
                    help="if set run in eval (ie no noise)")
parser.add_argument('--onscreen-rendering', action='store_true',
                    help="if set do (slower) onscreen rendering")
parser.add_argument('--post-episode-sleep', type=int, default=1,
                    help="time (sec) to sleep after each episode")
parser.add_argument('--malmo-ports', type=str, default="10000",
                    help="comma seperated list of malmo client ports")
parser.add_argument('--trainer-port', type=int, default=20045,
                    help="grpc port to trainer. set to 0 to disable sending episodes")

agents.add_opts(parser)
ckpt_util.add_opts(parser)
models.add_opts(parser)
util.add_opts(parser)
opts = parser.parse_args()
print >>sys.stderr, "OPTS", opts

overclock_tick_ms = 50 / opts.overclock_rate
post_action_delay = 0.1 / opts.overclock_rate
print >>sys.stderr, "opts.overclock_rate", opts.overclock_rate, \
  "overclock_tick_ms", overclock_tick_ms, \
  "post_action_delay", post_action_delay

def create_malmo_components():
  # setup client pool
  client_pool = MalmoPython.ClientPool()
  for port in map(int, opts.malmo_ports.split(",")):
    print >>sys.stderr, "adding client with port %d" % port
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port))
  # setup agent host
  malmo = MalmoPython.AgentHost()
  # can't do this without more complex caching of world state vid frames
  #malmo.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
  # load mission spec
  mission = MalmoPython.MissionSpec(specs.classroom(opts, overclock_tick_ms), True)
  mission_record = MalmoPython.MissionRecordSpec()
  # return all
  return client_pool, malmo, mission, mission_record

client_pool, malmo, mission, mission_record = create_malmo_components()

# init our rl_agent
agent_cstr = eval("agents.%sAgent" % opts.agent)
agent = agent_cstr(opts)

# init event log (if logging events)
event_log = event_log.EventLog(opts.event_log_out) if opts.event_log_out else None

# hook up connection to trainer
if opts.trainer_port == 0:
  trainer = None
else:
  channel = grpc.insecure_channel("localhost:%d" % opts.trainer_port)
  trainer = model_pb2.ModelStub(channel)

for episode_idx in itertools.count(0):
  print util.dts(), "EPISODE", episode_idx, "eval", opts.eval

  # start new mission; explicitly wait for first observation
  # (not just world_state.has_mission_begun)
  mission_start = time.time()
  while True:
    try:
      # TODO: work out why this blocks and how to get it timeout somehow...
      malmo.startMission(mission, client_pool, mission_record, 0, "")
      break
    except RuntimeError as r:
      # have observed that getting stuck here doesn't recover, even if the servers
      # are restarted. try to recreate everything
      print >>sys.stderr, util.dts(), "failed to start mission", r
      print >>sys.stderr, util.dts(), "recreating malmo components..."
      time.sleep(1)
      client_pool, malmo, mission, mission_record = create_malmo_components()


  world_state = malmo.getWorldState()
  while len(world_state.observations) == 0:
    time.sleep(0.1)
    world_state = malmo.getWorldState()
  print util.dts(), "START_TIME", time.time()-mission_start

  # run until the mission has ended
  episode = model_pb2.Episode()
  while world_state.is_mission_running:
    # extract render and convert to numpy array (w, h, 3) with values scaled 0.0 -> 1.0
    if len(world_state.video_frames) == 0:
      time.sleep(0.1)
      world_state = malmo.getWorldState()
      continue

    event = episode.event.add()

    frame = world_state.video_frames[0]
    img = np.array(Image.frombytes('RGB', (frame.width, frame.height), str(frame.pixels)))
    event.render.width = frame.width
    event.render.height = frame.height
    event.render.bytes = img.tostring()
    event.render.is_png_encoded = False

    # decide action given state and send to malmo
    turn, move = agent.action_given(img, is_eval=opts.eval)
    malmo.sendCommand("turn %f" % turn)
    malmo.sendCommand("move %f" % move)
    event.action.value.extend([turn, move])

    # wait for next state
    while True:
      time.sleep(0.01)
      world_state = malmo.getWorldState()
      num_obs = world_state.number_of_observations_since_last_state
      if num_obs > 0 or not world_state.is_mission_running: break

    # check for any reward
    if world_state.rewards:
      assert len(world_state.rewards) == 1
      event.reward = world_state.rewards[0].getValue()
    else:
      event.reward = opts.no_reward_default

    # dump debug
    print "ACTION\t%s" % json.dumps({"episode": episode_idx, "step": len(episode.event),
                                     "turn": turn, "move": move, "eval": opts.eval,
                                     "reward": event.reward})

  # report final reward for episode
  print "REWARD\t%s" % json.dumps({"episode": episode_idx,
                                   "reward": sum([e.reward for e in episode.event]),
                                   "steps": len(episode.event), "eval": opts.eval})

  # end of episode
  agent.end_of_episode()
  if trainer:
    try:
      # TODO: send back queue size so agent can decide to backoff a bit?
      trainer.AddEpisode(episode.event)
    except grpc._channel._Rendezvous as e:
      # TODO: be more robust here
      print "warning: failed to add episode", e
  if event_log:
    event_log.add_episode(episode)
  sys.stdout.flush()
  time.sleep(opts.post_episode_sleep)
