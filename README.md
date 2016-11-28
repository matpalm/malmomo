# malmomo

malmomo is the third in a series of deep rl projects.

![eg_rollout](eg_rollout.gif)

the first was [drivebot](http://matpalm.com/blog/drivebot/) which trained a [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
to do discrete control of a simulated rover. it included work on using [domain adversarial nets](https://arxiv.org/abs/1505.07818)
to make the controller robust to both simulated & real input but never had the transfer of control from sim to real working properly...

phase 2 was [cartpole++](https://github.com/matpalm/cartpoleplusplus) where the focus was on continuous control and raw pixel
input. this work included training a baseline [likelihood ratio policy gradient](http://www-anw.cs.umass.edu/~barto/courses/cs687/Policy%20Gradient-printable.pdf)
method for the low dimensional problem (7d pose of cart and pole) and training [DDPG](https://arxiv.org/abs/1509.02971) &
[NAF](https://arxiv.org/abs/1603.00748) for the high dimensional raw pixel input version.

phase 3 is *malmomo* (i.e. mo malmo (i.e. more malmo)). it includes training NAF against a [project malmo](https://github.com/Microsoft/malmo)
environment mapping raw pixels to continuous turn/move control. the animation above shows an evaluation in an agent trained in a simpler 2x2
maze that generalises to a larger maze (but still gets stuck in corners :)

# main components

* run.py : orchestrates the interaction between malmo and an RL agent
* agents.py : currently describes two agents; a random baseline and one based on a trained NAF network
* replay_memory.py : provides the replay memory functionality for off policy training
* event_log.py : provides the ability to read / write episodes to / from disk

# example usage

## gather some offline data and then train a naf agent

```
# start malmo client
<MALMO>/Minecraft/launchClient.sh

# start an agent randomly gathering data
mkdir random_agent
./run.py --agent=Random --episode-time-ms=5000 --event-log-out=random_agent/events

# review event data using event_log tools to dump imgs from every 10th episode
./event_log --file=random_agent/events --img-output-dir=random_agent/imgs --nth=10
# review /tmp/imgs/e00000/s00000.png, etc

# train a NAF agent with replay memory seeded with first 1000 episodes from random agent
mkdir naf_agent
./run.py --agent=Naf \
--episode-time-ms=5000 \
--replay-memory-size=150000 \
--event-log-in=random_agent/events --event-log-in-num=1000 \
--event-log-out=naf_agent/events \
--ckpt-dir=naf_agent/ckpts

# do an eval of this agent (i.e. no noise added to actions) using latest model checkpoint
./run.py --agent=Naf \
--dont-store-new-memories --eval-freq=1 \
--event-log-out=naf_agent/eval_events \
--ckpt-dir=naf_agent/ckpts
```

### gather data across multiple instances

```
# start N, say 3, malmo clients
shell1> launchClient.sh -port 11100
shell2> launchClient.sh -port 11200
shell3> launchClient.sh -port 11300
```

```
# start N overclocked random agents
shell4> ./run.py --agent=Random --overclock-rate=4 --event-log-out=random1.events --client-ports=11100,11200,11300
shell5> ./run.py --agent=Random --overclock-rate=4 --event-log-out=random2.events --client-ports=11100,11200,11300
shell6> ./run.py --agent=Random --overclock-rate=4 --event-log-out=random3.events --client-ports=11100,11200,11300
```

to enable / disable verbose debugging issue a `kill -sigusr1` to running process.

# install stuff

malmomo depends on

* [malmo](https://github.com/Microsoft/malmo) for orchestrating minecraft
* [grpc](http://www.grpc.io/) & [protobuffers](https://developers.google.com/protocol-buffers/) for binary storage and transport of experience data
* [minecraft](https://minecraft.net)

```
pip install grpcio grpcio-tools
python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. *proto
```
