# malmomo

malmomo is the third in a series of deep rl projects.

![eg_rollout](eg_rollout.gif)

the first was [drivebot](http://matpalm.com/blog/drivebot/) which trained a [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
to do discrete control of a simulated rover. it included work on using [domain adversarial nets](https://arxiv.org/abs/1505.07818)
to make the controller robust to both simulated & real input but never had the control side of things working properly...

phase 2 was [cartpole++](https://github.com/matpalm/cartpoleplusplus) where the focus was on continuous control and raw pixel
input. this work included training a baseline [likelihood ratio policy gradient](http://www-anw.cs.umass.edu/~barto/courses/cs687/Policy%20Gradient-printable.pdf)
method for the low dimensional problem (7d pose of cart and pole) and training [DDPG](https://arxiv.org/abs/1509.02971) &
[NAF](https://arxiv.org/abs/1603.00748) for the high dimensional raw pixel input version.

phase 3 is *malmomo* (i.e. mo malmo (i.e. more malmo)). it includes training NAF against a [project malmo](https://github.com/Microsoft/malmo)
environment mapping raw pixels to continuous turn/move control. 

# main components

* run.py : orchestrates the interaction between malmo and an RL agent
* agents.py : currently describes two agents; a random baseline and one based on a trained NAF network
* replay_memory.py : provides the replay memory functionality for off policy training
* event_log.py : provides the ability to read / write episodes to / from disk

# example usage

pretty early days but...

## gather some offline data and then train a naf agent 

```
# start malmo client
cd <MALMO>/Minecraft/launchClient.sh

# start an agent randomly gathering data
./run.py --agent=Random --episode-time-ms=5000 --event-log-out=random.events

# review event data using event_log tools to dump imgs from every 10th episode
./event_log --file=random.events --img-output-dir=/tmp/imgs --nth=10
# review /tmp/imgs/e00000/s00000.png, etc

# train a NAF agent against just this data, don't have it add new experiences and run an eval every once in awhile...
./run.py --agent=Naf \
--episode-time-ms=5000 \
--replay-memory-size=150000 \
--event-log-in=random.events \
--event-log-out=naf.events --dont-store-new-memories \
--batches-per-step=100 

# drop the --dont-store-new-memories to have it add to training, though noise is currently disabled (wip)

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

