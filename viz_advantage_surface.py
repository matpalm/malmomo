#!/usr/bin/env python

# hacktasic viz of the quadratic surface of advantage around the max output
# for a couple of clear block on right / left / center cases

import agents
import argparse
import base_network
import Image
import numpy as np
import models
import sys
import tensorflow as tf
import replay_memory
import util
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', type=int, default=160, help="render width")
parser.add_argument('--height', type=int, default=120, help="render height")
agents.add_opts(parser)
models.add_opts(parser)
replay_memory.add_opts(parser)
util.add_opts(parser)
opts = parser.parse_args()
#opts.ckpt_dir = "runs/14/d/ckpts"  # last known good
print >>sys.stderr, "OPTS", opts

# init our rl_agent
agent_cstr = eval("agents.NafAgent")
agent = agent_cstr(opts)
an = agent.network

# prepare three plots; one for each of block on left, in center, or on right
fig = plt.figure(figsize=plt.figaspect(0.3))
plt.title(opts.ckpt_dir)
R = np.arange(-1, 1.25, 0.25)
X, Y = np.meshgrid(R, R)
for plot_idx, (img_file, desc) in enumerate([("runs/14/d/imgs/ep_00007/e0000.png", "on left"),
                                             ("runs/14/d/imgs/ep_00005/e0010.png", "center"),
                                             ("runs/14/d/imgs/ep_00005/e0024.png", "on right")]):
  print "calculating for", desc, "..."

  # slurp in bitmap
  img = Image.open(img_file)
  img = np.array(img)[:,:,:3]

  # collect q-value for all x, y values in one hit
  all_x_y_pairs = np.stack(zip(np.ravel(X), np.ravel(Y)))
  img_repeated = [img] * all_x_y_pairs.shape[0]
  q_values = agent.sess.run(an.q_value,
                            feed_dict={an.input_state: img_repeated,
                                       an.input_action: all_x_y_pairs,
                                       base_network.FLIP_HORIZONTALLY: False})
  Z = q_values.reshape(X.shape)

  # plot as surface
  ax = fig.add_subplot(1,3,plot_idx+1, projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', cmap=cm.coolwarm, linewidth=1)
  ax.set_title(desc)
  ax.set_xlabel("turn")
  ax.set_ylabel("move")
  ax.set_zlabel("q")

  # include single vertical line where q was maximised (according to output_action)
  output = agent.sess.run(an.output_action,
                          feed_dict={an.input_state: [img],
                                     base_network.FLIP_HORIZONTALLY: False})
  turn, move = np.squeeze(output)
  q_value = agent.sess.run(an.q_value,
                           feed_dict={an.input_state: [img],
                                      an.input_action: [[turn, move]],
                                      base_network.FLIP_HORIZONTALLY: False})
  print "turn", turn, "move", move, "=> q", np.squeeze(q_value), "Zmin=", np.min(Z), "Zmax=", np.max(Z)
  ax.plot([turn, turn], [move, move], [np.min(Z), np.max(Z)], linewidth=5)


# render
plt.show()
