#!/usr/bin/env python

# hacktasic viz of the quadratic surface of advantage around the max output
# for a couple of clear block on right / left / center cases

import agents
import argparse
import base_network
import Image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import models
from mpl_toolkits.mplot3d import axes3d
import os
import replay_memory
import sys
import tensorflow as tf
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', type=int, default=160, help="render width")
parser.add_argument('--height', type=int, default=120, help="render height")
parser.add_argument('--input-dir', type=str, default="runs/14/d/imgs/ep_00007")
parser.add_argument('--output-dir', type=str, default="/tmp")
agents.add_opts(parser)
models.add_opts(parser)
replay_memory.add_opts(parser)
util.add_opts(parser)
opts = parser.parse_args()
#opts.ckpt_dir = "runs/14/d/ckpts"  # last known good
print >>sys.stderr, "OPTS", opts

util.make_dir(opts.output_dir)

# init our rl_agent
agent_cstr = eval("agents.NafAgent")
agent = agent_cstr(opts)
an = agent.network

# prepare three plots; one for each of block on left, in center, or on right
R = np.arange(-1, 1.25, 0.25)
X, Y = np.meshgrid(R, R)

for img_file in sorted(os.listdir(opts.input_dir)):
  # prep background; img will be on left, surface on right
  background = Image.new('RGB',
                         (10+320+10+320+10, 10+240+10),
                         (0, 0, 0))

  # slurp in bitmap
  png_img = Image.open("%s/%s" % (opts.input_dir, img_file))

  # take a copy for feeding into network
  img = np.array(png_img)[:,:,:3]

  # paste into background
  png_img = png_img.resize((320, 240))
  background.paste(png_img, (10, 10))

  # collect q-value for all x, y values in one hit
  all_x_y_pairs = np.stack(zip(np.ravel(X), np.ravel(Y)))
  img_repeated = [img] * all_x_y_pairs.shape[0]
  q_values = agent.sess.run(an.q_value,
                            feed_dict={an.input_state: img_repeated,
                                       an.input_action: all_x_y_pairs,
                                       base_network.FLIP_HORIZONTALLY: False})
  Z = q_values.reshape(X.shape)

  # plot as surface. make fig size x1.5 what it will be in composite
  # as hacky way of scaling down fonts :/
  dpi = 100
  fig = plt.figure(figsize=(480./dpi, 360./dpi), dpi=dpi)
  ax = fig.gca(projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', cmap=cm.coolwarm, linewidth=1)
  # TODO: set zlim as max over all images

#  ax.set_title(desc)
  ax.set_xlabel("turn")
  ax.set_ylabel("move")
  ax.set_zlabel("q")
  ax.set_zlim(0, 100)

  # include single vertical line where q was maximised (according to output_action)
  output = agent.sess.run(an.output_action,
                          feed_dict={an.input_state: [img],
                                     base_network.IS_TRAINING: False,
                                     base_network.FLIP_HORIZONTALLY: False})
  q_value = agent.sess.run(an.q_value,
                           feed_dict={an.input_state: [img],
                                      an.input_action: output,
                                      base_network.IS_TRAINING: False,
                                      base_network.FLIP_HORIZONTALLY: False})
  turn, move = np.squeeze(output)
  print "img_file", img_file, "turn", turn, "move", move, "=> q", np.squeeze(q_value), "Zmin=", np.min(Z), "Zmax=", np.max(Z)
  ax.plot([turn, turn], [move, move], [0, 100], linewidth=5)

  # render and paste into RHS of composite image
  plt.savefig("/dev/shm/foo.png")
  img = Image.open("/dev/shm/foo.png")
  img = img.resize((320, 240))
  img.save("/dev/shm/foo2.png")
  background.paste(img, (10+320+10,10))

  background.save("%s/%s" % (opts.output_dir, img_file))
