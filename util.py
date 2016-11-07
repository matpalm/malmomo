import datetime
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import StringIO


def add_opts(parser):
  parser.add_argument('--gradient-clip', type=float, default=5,
                      help="do global clipping to this norm")
  parser.add_argument('--optimiser', type=str, default="GradientDescent",
                      help="tf.train.XXXOptimizer to use")
  parser.add_argument('--optimiser-args', type=str, default="{\"learning_rate\": 0.001}",
                      help="json serialised args for optimiser constructor")
  parser.add_argument('--print-gradients', action='store_true', 
                      help="whether to verbose print all gradients and l2 norms")

def dts():
  return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def clip_and_debug_gradients(gradients, opts):
  # extract just the gradients temporarily for global clipping and then rezip
  if opts.gradient_clip is not None:
    just_gradients, variables = zip(*gradients)
    just_gradients, _ = tf.clip_by_global_norm(just_gradients, opts.gradient_clip)
    gradients = zip(just_gradients, variables)
  # verbose debugging
  if opts.print_gradients:
    for i, (gradient, variable) in enumerate(gradients):
      if gradient is not None:
        gradients[i] = (tf.Print(gradient, [l2_norm(gradient)],
                                 "gradient %s l2_norm " % variable.name), variable)
  # done
  return gradients

def construct_optimiser(opts):
  optimiser_cstr = eval("tf.train.%sOptimizer" % opts.optimiser)
  args = json.loads(opts.optimiser_args)
  return optimiser_cstr(**args)

def make_dir(d):
  if not os.path.exists(d):
    os.makedirs(d)

def shape_and_product_of(t):
  shape_product = 1
  for dim in t.get_shape():
    try:
      shape_product *= int(dim)
    except TypeError:
      # Dimension(None)
      pass
  return "%s #%s" % (t.get_shape(), shape_product)

def rgb_to_png(rgb):
  """convert RGB data from render to png"""
  sio = StringIO.StringIO()
  plt.imsave(sio, rgb)
  return sio.getvalue()

def png_to_rgb(png_bytes):
  """convert png (from rgb_to_png) to RGB"""
  # note PNG is always RGBA so we need to slice off A
  rgba = plt.imread(StringIO.StringIO(png_bytes))
  return rgba[:,:,:3]


class OrnsteinUhlenbeckNoise(object):
  """generate time correlated noise for action exploration"""

  def __init__(self, dim, theta=0.01, sigma=0.2, max_magnitude=1.5):
    # dim: dimensionality of returned noise
    # theta: how quickly the value moves; near zero => slow, near one => fast
    #   0.01 gives very roughly 2/3 peaks troughs over ~1000 samples
    # sigma: maximum range of values; 0.2 gives approximately the range (-1.5, 1.5)
    #   which is useful for shifting the output of a tanh which is (-1, 1)
    # max_magnitude: max +ve / -ve value to clip at. dft clip at 1.5 (again for
    #   adding to output from tanh. we do this since sigma gives no guarantees
    #   regarding min/max values.
    self.dim = dim
    self.theta = theta
    self.sigma = sigma
    self.max_magnitude = max_magnitude
    self.state = np.zeros(self.dim)

  def sample(self):
    self.state += self.theta * -self.state
    self.state += self.sigma * np.random.randn(self.dim)
    self.state = np.clip(self.max_magnitude, -self.max_magnitude, self.state)
    return np.copy(self.state)

