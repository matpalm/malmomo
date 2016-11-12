import datetime
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
import StringIO
import sys
import yaml

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

def l2_norm(tensor):
  """(row wise) l2 norm of a tensor"""
  return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))

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

def smooth(values, factor=0.9):
  """smooth non zero values (by factor) towards 0th element."""
  new_values = [0] * len(values)
  for i in reversed(range(len(values))):
    if values[i] != 0:
      smoothed_value = values[i]
      j = 0
      while True:
        if i-j < 0: break
        new_values[i-j] += smoothed_value
        smoothed_value *= factor
        if abs(smoothed_value) < 1: break
        j += 1
  return new_values

def _unpack_rgb_bytes(render):
  assert not render.is_png_encoded
  flat_rgb = np.fromstring(render.bytes, dtype=np.float16)
  rgb = flat_rgb.reshape((render.height, render.width, 3))
  return rgb

def rgb_to_png_bytes(rgb):
  sio = StringIO.StringIO()
  plt.imsave(sio, rgb)
  return sio.getvalue()

def ensure_render_is_png_encoded(render):
  if render.is_png_encoded: return
  rgb = _unpack_rgb_bytes(render)
  render.bytes = rgb_to_png_bytes(rgb)
  render.is_png_encoded = True

def rgb_from_render(render):
  if render.is_png_encoded:
    # note PNG is always RGBA so we need to slice off A
    rgba = plt.imread(StringIO.StringIO(render.bytes))
    return rgba[:,:,:3]
  else:
    return _unpack_rgb_bytes(render)

class SaverUtil(object):
  def __init__(self, sess, ckpt_dir="/tmp", save_freq=60):
    self.sess = sess
    self.saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)
    self.ckpt_dir = ckpt_dir
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)
    assert save_freq > 0
    self.save_freq = save_freq
    self.load_latest_ckpt_or_init_if_none()

  def load_latest_ckpt_or_init_if_none(self):
    """loads latests ckpt from dir. if there are non run init variables."""
    # if no latest checkpoint init vars and return
    ckpt_info_file = "%s/checkpoint" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      # load latest ckpt
      info = yaml.load(open(ckpt_info_file, "r"))
      assert 'model_checkpoint_path' in info
      most_recent_ckpt = info['model_checkpoint_path']
      sys.stderr.write("loading ckpt %s\n" % most_recent_ckpt)
      self.saver.restore(self.sess, most_recent_ckpt)
      self.next_scheduled_save_time = time.time() + self.save_freq
    else:
      # no latest ckpts, init and force a save now
      sys.stderr.write("no latest ckpt in %s, just initing vars...\n" % self.ckpt_dir)
      self.sess.run(tf.initialize_all_variables())
      self.force_save()

  def force_save(self):
    """force a save now."""
    dts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_ckpt = "%s/ckpt.%s" % (self.ckpt_dir, dts)
    sys.stderr.write("saving ckpt %s\n" % new_ckpt)
    start_time = time.time()
    self.saver.save(self.sess, new_ckpt)
    print "save_took", time.time() - start_time
    self.next_scheduled_save_time = time.time() + self.save_freq

  def save_if_required(self):
    """check if save is required based on time and if so, save."""
    if time.time() >= self.next_scheduled_save_time:
      self.force_save()


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

