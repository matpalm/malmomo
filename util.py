import datetime
import gzip
import Image
import json
import numpy as np
import os
import tensorflow as tf
#import time
import StringIO
#import sys
#import yaml

def add_opts(parser):
  parser.add_argument('--gradient-clip', type=float, default=10,
                      help="do global clipping to this norm")
  parser.add_argument('--optimiser', type=str, default="Adam",
                      help="tf.train.XXXOptimizer to use")
  parser.add_argument('--optimiser-args', type=str, default="{\"learning_rate\": 0.0001}",
                      help="json serialised args for optimiser constructor")

def dts():
  return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def l2_norm(tensor):
  """(row wise) l2 norm of a tensor"""
  return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))

def clip_and_debug_gradients(gradients, opts):
  # extract just the gradients temporarily for global clipping and then rezip
  if opts.gradient_clip is not None:
    just_gradients, variables = zip(*gradients)
    just_gradients, _ = tf.clip_by_global_norm(just_gradients, opts.gradient_clip)
    gradients = zip(just_gradients, variables)
  # create verbose debugging version that when evaled will print norms
  print_gradient_norms = []
  for i, (gradient, variable) in enumerate(gradients):
    if gradient is not None:
      print_gradient_norms.append(tf.Print(tf.constant(0), # don't actually return anything
                                           [l2_norm(gradient)],
                                           "gradient %s l2_norm " % variable.name))
  # done
  return gradients, print_gradient_norms

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

def smooth(values, factor):
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
  flat_rgb = np.fromstring(render.bytes, dtype=np.uint8)
  rgb = flat_rgb.reshape((render.height, render.width, 3))
  return rgb

def rgb_to_png_bytes(rgb):
  img = Image.fromarray(rgb)
  sio = StringIO.StringIO()
  img.save(sio, format="png")
  return sio.getvalue()

def ensure_render_is_png_encoded(render):
  if render.is_png_encoded: return
  rgb = _unpack_rgb_bytes(render)
  render.bytes = rgb_to_png_bytes(rgb)
  render.is_png_encoded = True

def rgb_from_render(render):
  if render.is_png_encoded:
    # note PNG is always RGBA so we need to slice off A
    img = Image.open(StringIO.StringIO(render.bytes))
    img = np.array(img)
    if img.shape[2] == 4: img = img[:,:,:3]  # backwards compat with older float32 events
    return img
  else:
    return _unpack_rgb_bytes(render)
