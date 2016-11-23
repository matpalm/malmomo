#!/usr/bin/env python

import numpy as np

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

if __name__ == '__main__':
  import sys
  theta, sigma = map(float, sys.argv[1:])
  ou = OrnsteinUhlenbeckNoise(1, theta, sigma)
  for n in range(500):
    print n, np.squeeze(ou.sample())
