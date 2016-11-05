import os

def make_dir(d):
  if not os.path.exists(d):
    os.makedirs(d)

