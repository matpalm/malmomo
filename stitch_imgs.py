#!/usr/bin/env python
import argparse
import Image, ImageDraw, ImageChops
import os
import sys
import util

# stitches a bunch of images together into a grid

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--x', type=int, default=3, help="output columns")
parser.add_argument('--y', type=int, default=2, help="output rows")
parser.add_argument('--nth', type=int, default=1, help="only process every nth frame")
parser.add_argument('--max-frame', type=int, default=0,
                    help="if >0 ignore frames past this")
parser.add_argument('--output-dir', type=str, default="/tmp/stitch",
                    help="where to output stitched imgs")
parser.add_argument('dirs', nargs='+')

opts = parser.parse_args()
print opts

X, Y = opts.x, opts.y
W, H = 160, 120
pixel_buffer = 3

util.make_dir(opts.output_dir)
imgs_per_directory = {}
max_imgs = 0

assert len(opts.dirs) == X * Y, opts.dirs
for directory in opts.dirs:
  i = sorted(os.listdir(directory))
  if opts.max_frame > 0:
    i = i[:opts.max_frame]
  imgs_per_directory[directory] = i
  print "imgs per dir", directory, len(i)
  max_imgs = max(max_imgs, len(i))

i = 0
while i <= max_imgs:
  print i, "/", max_imgs
  background = Image.new('RGB',
                         ((W*X)+(X*pixel_buffer), (H*Y)+(Y*pixel_buffer)),
                         (0, 0, 0))
  for n, directory in enumerate(opts.dirs):
    imgs = imgs_per_directory[directory]
    img_file = imgs[min(len(imgs)-1, i)]
    img = Image.open("%s/%s" % (directory, img_file))
    gx, gy = n%X, n/X
    x_offset = (gx*W)+(gx*pixel_buffer)
    y_offset = (gy*H)+(gy*pixel_buffer)
    background.paste(img, (x_offset, y_offset))
  background.save("%s/stitched_%03d.png" % (opts.output_dir, i))
  i += opts.nth

print "mencoder mf://%s/ -ovc lavc -mf fps=10 -o stitched.avi" % opts.output_dir
