#!/usr/bin/env python
import sys
import os
import Image, ImageDraw, ImageChops

# stitches a bunch of images together into a grid

X, Y = 4, 3
W, H = 160, 120
pixel_buffer = 3

outdir = sys.argv[1]
dirs = sys.argv[2:]
imgs_per_directory = {}
max_imgs = 0

print dirs, len(dirs)
for directory in dirs:
  i = sorted(os.listdir(directory))
  imgs_per_directory[directory] = i
  max_imgs = max(max_imgs, len(i))

for i in range(max_imgs):
  background = Image.new('RGB', 
                         ((W*X)+(X*pixel_buffer), (H*Y)+(Y*pixel_buffer)), 
                         (0, 0, 0))
  for n, directory in enumerate(dirs):
    imgs = imgs_per_directory[directory]
    img_file = imgs[min(len(imgs)-1, i)]
    img = Image.open("%s/%s" % (directory, img_file))
    gx, gy = n%X, n/X
    x_offset = (gx*W)+(gx*pixel_buffer)
    y_offset = (gy*H)+(gy*pixel_buffer)
    background.paste(img, (x_offset, y_offset))
  background.save("%s/stitched_%03d.png" % (outdir, i))
  
