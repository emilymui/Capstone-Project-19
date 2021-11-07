import os
import json
from skimage import io
root = '../data/mimic-cxr-jpg/files/'
lowest_w = (10000,10000)
lowest_h = (10000,10000)
for root, dirs, files in os.walk(root, topdown=False):
    for f in files:
        if f.endswith('.jpg'):
            image = io.imread(os.path.join(root,f))
            if image.shape[0] < lowest_w[0]:
                lowest_w = image.shape
            if image.shape[1] < lowest_h[1]:
                lowest_h = image.shape
print(f'Image with lowest width has shape {lowest_w}\nImage with lowest height has shape {lowest_h}')
