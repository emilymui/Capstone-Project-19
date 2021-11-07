import os
import json
from skimage import io
root = '../data/mimic-cxr-jpg/files/'
running_mean = 0
running_std = 0
count = 0
for root, dirs, files in os.walk(root, topdown=False):
    for f in files:
        if f.endswith('.jpg'):
            image = io.imread(os.path.join(root,f))
            running_mean += image.mean()/255 # /255 is based on how ImageNet means calc'd
            running_std += image.std()/255
            count += 1
d = {'mean': running_mean/count, 'std': running_std/count}
json.dump(d, open('cxr_stats.json','w'))
