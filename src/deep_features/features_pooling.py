# Use pretrained network for feature extraction
from network import *
from dataset import *
import scipy.io
import numpy as np

domains = ['amazon','dslr','webcam']
layers = ['fc6','fc7','fc8']

print('Initializing network')
network = PretrainedNetwork()
outdir = '~/data/office31/vgg'

for domain in domains:
  # load data
  print('Loading {}'.format(domain))
  x,y = load_office31_domain(domain)
  y = np.array(y)
  # preprocess images
  xs = []
  for flip in [False,True]:
    for crop_h,crop_w in [(0,0),(0,1),(1,0),(1,1),(0.5,0.5)]:
      print('Preprocessing {} ({}) '.format(domain,len(xs)), end='\r')
      px = network.preprocess_many(x,crop_h=crop_h,crop_w=crop_w,flip=flip,size=224*3//2)
      xs.append(px)
  # for each layer
  for layer in layers:
    print('Generating {} {}'.format(domain, layer))
    out = [network.get_features(x, layer, preprocess=False) for x in xs]
    sums = sum(out)
    maxes = np.amax(out, 0)
    # Write to matlab matrix file
    scipy.io.savemat('{}/office-vgg-sumpool-{}-{}.mat'.format(outdir,domain,layer),{'x':sums,'y':y+1})
    scipy.io.savemat('{}/office-vgg-maxpool-{}-{}.mat'.format(outdir,domain,layer),{'x':maxes,'y':y+1})
    scipy.io.savemat('{}/office-vgg-catpool-{}-{}.mat'.format(outdir,domain,layer),{'x':np.array(out),'y':y+1})

