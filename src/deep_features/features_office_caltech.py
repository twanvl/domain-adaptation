# Use pretrained network for feature extraction,
# for the office-caltech dataset (see https://cs.stanford.edu/~jhoffman/domainadapt/)
# get image names from matlab files.
from network import *
from dataset import *
import scipy.io

domains = ['amazon','Caltech','dslr','webcam']
layers = ['fc6','fc7']
outdir = '~/data/office-caltech'

print('Initializing network')
network = PretrainedNetwork()

for domain in domains:
  # load data
  print('Loading {}'.format(domain))
  x,y = load_office_caltech_domain(domain)
  y = np.array(y)
  # preprocess images
  print('Preprocessing {}'.format(domain))
  print('NB: shape is {}'.format(np.shape(x[0])))
  xp = network.preprocess_many(x)
  # for each layer
  for layer in layers:
    print('Generating {} {}'.format(domain, layer))
    out = network.get_features(xp, layer, preprocess=False)
    # Write to matlab matrix file
    scipy.io.savemat('{}/office-caltech-vgg-{}-{}.mat'.format(outdir,domain,layer),{'x':out,'y':y+1})
    
  # With pooling
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
    scipy.io.savemat('{}/office-caltech-vgg-sumpool-{}-{}.mat'.format(outdir,domain,layer),{'x':sums,'y':y+1})
    scipy.io.savemat('{}/office-caltech-vgg-maxpool-{}-{}.mat'.format(outdir,domain,layer),{'x':maxes,'y':y+1})
    scipy.io.savemat('{}/office-caltech-vgg-catpool-{}-{}.mat'.format(outdir,domain,layer),{'x':np.array(out),'y':y+1})

