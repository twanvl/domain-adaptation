# Use pretrained network for feature extraction
from network import *
from dataset import *
import scipy.io

domains = ['amazon','dslr','webcam']
layers = ['pool5','fc6','fc7','fc8']

print('Initializing network')
network = PretrainedNetwork()

for domain in domains:
  # load data
  print('Loading {}'.format(domain))
  x,y = load_office31_domain(domain)
  y = np.array(y)
  # preprocess images
  print('Preprocessing {}'.format(domain))
  x = network.preprocess_many(x)
  # for each layer
  for layer in layers:
    print('Generating {} {}'.format(domain, layer))
    out = network.get_features(x, layer, preprocess=False)
    # Write to matlab matrix file
    scipy.io.savemat('data/office-vgg-{}-{}.mat'.format(domain,layer),{'x':out,'y':y+1})

