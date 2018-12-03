# Use pretrained network for feature extraction
import scipy.io
import os.path
import keras
from keras import backend as K
from keras.models import Model, Input
import tensorflow as tf
from os.path import expanduser
import progressbar

from dataset import *

#-------------------------------------------------------------------------------
# Configuration and initialization
#-------------------------------------------------------------------------------

# Set Memory allocation in tf/keras to Growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

outdir = expanduser('~/data/domain-adaptation/')

#-------------------------------------------------------------------------------
# Feature extraction
#-------------------------------------------------------------------------------

def extract_features(dataset, architecture, model, augmentation):
  for domain in dataset_domains(dataset):
    if augmentation:
      filename = '{}/{}-{}-{}.mat'.format(outdir,dataset,domain,architecture)
    else:
      filename = '{}/{}-{}-{}-no-augment.mat'.format(outdir,dataset,domain,architecture)
    if os.path.isfile(filename):
      continue
    print('Output file does not exist: {}'.format(filename))
    
    if augmentation:
      padding = 32
      flipping = True
    else:
      padding = 0
      flipping = False
    
    print('Loading {} {} {}'.format(dataset,domain,'' if augmentation else '(no augmentation)'))
    x,y = load_data(dataset,domain)
    y = np.asarray(y, dtype=np.float32) + 1
    
    print('Preprocessing')
    bar = progressbar.ProgressBar()
    x = [preprocess_for(architecture,im,padding) for im in bar(x)]
    
    print('Calculating features for {} {} {}'.format(dataset,domain,'' if augmentation else '(no augmentation)'))
    bar = progressbar.ProgressBar()
    features = np.stack([image_features(im,model,padding,flipping) for im in bar(x)])
    
    print('Saving {}'.format(filename))
    scipy.io.savemat(filename,{'x':features,'y':y})

# Removes the last layer from a Keras pretrained model
def remove_last_layer(model):
  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  model.layers[-1].outbound_nodes = []
  model = Model(inputs=model.inputs, outputs=model.outputs)
  return model

def create_model(architecture):
  if architecture is 'inception_resnet_v2':
    return remove_last_layer(keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet'))
  elif architecture is 'xception':
    return remove_last_layer(keras.applications.xception.Xception(weights='imagenet'))
  elif architecture is 'inception_v3':
    return remove_last_layer(keras.applications.inception_v3.InceptionV3(weights='imagenet'))
  elif architecture is 'resnet50':
    return remove_last_layer(keras.applications.resnet50.ResNet50(weights='imagenet'))
  elif architecture is 'vgg19':
    return remove_last_layer(keras.applications.vgg19.VGG19(weights='imagenet'))
  elif architecture is 'vgg16':
    return remove_last_layer(keras.applications.vgg16.VGG16(weights='imagenet'))

def resize_image(x,size,padding):
  return np.asarray(scipy.misc.imresize(x, (size+2*padding, size+2*padding, 3)),dtype='float32')

def preprocess_for(architecture, x, padding):
  if architecture is 'inception_resnet_v2':
    return keras.applications.inception_resnet_v2.preprocess_input(resize_image(x, 259, padding))
  elif architecture is 'xception':
    return keras.applications.xception.preprocess_input(resize_image(x, 259, padding))
  elif architecture is 'inception_v3':
    return keras.applications.inception_v3.preprocess_input(resize_image(x, 259, padding))
  elif architecture is 'resnet50':
    return keras.applications.resnet50.preprocess_input(resize_image(x, 224, padding))
  elif architecture is 'vgg19':
    return keras.applications.vgg19.preprocess_input(resize_image(x, 224, padding))
  elif architecture is 'vgg16':
    return keras.applications.vgg16.preprocess_input(resize_image(x, 224, padding))

# Get features for x_in images from model (Average of 18 image representations = 9 image crops * 2 variantions (normal and horizontally flipped))
def image_features(x, model, padding, flipping):
  size = x.shape[0] - 2*padding
  images = []
  if padding == 0:
    images.append(x)
  else:
    for i in range(3):
      for j in range(3):
        images.append(x[padding*i:padding*i+size, padding*j:padding*j+size, :])
  if flipping:
    images.extend([np.fliplr(x) for x in images])
      # Note: xs.extend(f(x) for x in xs) results in an infinite loop, converting generator to a list first solves that
  return np.mean(model.predict(np.stack(images)), axis=0)

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  for architecture in ['vgg16','vgg19','resnet50','inception_v3','xception','inception_resnet_v2']:
    model = create_model(architecture)
    for dataset in ['office-caltech','office-31']:
      for augment in [True,False]:
        extract_features(dataset, architecture, model, augment)

