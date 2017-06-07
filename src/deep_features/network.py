# The used network architecture
# Based on
#  https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20%28VGG_S%29.ipynb

import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.regularization import l2, regularize_network_params
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
import skimage.transform

class PretrainedNetwork:
  def __init__(self, load=True):
    # Architecture
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    self.output_layer = net['fc8']
    self.net = net
    
    if load:
      self.load_weights()
    
    # Compile
    self.predict_fn = None
    self.predict_fns = {}
    self.train_fn = {}
    self.lr = theano.shared(np.array(1e-2, dtype=np.float32))
    self.regularizer_amount = theano.shared(np.array(4e-5, dtype=np.float32))
  
  def get_output_fn(self,layer):
    input_var = self.net['input'].input_var
    out = lasagne.layers.get_output(layer, deterministic=True)
    return theano.function([input_var], out)
  
  def add_output_layer(self, num_units, after='drop7'):
    self.output_layer = DenseLayer(self.net[after], num_units=num_units, nonlinearity=lasagne.nonlinearities.softmax)
    self.predict_fn = None
    self.train_fn = {}
  
  def load_weights(self):
    # weights
    import pickle
    with open('/home/twanvl/test/vgg_cnn_s.pkl','rb') as file:
      model = pickle.load(file, encoding='latin1')
    self.classes = model['synset words']
    self.mean_image = model['mean image']
    lasagne.layers.set_all_param_values(self.output_layer, model['values'])
  
  def save_weights_np(self, filename):
    np.savez(filename, *lasagne.layers.get_all_param_values(self.output_layer), mean_image=self.mean_image)
  def load_weights_np(self, filename):
    params = lasagne.layers.get_all_params(self.output_layer)
    with np.load(filename) as f:
      param_values = [f['arr_%d' % i] for i in range(len(params))]
      self.mean_image = f['mean_image']
    lasagne.layers.set_all_param_values(self.output_layer, param_values)

  def preprocess_many(self, ims, **kwargs):
    # Preprocess a list of images
    return np.array([self.preprocess(x, many=True, **kwargs) for x in ims])
  
  def preprocess(self, im, many=False, crop_h=0.5, crop_w=0.5, flip=False, size=256, smallest=True, random=False):
    # Preprocess an image
    # Resize so smallest/largest dim = 256, preserving aspect ratio
    im = resize(im, size, smallest)
    # Central crop to 224x224
    h, w, _ = im.shape
    if random:
      y0 = np.random.randint(h-224)
      x0 = np.random.randint(w-224)
      flip = np.random.randint(2)
    else:
      y0 = int((h-224)*crop_h)
      x0 = int((w-224)*crop_w)
    im = im[y0:y0+224, x0:x0+224]
    # Flip horizontally?
    if flip:
      im = im[:,::-1]
    if not many:
      rawim = np.copy(im).astype('uint8')
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    # Convert to BGR
    im = im[::-1, :, :]
    # Subtract mean
    im = im - self.mean_image
    if many:
      return floatX(im)
    else:
      return rawim, floatX(im[np.newaxis])

  def classify(self,im, preprocess=False, **kwargs):
    if preprocess:
      im = self.preprocess_many(im, **kwargs)
    if self.predict_fn is None:
      self.predict_fn = self.get_output_fn(self.output_layer)
    prob = batch_predict(self.predict_fn, im)
    return np.array(np.argmax(prob, axis=1), dtype=np.int32)
  
  def classify_test(self,im, **kwargs):
    # Run a test of the classifier, output nice looking matplotlib figure
    rawim, im = self.preprocess(im, **kwargs)
    #prob = np.array(lasagne.layers.get_output(self.output_layer, im, deterministic=True).eval())
    if self.predict_fn is None:
      self.predict_fn = self.get_output_fn(self.output_layer)
    prob = np.array(self.predict_fn(im))
    top5 = np.argsort(prob[0])[-1:-6:-1]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(rawim.astype('uint8'))
    plt.axis('off')
    for n, label in enumerate(top5):
      plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, self.classes[label]), fontsize=14)
    
  def get_features(self, im, layer, preprocess=False):
    if layer not in self.predict_fns:
      self.predict_fns[layer] = self.get_output_fn(self.net[layer])
    # apply
    if preprocess:
      rawim, im = self.preprocess(im)
    return batch_predict(self.predict_fns[layer], im)

  def get_train_fn(self, last_only=False):
    input_var = self.net['input'].input_var
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(self.output_layer)
    loss = categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    error = T.mean(T.neq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    regularization = self.regularizer_amount * regularize_network_params(self.output_layer, l2)
    if last_only:
      all_params = self.output_layer.get_params(trainable=True)
    else:
      all_params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
    updates = nesterov_momentum(loss + regularization, all_params, learning_rate=self.lr)
    return theano.function([input_var, target_var], (loss,error), updates=updates)
  
  def train(self, x, y, num_epochs=50, learning_rate=1e-3, batchsize=128, regularizer_amount=5e-4, preprocess=False, last_only=False):
    if last_only not in self.train_fn:
      self.train_fn[last_only] = self.get_train_fn(last_only)
    train_fn = self.train_fn[last_only]
    self.regularizer_amount.set_value(np.float32(regularizer_amount))
    #augment = augment_data
    augment = None
    
    for epoch in range(num_epochs):
      if epoch < 0.8*num_epochs:
        lr = learning_rate
      elif epoch < 0.9*num_epochs:
        lr = learning_rate / 10
      else:
        lr = learning_rate / 100
      self.lr.set_value(np.float32(lr))
      
      loss = 0
      err  = 0
      n    = 0
      for batch_x,batch_y in iterate_minibatches(x, y, batchsize=batchsize, shuffle=True, augment=augment):
        if preprocess:
          batch_x = self.preprocess_many(batch_x, random=True)
        l,e = train_fn(batch_x,batch_y)
        loss += l
        err += e
        n += 1
        print("  {:3} / {:3}:  loss={:6.3f}, error={:5.3f}  ".format(epoch,num_epochs,loss/n,err/n), end='\r')
      if epoch%10 == 9:
        print()
    

def batch_predict(fun, x, batchsize=128):
  if x.shape[0] < batchsize:
    return fun(x)
  else:
    y = []
    for start in range(0, x.shape[0], batchsize):
      end = min(start + batchsize, x.shape[0])
      y.append(fun(x[start:end]))
    return np.concatenate(y)

# > We follow the simple data augmentation in [24] for training:
# > 4 pixels are padded on each side, and a 32*32 crop is randomly sampled
# > from the padded image or its horizontal flip.
def augment_data(data,max_shift=100):
  # input (N,channel,h,w)
  out = np.empty_like(data)
  for i in range(np.shape(data)[0]):
    # sample same size image from padded image
    xoffs = np.random.randint(max_shift*2+1)
    yoffs = np.random.randint(max_shift*2+1)
    out[i] = np.pad(data[i],[(0,0),(max_shift,max_shift),(max_shift,max_shift)],'constant')[:, xoffs:xoffs+np.shape(data)[2], yoffs:yoffs+np.shape(data)[2]]
    if np.random.random_sample() < 0.5:
      # flip horizontally
      out[i] = out[i][:,:,::-1]
  return out
  
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=None):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs), batchsize):
    end_idx = min(start_idx + batchsize, len(inputs))
    if shuffle:
      excerpt = indices[start_idx:end_idx]
    else:
      excerpt = slice(start_idx, end_idx)
    # Data augmentation
    if augment:
      yield augment(inputs[excerpt]), targets[excerpt]
    else:
      yield inputs[excerpt], targets[excerpt]

def resize(im, size=256, smallest=True):
  h, w, _ = im.shape
  if (h < w and smallest) or (h > w and not smallest):
    if h != size:
      im = skimage.transform.resize(im, (size, (w*size)//h), preserve_range=True)
  else:
    if w != size:
      im = skimage.transform.resize(im, ((h*size)//w, size), preserve_range=True)
  return im

class ConcatDatasets:
  def __init__(self, a, b):
    self.a = a
    self.b = b
  def __len__(self):
    return len(self.a) + len(self.b)
  def __getitem__(self,idxs):
    pass #TODO
