import mxnet as mx
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params'),
 mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]

ctx = mx.cpu()

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

import matplotlib.pyplot as plt
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = mx.image.imread(fname)
    if img is None:
        return None
    if show:
        plt.imshow(img.asnumpy())
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    return img

def predict(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([img]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

#predict('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')

loop = True

my_map = {"pic2" : 'https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true',
          "pic1" : 'https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true',
          "pic5" : 'https://i.ytimg.com/vi/dOULgRg0Sf8/maxresdefault.jpg?raw=true',
	  "pic3" : 'http://www.daftrucks.de/~/media/images/daf%20trucks/online%20truck%20configurator/background/backgroundvisual.jpg?h=1184&w=1875&la=de-DE?raw=true',
	  "pic4" : 'https://i.ebayimg.com/00/s/NjcxWDEwMjQ=/z/q~IAAOSwkEVXGp5B/$_72.JPG?raw=true',
	  "pic6" : 'https://keyassets.timeincuk.net/inspirewp/live/wp-content/uploads/sites/2/2017/01/Orbea-gain.jpeg?raw=true',
          "shit" : 'https://telebasel.ch/wp-content/uploads/2017/09/wut6hxr.jpg?raw=true'}

while loop:
    name = raw_input("Name of Image to classify: ")
    if name == "exit":
        loop = False
    else:
        if name in my_map:
            url = my_map[name];
            predict(url)
        