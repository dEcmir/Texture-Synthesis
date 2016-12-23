import pickle
import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
# The layer below requires GPU, replace Conv2DDNNLayer by Conv2DLayer
# and remove the flip_filter option
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

# Note: tweaked to use average pooling instead of maxpooling
def build_model(image_w, path=None):
    net = {}
    net['input'] = InputLayer((1, 3, image_w, image_w))
    for i in [3,5,7,11,15,23,37,55]:
        net[str(i)] = ConvLayer(net['input'], 128, i, pad=(i-1)/2)

    return net
