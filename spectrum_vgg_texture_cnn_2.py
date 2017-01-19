
import lasagne
import numpy as np
import skimage.transform
import scipy
import PIL.Image

import sys

import theano
import theano.tensor as T

from lasagne.utils import floatX

import matplotlib.pyplot as plt


from utils import prep_image, deprocess
from fourier_loss import define_fourier_loss, define_fourier_grad
from vgg import build_model

def main(path):
    IMAGE_W = 256

    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

    net = build_model(IMAGE_W, 'vgg19_normalized.pkl')

    art = plt.imread(path)
    art = np.array(art)
    rawim, art = prep_image(art, MEAN_VALUES, image_w=IMAGE_W)

    def gram_matrix(x):
        x = x.flatten(ndim=3)
        g = T.tensordot(x, x, axes=([2], [2]))
        return g


    def style_loss(A, X, layer):
        a = A[layer]
        x = X[layer]

        A = gram_matrix(a)
        G = gram_matrix(x)

        N = a.shape[1]
        M = a.shape[2] * a.shape[3]

        loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
        return loss

    def total_variation_loss(x):
        return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

    layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']# ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    # layers = ['conv1_1', 'conv2_1', 'conv3_1']
    layers = {k: net[k] for k in layers}

    # Precompute layer activations for photo and artwork
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}

    # Get expressions for layer activations for generated image
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    # Define loss function
    losses = []

    # style loss
    for lay in layers:
        losses.append(0.2e6 * style_loss(art_features, gen_features, lay))

    # total variation penalty
    losses.append(0.1e-6 * total_variation_loss(generated_image))

    total_loss = sum(losses)
    grad = T.grad(total_loss, generated_image)

    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)

    # spectral loss and gradient
    spectral_loss = define_fourier_loss(art[0])
    spectral_grad = define_fourier_grad(art[0])

    # spectral weight
    spectral_weight = 1e-5

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        s = spectral_weight * spectral_loss(x0)
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64') + s.astype('float64')

    def eval_grad(x0):
        s = spectral_weight * np.real(spectral_grad(x0))
        ss = np.empty((1, 3, IMAGE_W, IMAGE_W))
        ss[0, 0, :, :] = s
        ss[0, 1, :, :] = s
        ss[0, 2, :, :] = s
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64') \
            + ss.flatten().astype('float64')

    # Initialize with a noise image
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)
    # Optimize, saving the result periodically
    for i in range(15):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, m=100, maxfun=40)
        # scipy.optimize.fmin_cg(eval_loss,x0.flatten(), fprime=eval_grad)
        x0 = generated_image.get_value().astype('float64')
        PIL.Image.fromarray(deprocess(x0, MEAN_VALUES)).save(
            path.split('.')[0]+str(i)+".png")
        xs.append(x0)

if __name__ == '__main__':
    main(sys.argv[1])
