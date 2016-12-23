
import lasagne
import numpy as np
import scipy
import PIL.Image
import sys

import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from lasagne.utils import floatX

from fourier_loss import define_fourier_loss
from multiscale import build_model

IMAGE_W = 600


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
FILTER_SIZES = [3,5,7,11,15,23,37,55]
def main(path):
    net = build_model(IMAGE_W)

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

    layers = [str(i) for i in FILTER_SIZES]
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
    for i in FILTER_SIZES:
        losses.append(1e7/len(FILTER_SIZES) * style_loss(art_features, gen_features, str(i)))


    # total variation penalty
    losses.append(0.1e-7 * total_variation_loss(generated_image))

    total_loss = sum(losses)
    grad = T.grad(total_loss, generated_image)

    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
   
   # spectral loss and gradient
   spectral_loss = define_fourier_loss(art[0])
   
    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        s = spectral_loss(x0)
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64') + s

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')

    # Initialize with a noise image
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

    x0 = generated_image.get_value().astype('float64')

    # Optimize, saving the result periodically
    for i in range(8):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
        # scipy.optimize.fmin_cg(eval_loss,x0.flatten(), fprime=eval_grad)
        x0 = generated_image.get_value().astype('float64')
        outpath = path.split('.')[0]
        PIL.Image.fromarray(deprocess(x0)).save(outpath+str(i)+".png")

if __name__ == '__main__':
    main(sys.argv[1])
