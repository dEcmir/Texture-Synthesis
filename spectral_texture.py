
import time
import lasagne
import numpy as np
import scipy
import PIL.Image
import sys

import matplotlib.pyplot as plt

from lasagne.utils import floatX

from utils import prep_image, deprocess
from fourier_loss import define_fourier_loss, define_fourier_grad
from multiscale import build_model

IMAGE_W = 600


MEAN_VALUES = np.array([104, 117, 123]).reshape((3, 1, 1))
FILTER_SIZES = [3, 5, 7, 11, 15, 23, 37, 55]


def main(path):

    art = plt.imread(path)
    art = np.array(art)
    rawim, art = prep_image(art, MEAN_VALUES, image_w=IMAGE_W)

    # spectral loss and gradient
    spectral_loss = define_fourier_loss(art[0])
    spectral_grad = define_fourier_grad(art[0])

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        s = spectral_loss(x0)
        return s.astype('float64')

    def eval_grad(x0):
        # this put the same value for BGR and flatten
        # s = 1e3 * np.repeat(spectral_grad(x0)[np.newaxis,:,:], 3).flatten() 
        s = spectral_grad(x0)
        ss = np.empty((1, 3, IMAGE_W, IMAGE_W))
        ss[0, 0] = s
        ss[0, 1] = s
        ss[0, 2] = s
        return ss.flatten().astype('float64')

    # Initialize with a noise image

    x0 = np.random.uniform(-128,128, (1, 3, IMAGE_W, IMAGE_W)) 
  
    start = time.time()
    # Optimize, saving the result periodically
    for i in range(8):
        print(i)
        print time.time()-start
        x0, f,e  = scipy.optimize.fmin_l_bfgs_b(
                                    eval_loss, x0.flatten(),
                                    fprime=eval_grad,
                                     maxfun=40,
                                    )
        print "function value", f, " x0.mean()=", x0.mean()
        x0 = x0.reshape((1, 3 , IMAGE_W, IMAGE_W))
        outpath = path.split('.')[0]
        PIL.Image.fromarray(deprocess(x0, MEAN_VALUES)).save(outpath+str(i)+".png")

if __name__ == '__main__':
    main(sys.argv[1])
