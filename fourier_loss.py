from numpy.fft import fft2, ifft2
import numpy as np


def define_fourier_loss(tex):
    '''
    returns a loss function computing the distance
    to the spectrum of tex
    Args :
           - tex a numpy array of size 3, w, w
    output loss (function x0 -> R) where x0 has the same shape as tex
    '''
    # store the fft of the grayscale texture in a variable
    shape = tex.shape
    tex = (tex[0] + tex[1] + tex[2]) / 3
    tex = fft2(tex)

    def loss(x0):
        x0 = x0.reshape(shape)
        # we constrain on the spectrum of the grayscale image
        x0 = (x0[0] + x0[1] + x0[2]) / 3

        spectrum = fft2(x0)

        ortho = tex * np.conj(spectrum)
        projection = ifft2((ortho / np.linalg.norm(ortho)) * spectrum)
        # return the square norm of the difference
        return np.real(np.square(projection - tex).sum())

    return loss


define_fourier_grad(tex):
    # a toi de jouer grand fou