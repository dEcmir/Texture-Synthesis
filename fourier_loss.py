from numpy.fft import fft2, ifft2
import numpy as np


def define_fourier_loss(tex):
    '''
    returns a loss function computing the distance
    to the spectrum of tex
    Args :
           - tex a numpy array of size 3, w, w. being the generated texture
    output loss (function x0 -> R) where x0 has the same shape as tex
    '''
    # store the fft of the grayscale texture in a variable
    shape = tex.shape
    tex =  _gray_bgr(tex)
    tex = fft2(tex)

    def spectrum_loss(x0):
        # x0 is the generated texture image
        x0 = x0.reshape(shape)
        # we constrain on the spectrum of the grayscale image
        x0 =  _gray_bgr(x0)

        spectrum = fft2(x0)

        ortho = tex * np.conj(spectrum)
        projection = ifft2((ortho / np.linalg.norm(ortho)) * spectrum)
        # return the square norm of the difference
        return 0.5 * np.real(np.square(projection - tex).sum())

    return spectrum_loss


def define_fourier_grad(tex):
    ''' From a color image 'x_0' compute its spectrum constraint gradient
        in regard to the the original image 'tex'
        Args:
            - tex (list): list of the used arms
        Output:
            - fourier_grad (function): x0 -> gradient of the spectrum
                                                   loss
    '''

    # store the fft of the grayscale texture in a variable
    shape = tex.shape
    tex =  _gray_bgr(tex)
    tex = fft2(tex)

    def fourier_grad(x0):
        # x0 is the generated texture image
        x0 = x0.reshape(shape)
        # we constrain on the spectrum of the grayscale image
        x0 = _gray_bgr(x0)

        spectrum = fft2(x0)

        ortho = tex * np.conj(spectrum)
        projection = ifft2((ortho / np.linalg.norm(ortho)) * spectrum)
        return x0 - projection

    return fourier_grad


def _gray_bgr(im):
    return 0.2989 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
