import numpy as np
from lasagne.utils import floatX
import skimage.transform


def prep_image(im, mean_values, image_w=600):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (image_w, w*image_w/h),
                                      preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*image_w/w, image_w),
                                      preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-image_w//2:h//2+image_w//2, w//2-image_w//2:w//2+image_w//2]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - mean_values
    return rawim, floatX(im[np.newaxis])


def deprocess(x, mean_values):
    x = np.copy(x[0])
    x += mean_values

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

    x = np.clip(x, 0, 255).astype('uint8')
    return x
