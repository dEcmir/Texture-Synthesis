# Texture Synthesis
code from a small project on the topic of texture generation

### Prerequisites

The following Python packages are required :
  - Theano
  - Lasagne
  - Pillow
  - Matplotlib
  - Numpy
  
 Although it is not required it is better to have CUDNN in conjunction with Theano. If not, replace 
 `from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer`
 by `from lasagne.layers import Conv2DLayer as ConvLayer` in the network definition files.
 
 ### Usage 
 
 Both scripts *vgg_texture_cnn.py* and *texture_cnn.py* are used thusly :
 `python <scipt.py> <image_file>`
 and produce a few images to show the convergence of the L_BFGS method.
 
 
### Organisation

#### Scripts


 - _texture_cnn.py_ texture generation using the shallow network
 - _vgg_texture_cnn.py_ implementation of the texture generation using VGG
 - _spectrum_texture_cnn.py_	shallow network + spectrum
 - _spectrum_vgg_texture_cnn_2.py_	vgg network + spectrum
 - _spectral_texture.py_	variational texture method (for test)
 
 #### Utils
 - _vgg.py_ definition of VGG19 using *Lasagne*
 - _fourier_loss.py_ definition of the spectral loss
 - _multiscale.py_ definition of a shallow network with different filter sizes using *Lasagne*
 - _texture_cnn.py_ algorithm with both spectral loss and Gram loss 
 - _utils.py_

 

### Acknowledgements

We took inspiration from https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
