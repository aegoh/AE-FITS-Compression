# AE-FITS-Compression
Performing image compression on FITS images using autoencoder architecture using the TensorFlow framework.

Encoder is the map from the image $\hat{x}$ to latent vector $\hat{z}$,  $$f: \hat{x} \rightarrow \hat{z}$$

The latent space $\hat{z}$ is passed into two decoder models in the forward pass. One decoder model predicts the image ‘mean’. The mean decoder is the map from the latent vector to image mean $$g_{\mu}: \hat{z} \rightarrow \hat{\mu}$$ 

where the objective function to be minimized is the mean squared error:

$$C_{mse} = [\hat{x} - g_{\mu}(f(\hat{x}))]^2$$

The other decoder model predicts the ‘variance’. The variance decoder is the map from the latent vector to variance map $$g_{\sigma}: \hat{z} \rightarrow \hat{\sigma^2}$$

where the objective function is the log likelihood. 

$$C_{l} = \log(g_{\sigma}(f(\hat{x})) + \epsilon) + \frac{[\hat{x} - g_{\mu}(f(\hat{x}))]^2}{g_{\sigma}(f(\hat{x}))+\epsilon}$$

where $\epsilon$ is a small number to prevent divergence in the loss while training. The two decoder models are trained independently from each other with two separate optimizers.



## 1D_tunable_mnist/toy.ipynb

mnist: Using mnist digits as dataset.

toy: Using simulated data as dataset.

The latent space is a 1D vector, where the bottleneck layer is a fully connected layer. I propose a tunable compression scheme where a tunable parameter is specified manually at prediction that masks a fraction of the latent space when passing into the decoder models. The mask vector $m$ can be determined by some parameter $\eta$ 

$$\eta \rightarrow m_i = \ \begin{cases} 
      1 & i \leq \eta \\
      0 & i > \eta
   \end{cases}
\ $$

where $m_i$ denote the components of $m$ and $dim(m) = dim(\hat{z})$

Given a specific parameter $\eta$ that yields a corresponding mask vector $m$, a forward pass image mean prediction of the model is $\hat{\mu} = g_{\mu}(m * f(\hat{x}))$ where $*$ indicates the element-wise product. During training, we initialize a random value for the $\eta$ parameter for each training batch in the forward pass and fit for the objective functions. A trained ideal model is expected to approximate principal component analysis (PCA), where $z_1$ is the first principal component. 

## fullyconv_tunable_toy.ipynb 

If the images are large, with dimensions (256,256,1), then the number of model parameters will blow up if we have a fully connected layer at the bottleneck. A solution to this is to implement a fully convolutional model, so the latent space has the shape (num_rows, num_columns, num_channels). The proposed tunable compression scheme for a fully convolutional model is similar to the one presented in 1D_tunable_mnist/toy.ipynb    

$$\eta \rightarrow m_{ijk} = \ \begin{cases} 
      1 & \text{for all i,j if } k \leq \eta \\
      0 & \text{for all i,j if } k > \eta
   \end{cases}
\ $$

where i,j indicates the row and column of the latent space and k indicates the channel. This method seems to behave unexpectedly; it does not perform the best at 1x compression (no compression) but instead at seemingly arbitrary specific compression ratio.   

## beta-vae.ipynb 

An attempt to implement the beta parameterized variational autoencoder (BVAE) from the article, Machine-Learning Compression for Particle Physics Discoveries: https://arxiv.org/abs/2210.11489

code: https://github.com/erichuangyf/EMD_VAE/tree/neurips_ml4ps

It does not work as expected so there could be an error in the implementation. 

## fullyconv_denoise.ipynb

Implementation of a fully convolutional denoising autoencoder on toy dataset. The tuning scheme is implemented but has not been tested.    

## ZTF_stamp_download.ipynb 

Uses the package ztfquery to download random raw ZTF images and cut out stamps of size 256x256. Creates an image mask using the astropy and photutils packages that masks parts of the image that only has background noise.     

code: https://github.com/MickaelRigault/ztfquery

## Pixel_AE.ipynb

Pixel Autoencoder with residual connections (https://keras.io/examples/generative/pixelcnn/) with focal frequency loss (paper: https://arxiv.org/pdf/2012.12821.pdf, code: https://github.com/ZohebAbai/tf-focal-frequency-loss/tree/main). Includes comparison with Hcompress algorithm using h-scale compression parameter from the astropy.io.fits.CompImageHDU class (https://docs.astropy.org/en/stable/io/fits/api/images.html) and the fpack program (paper: https://iopscience.iop.org/article/10.1086/656249/pdf, https://heasarc.gsfc.nasa.gov/fitsio/fpack/). 


Training the network on unnormalized 16 bit images (pixel values can range from 0 to the thousands for an image stamp) is inadequate as it is unable to 'recover' signal that is close to the noise background (dim stars). I use a logarithmic (base 10) function to normalize the images and have the autoencoder train on these normalized images instead. Rescaling the predicted mean image to the unnormalized scale and calculating the difference between the raw image still yields large errors (e.g. predicted star is much dimmer than ground truth star). So I attempt to ease this error by adding a second residual block that tries to learn the mapping from the normalized image to the unnormalized image.

<img width="435" alt="Resdual" src="https://github.com/Cuzime/AE-FITS-Compression/assets/74683524/4b152559-aa1a-4c17-8826-2a43f249e603">







