# AE-FITS-Compression
Performing image compression on FITS images using autoencoder architecture using the tensorflow framework.

Encoder is the map from the image $\hat{x}$ to latent vector $\hat{z}$,  $$f: \hat{x} \rightarrow \hat{z}$$

The latent space $\hat{z}$ is passed into two decoder models in the forward pass. One decoder model predicts the image ‘mean’, where the objective function to be minimized is the mean squared error. The other decoder model predicts the ‘variance’, where the objective function is the log likelihood. The two decoder models are trained independently from each other with two separate optimizers.

Mean Decoder is the map from the latent vector to image mean $$g_{\mu}: \hat{z} \rightarrow \hat{\mu}$$ 

Variance Decoder is the map from the latent vector to variance map $$g_{\sigma}: \hat{z} \rightarrow \hat{\sigma}$$


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

If the images are large, with dimensions (256,256,1), then the number of model parameters will blow up if we have a fully connected layer at the bottleneck. A solution to this is to implement a fully convolutional model, so the latent space has the shape (num_rows, num_columns, num_channels). The proposed tunable compression scheme is similar to the one presented in 1D_tunable_mnist/toy.ipynb    

$$\eta \rightarrow m_{ijk} = \ \begin{cases} 
      1 & \text{for all i,j if } k \leq \eta \\
      0 & \text{for all i,j if } k > \eta
   \end{cases}
\ $$

where i,j indicates the row and column of the latent space and k indicates the channel. This method seems to behave unexpectedly; it does not perform the best at 1x compression (no compression) but at a specific compression ratio.   

## beta-vae.ipynb 

An attempt to implement the beta parameterized variational autoencoder (BVAE) from the article, Machine-Learning Compression for Particle Physics Discoveries: https://arxiv.org/abs/2210.11489

code: https://github.com/erichuangyf/EMD_VAE/tree/neurips_ml4ps

It does not work as expected so there could be an error in the implementation. 

