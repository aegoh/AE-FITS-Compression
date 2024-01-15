# AE-FITS-Compression
Performing image compression on FITS images using autoencoder architecture using the tensorflow framework.

Encoder is the map from the image $\hat{x}$ to latent vector $\hat{z}$,  $$f: \hat{x} \rightarrow \hat{z}$$

Bottleneck layer (latent space) is passed into two decoder models. One decoder model predicts the image ‘mean’, where the objective function to be minimized is the mean squared error. The other decoder model predicts the ‘variance’, where the objective function is the log likelihood. The two decoder models are trained independently from each other with two separate optimizers.

Mean Decoder is the map from the latent vector to image mean $$g_{\mu}: \hat{z} \rightarrow \hat{\mu}$$ 

Variance Decoder is the map from the latent vector to variance map $$g_{\sigma}: \hat{z} \rightarrow \hat{\sigma}$$


## 1D_tunable_mnist/toy.ipynb

mnist: Using mnist digits as dataset.

toy: Using simulated data as dataset.

The latent space is a 1D vector, where the bottleneck layer is a fully connected layer. There is a tunable parameter at prediction that masks a fraction of the latent space when passing into the decoder models. The mask vector $m_i$ can be determined by some parameter $\eta$ 

$$\eta \rightarrow m_i = \ \begin{cases} 
      1 & i \leq \eta \\
      0 & i > \eta
   \end{cases}
\ $$

where $dim(m_i) = dim(\hat{z})$.


