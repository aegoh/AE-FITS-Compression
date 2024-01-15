# AE-FITS-Compression
Performing image compression on FITS images using autoencoder architecture

Bottleneck layer (latent space) is passed into two decoder models. One decoder model predicts the image ‘mean’, where the objective function to be minimized is the mean squared error. The other decoder model predicts the ‘variance’, where the objective function is the log likelihood. The two decoder models are trained independently from each other with two separate optimizers.


## 1D_tunable_mnist/toy.ipynb

mnist: Using mnist digits as dataset.

toy: Using simulated data as dataset.

The latent space is a 1D vector, where the bottleneck is a fully connected layer. The compression ratio is dim(Image) / dim (latent_vector). There is a tunable parameter at prediction that masks a fraction of the latent space when passing into the decoder models.  
