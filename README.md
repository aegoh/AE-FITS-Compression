# AE-FITS-Compression
Performing image compression on FITS images using autoencoder architecture

Bottleneck layer (latent space) is passed into two decoder models. One decoder model predicts the image ‘mean’, where the objective function is the mean squared error. The other decoder model predicts the ‘variance’, where the objective function is the log likelihood. The two decoder models are trained independently from each other with two separate optimizers.


## 1D_tunable_mnist/toy 
