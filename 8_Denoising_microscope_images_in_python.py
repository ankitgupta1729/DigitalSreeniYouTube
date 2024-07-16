# digital filters are nothing but the convolution between the kernel and the image
# a kernel is a small matrix

# Typically non-linear filters preserve edges and median filters also preserve the edges
# but also cleans up the image

# my favorite denoising filter is non-local means

from skimage import io,img_as_float
from scipy import ndimage as nd
import matplotlib.pyplot as plt

img=img_as_float(io.imread("images/denoising/noisy_img.jpg"))
gaussian_img=nd.gaussian_filter(img,sigma=5) # not a very good filter because edges are not
# preserved..though image gets cleaned but becomes blur image
plt.imsave("images/denoising/gaussian.jpg",gaussian_img)

# median filter

median_img=nd.median_filter(img,size=5) # size of window=3
plt.imsave("images/denoising/median.jpg",median_img)

# non-local means filter

import numpy as np
from skimage.restoration import denoise_nl_means,estimate_sigma
sigma_est=np.mean(estimate_sigma(img,channel_axis=-1))
nlm =denoise_nl_means(img,h=1.15*sigma_est,channel_axis=-1,fast_mode=True,patch_size=5,patch_distance=3)
plt.imsave("images/denoising/nlm.jpg",nlm)