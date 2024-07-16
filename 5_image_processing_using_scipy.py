# scipy is just part of numpy stack. It contains modules for Linear Algebra,
# FFT(Fast Fourier Transformation) in Signal Processing and also image processing
# but scipy is not specifically designed for image processing
import scipy
# scipy is built on top of numpy

from scipy import misc
from skimage import io

from scipy import ndimage

# it does not work
# img = misc.imread("images/monkey.jpg")
# print(img.shape)
# print(img.dtype)

img=io.imread("images/monkey.jpg",as_gray=True)
print(type(img))
print(img.dtype)
print(img.shape)
print(img)

print(img[0,0])

mean_gray=img.mean()
max_gray=img.max()
min_gray=img.min()

print(mean_gray,max_gray,min_gray)

import numpy as np
import matplotlib.pyplot as plt

flippedLR = np.fliplr(img)
flippedUD= np.flipud(img)

# plt.subplot(2,1,1)
# plt.imshow(img,cmap='Greys')
# plt.subplot(2,2,3)
# plt.imshow(flippedLR,cmap='Blues')
# plt.subplot(2,2,4)
# plt.imshow(flippedUD,cmap='hsv')
# plt.show()

# rotated=ndimage.rotate(img,45)
# plt.imshow(rotated,cmap='gray')
# plt.show()

# Filters from scipy

# uniform_filtered=ndimage.uniform_filter(img,size=9) # it is a kind of blurring filter
# plt.imshow(uniform_filtered)
# plt.show()

# Another type of blurring filter is Gaussian Filter. It smoothes the noise but be careful
# because if you apply heavily then edges will be gone

# gaussian_filtered=ndimage.gaussian_filter(img,sigma=3)
# plt.imshow(gaussian_filtered)
# plt.show()

# These filtering algorithms are denoising algorithms. They work like that if we have a 3*3 patch
# One way it works as it does averages of 3*3 values and replaces all the pixel values.
# Gaussian filter blurs the edges and does not preserve the edges. Median filter better one.

# median_filtered=ndimage.median_filter(img,3) # try it on some noisy image from internet
# plt.imshow(median_filtered)
# plt.show()

# sobel filter for edge detection

sobel_img=ndimage.sobel(img,axis=1)
plt.imshow(sobel_img)
plt.show()
