from skimage import io
from matplotlib import  pyplot as plt

img=io.imread("images/test_image.jpg",as_gray=True)

from skimage.transform import rescale,resize,downscale_local_mean

# rescaled_img=rescale(img,1.0/4.0,anti_aliasing=True)
# resized_img=resize(img,(200,200),anti_aliasing=True)
# downscaled_img=downscale_local_mean(img,(4,3)) # downscaling by 4*3 block and taking mean for that block
# plt.imshow(downscaled_img)
# plt.show()

# Edge Detection

from skimage.filters import  prewitt,roberts,sobel,scharr

img=io.imread("images/test_image_cropped.jpg",as_gray=True)

# edge_roberts=roberts(img)
# edge_sobel=sobel(img)
# edge_scharr=scharr(img)
# edge_prewitt=prewitt(img)
#
# fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize= (8,8))
# ax=axes.ravel()
# ax[0].imshow(img,cmap=plt.cm.gray)
# ax[0].set_title('Original Image')
# ax[1].imshow(edge_roberts,cmap=plt.cm.gray)
# ax[1].set_title('ROBERTS')
# ax[2].imshow(edge_sobel,cmap=plt.cm.gray)
# ax[2].set_title('Sobel')
# ax[3].imshow(edge_scharr,cmap=plt.cm.gray)
# ax[3].set_title('Scharr')
#
# for a in ax:
#     a.axis('off')
#
# plt.tight_layout()
# plt.show()

# My favorite: Canny Edge Detector and it is found in skimage.features, not skimage.filter
# Output of canny edge detector is a binary image

# Canny is also used for noise reduction, gradient calculation,edge tracking etc
# so a good sigma value is better

from  skimage.feature import canny
#
# edge_canny=canny(img,sigma=3)
# plt.imshow(edge_canny)
# plt.show()

# deconvolution

# it requires original image, a point spread function and it calculates the deconvolution
# All it's doing the sharpening the image

from skimage import restoration
import numpy as np
# first define the point spread function(psf)
# psf is a matrix of numbers and it gives the reason why the images is blurred
# and we know the reason image will be sharpened but here we define as:

# psf= np.ones((3,3))/9 # we are normalizing the matrix by dividing by 9
# deconvolved,_ = restoration.unsupervised_wiener(img,psf)
#
# plt.imsave("images/deconvolved_new.jpg",deconvolved,cmap='gray')


#####################################################################################

# Scratch Essay Analysis or wound healing essay

# A glass slide is a whole bunch of cells and you artificially create a scratch and after
# some time gap gets closed because cells proliferate. We can say cells migrate and grow
# and wound gets healed

import matplotlib.pyplot as plt
from skimage import  io,restoration
from skimage.filters.rank import entropy
from skimage.morphology import disk

img=io.imread("images/scratch.jpg")
ent_img=entropy(img,disk(10))
plt.imshow(ent_img,cmap='gray')
#plt.show()

# segmentation of scratch area and cell area using threshold
from skimage.filters import try_all_threshold


# fig,ax=try_all_threshold(ent_img,figsize=(10,8),verbose=False)
# plt.show() # using this we found that otsu is working well

from skimage.filters import threshold_otsu

thresh = threshold_otsu(ent_img) # a single number

binary = ent_img <=thresh # mask
plt.imshow(binary,cmap='gray')
plt.show()

import numpy as np

#finding area
print("The percent of white region is: ",((np.sum(binary==1))*100)/(np.sum(binary==1)+np.sum(binary==0)))

