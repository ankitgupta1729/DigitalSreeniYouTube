# Feature is a characteristic that describe your data.

# For example, an image has 5 regions to distinguish, so we can use pixel value
# or brightness level to distinguish it. So pixel value can be a great feature.

# Other features can be that describes the edges (edges are thos where pixel value changes)

# other features can be texture, orientation, local contrast.

# So, in case of images, we can think of features as the result of  various filters that
# we apply to this image. Now what filters can be applied ? By seeing the image,
# user can say these features are best one. if we have feature vector of size 10 it means
# we have 10 different features.

# If we know what features are needed then we don't need ML.

# How to define features and feature vectors

# import cv2
# from  skimage.filters.rank import  entropy
# from  skimage.morphology import disk
#
# # img=cv2.imread('images/scratch.jpg')
# # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # # for this image entropy filter is a good feature for segmentation based on texture.
# # # entropy is a measure of disorderness
# #
# # entropy_img = entropy(img,disk(1))
# #
# # cv2.imshow('original img',img)
# # cv2.imshow('entropy image',entropy_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# img=cv2.imread('images/Yeast_Cells.png')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # for this image entropy filter is a good feature for segmentation based on texture.
# # entropy is a measure of disorderness
#
# entropy_img = entropy(img,disk(1))
#
# # cv2.imshow('original img',img)
# # cv2.imshow('entropy image',entropy_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# # # Here entropy is not a good filter
#
# # use gaussian_blur filter
#
# from scipy import ndimage as nd
# from skimage.filters import sobel
#
# gaussian_img = nd.gaussian_filter(img,sigma=3)
# # larger the sigma value the more blurry the image is
#
# sobel=sobel(img)
#
# cv2.imshow('original img',img)
# cv2.imshow('gaussian image',gaussian_img)
# cv2.imshow('sobel image',sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Here sobel filter is a good filter to separate the cells in the image


####################################################################################

# How to create feature vector


import cv2
from  skimage.filters.rank import  entropy
from  skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel
import  pandas as pd

img=cv2.imread('images/scratch.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create 1-d array of image pixels in pandas df

img2=img.reshape(-1)

df=pd.DataFrame()

df['Original Pixel Values']=img2

entropy_img = entropy(img,disk(1))
entropy1=entropy_img.reshape(-1)
df['Entropy']=entropy1

gaussian_img = nd.gaussian_filter(img,sigma=3)
# # larger the sigma value the more blurry the image is
gaussian1=gaussian_img.reshape(-1)
df['Gaussian']=gaussian1

sobel_img=sobel(img)
sobel1=sobel_img.reshape(-1)
df['Sobel']=sobel1

print(df)

# In many situations, Gabol Filter is so useful so we should also have to use it.