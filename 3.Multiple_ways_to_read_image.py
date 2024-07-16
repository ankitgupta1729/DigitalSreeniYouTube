# we use 4 libraries to load and read images

# 1. Pillow
# 2. Matplotlib
# 3. Scikit-Image
# 4. opencv

# Others: omitive library (open microscopy environment)-- it processes a tiff file which has an
# embedded xml which has all the meta data about a specific image

###################################################################################

# 1. Pillow

# It is an image manipulation and processing library like crop/resize image and
# some sort of basic filtering.

# But for advanced tasks like object detection example from computer vision or ML, we use
# scikit image and opencv.

# For basic taks, pillow is an excellent library

# pip install pillow

from PIL import Image
import  numpy as np

img = Image.open("images/nature.jpg")
print(type(img)) # it is not numpy array, so we can't do image processing and so we have
# to convert into numpy array
#img.show()
print(img.format)

img1=np.asarray(img)
print(img1.shape)
print(img1)

################################################################################

# 2. Maplotlib

# This is not actually an image processing library, it is a plotting library
# Pyplot is a library to plot 2d graphs,images and it is similar to matlab but
# we don't need matlab licence

# pip install matplotlib==3.5.3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread("images/nature.jpg")
print(type(img)) # it is an numpy array
print(img.shape)
#plt.imshow(img)
#plt.colorbar()
#plt.show()

##################################################################################

# 3. Scikit- Image

# It is an image processing library that is used for image segmentation and
# geometric transformation and color space manipulation, analysis, filtering, feature detection
# it is a very good package if we want to do some traditional ML like random forest, svm etc

from skimage import io

img = io.imread("images/nature.jpg")
print(type(img))
#plt.imshow(img)
#plt.show()

##################################################################################

# 4. OpenCV

# it is a library of programming functions that are mainlly aimed at computer vision
# Means it is good for images, videos like if we want to do facial detection, object detection,
# motion tracking, OCR. It is also good for segmentation with deep learning

# pip install opencv-python

# By default cv2 handles images as BGR(blue,green,red) not RGB

import cv2

gray_img=cv2.imread("images/nature.jpg",0) # 0 for gray image and 1 for color image
#cv2.imshow("gray_img",gray_img)

# By default cv2 handles images as BGR(blue,green,red) not RGB but we can convert it
color_img=cv2.imread("images/nature.jpg",1)
#plt.imshow(cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB)) # using plt
#plt.show()
# we can change it into other spaces like HSV (Hue,Saturation,..space)

#cv2.waitKey(0) # 0 means wait until we terminate it, if it is 1000 then it means wait till 1000 milli seconds
#cv2.destroyAllWindows() # it destroys all the created windows

#########################################################################################

# Other files

# CZIfile (Image Format for Microscopes) (5d images..(x,y,z,time))


# pip install czifile

# import  czifile
#
# img= czifile.imread("images/nature.czi")


# OME-TIFF -- 5D image and has embedded xml which has meta data

# pip install apeer-ometiff-library

# from apeer_ometiff_library import io
#
# (pic2,omexml)=io.read_ometiff("images/nature.ome.tif")
#
# print(pic2.shape)
# print(pic2)
# print(omexml)

############################################################################

# Read all the files from a folder

# pip install glob

import glob
path="images/*"

for file in glob.glob(path):
    print(file)
    a=cv2.imread(file)
    print(a)
    c=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    cv2.imshow('Color Image',c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

