# In the context of image processing and computer vision, these filters are used for
# texture analysis and edge detection and feature extraction

# These Gabor Filters are bandpass filters means they allow certain band of frequencies
# and reject other types

# You can think Gabor as gaussian and it can be 3D or multidimensional.

# It is a function of many parameters.

# suppose, (x,y) denotes the kernel size, theta=angle, lambda= wavelength, gamma= aspect ratio
# psi=

# g(x,y; sigma,theta,lambda,gamma,psi) =exp(-(x^2+gamma^2*y^2)/(2*sigma^2))*exp(i*((2*pi*x/lambda)+psi))

# g(.) stands for gabor

import numpy as np
import cv2
import matplotlib.pyplot as plt

ksize = 5 # it depends on image size and feature size
sigma=3
theta=1*(np.pi/4)
lamda =1*(np.pi/4)
gamma=0.5 # between 0 and 1 and when gamma=1 then it becomes circular
phi=0
# we can make filter bank and check various values of these parameters
# all these value depends what we want to extract from the image

kernel= cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)

# ktype means once the kernel is generated how to store the numbers. 32F =float 32

# we have generated a gabor kernel

# visualize

#plt.imshow(kernel)
# plt.show()

# img=cv2.imread("images/BSE_Image.jpg")
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # cv2.imshow("Original",img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# fimg=cv2.filter2D(img,cv2.CV_8UC3,kernel)
# #resize the kernel
#
# kernel_resized=cv2.resize(kernel,(400,400))
#
# cv2.imshow("Original",img)
# cv2.imshow("Filtered Image",fimg)
# cv2.imshow("Resized Kernel",kernel_resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img=cv2.imread("images/synthetic.JPG")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow("Original",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fimg=cv2.filter2D(img,cv2.CV_8UC3,kernel)
#resize the kernel

kernel_resized=cv2.resize(kernel,(400,400))

cv2.imshow("Original",img)
cv2.imshow("Filtered Image",fimg)
cv2.imshow("Resized Kernel",kernel_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
