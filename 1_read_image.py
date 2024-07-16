import cv2
from skimage import io
from skimage import img_as_float
from  matplotlib import pyplot as plt
import numpy as np

my_image=io.imread('images/nature.jpg')
print(type(my_image)) # image is a numpy array
print(my_image.dtype) # image is of dtype
print(my_image.shape) # image is of shape
print(my_image[0][16]) # image pixel position (16,0) has value of RGB as [84,79,15]
print(my_image) # pixel (0,0) has value 55,67,0 and so on (check using imageJ tool)

# draw a red box

my_image[10:200,10:200,:]=[255,0,0]
my_image=cv2.resize(my_image,(500,500))
cv2.imshow('image',my_image)


img=cv2.imread("images/nature.jpg",0) # 0 is for gray image and 1 is for color image
img=cv2.resize(img,(500,500))
#cv2.imshow("pic",img)
print(img.shape)

my_float_image=img_as_float(my_image)
print(my_float_image.min(),my_float_image.max())

dark_image=my_image*0.5
print(dark_image.max())

# Creating a random image

random_image=np.random.random([500,500]) # random noise for values between 0 and 1
print(random_image)
cv2.imshow("pic",random_image)

cv2.waitKey(0)
cv2.destroyAllWindows()