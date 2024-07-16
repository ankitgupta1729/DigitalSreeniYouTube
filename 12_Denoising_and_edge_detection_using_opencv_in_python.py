import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("images/BSE_Google_noisy.jpg",1)

#denoising and bluring functions
kernel= np.ones((3,3),np.float32)/9
filt_2D=cv2.filter2D(img,-1,kernel)
blur=cv2.blur(img,(3,3))
gaussian_blur=cv2.GaussianBlur(img,(3,3),0)
median_blur=cv2.medianBlur(img,3)

# my favorite is bilateral because because it is good for noise removal and retain edges
# better than median

bilateral_blur=cv2.bilateralFilter(img,9,75,75)

# cv2.imshow("Original",img)
# cv2.imshow("2D Custom Filter",filt_2D)
# cv2.imshow("Blurr",blur)
# cv2.imshow("Gaussian Blurr",gaussian_blur)
# cv2.imshow("Median Filter",median_blur)
# cv2.imshow("Bilateral Blurr",bilateral_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Edge Detection functions

img=cv2.imread("images/Neuron.jpg",0)

edges=cv2.Canny(img,100,200)

cv2.imshow("Canny",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

