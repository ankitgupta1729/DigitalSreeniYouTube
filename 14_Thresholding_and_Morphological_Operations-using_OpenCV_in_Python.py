import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("images/BSE_Google_noisy.jpg",0)
#plt.hist(img.flat,bins=100, range=(0,255))
#plt.show()

# otsu automatically find the threshold as 100
# otsu is a binary threshold. It finds one best location for thresholding

ret, th=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#print(ret)
#print(th)


# cv2.imshow("Original ",img)
# cv2.imshow("Otsu Image",th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Once we have done thresholding, next operation we probably do is erosion and dilation

# erosion means removing certain pixels  and dilation means adding the pixels.
# no of pixels to remove or add is decided by kernel

kernel = np.ones((3,3),np.uint8)
eroision=cv2.erode(th,kernel,iterations=1)
# dilation=cv2.dilate(eroision,kernel,iterations=2)
dilation=cv2.dilate(eroision,kernel,iterations=1)
opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
#cv2.imshow("Original ",img)
# cv2.imshow("Otsu Image",th)
# #cv2.imshow("eroded image",eroision)
# cv2.imshow("eroded+dilated image",dilation)
# cv2.imshow("opened image",opening) # it is identical to eroded+dilated
# opening operation = erosion followed by dilation
# closing operation (opposite of opening)= dilation followed by erosion
# gradient operation = difference between the dilated image and eroded image
# tophat operation = difference between input image and opened image
# blackhat operation (opposite of tophat)= closed image - input image
# these all are features for cleaning image without
# cv2.waitKey(0)
# cv2.destroyAllWindows()

median=cv2.medianBlur(img,3)
cv2.imshow("Original",img)
cv2.imshow("Thresholded image",th)
cv2.imshow("Median Image",median)
cv2.waitKey(0)
cv2.destroyAllWindows()
# so it has done a good job. All the isolated pixels are cleaned up.


# Summary: use median and non-local filters for cleaning image and if they don't
# improve then use morphological operations like erosion,dilation,gradient etc