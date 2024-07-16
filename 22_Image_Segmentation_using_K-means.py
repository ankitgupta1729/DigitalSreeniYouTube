# we have image in which pixels have different color/intensities and we need to
# partition/segment it.

# for the given image, we can make 4 or 5 clusters based on 4 or 5 different colors in gray image

import cv2
import numpy as np

img=cv2.imread("images/BSE_Image.jpg")

# Though it looks gray but it is a rgb image because it has 3 channels
# By default cv2 has image of BGR space.
print(img.shape)
# cv2.imshow("original image",img)
# cv2.waitKey(0)

# for k-means, we need to flat this image i.e. shape of 751000

img2=img.reshape((-1,3)) # -1 takes care of dimension of flat array
print(img2.shape)
print(img2.dtype)

# k-means clustering in opencv allows image in only float32 format. so we need to change it.
# see the documentation on google

img2 = np.float32(img2)
print(img2.dtype)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # max iteration=10,epsilon=1

# no. of clusters
k=4

attempts=10

ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

# ret (return) is compactness means sum of squares of distances from each point to corresponding
# centres

# here center has 4*3 matrix because we have 4 clusters and 3 channels

print(center)

# converts centers into unsigned integers as per documentation

center=np.uint8(center)
print(center.dtype)

res=center[label.flatten()]
res2=res.reshape((img.shape))
print(res2.shape)

# save the file
cv2.imwrite("Segmented_Image.jpg",res2)
cv2.imshow("Segmented Image",res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

