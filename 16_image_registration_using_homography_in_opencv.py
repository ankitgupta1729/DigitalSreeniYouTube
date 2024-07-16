# Here we have a reference image and deformed image and we transform it and match
# with the reference image

"""
1. import 2 images
2. convert to grayscale image
3. Initiate ORB detector
4. Find key points and describe them
5. Match Keypoints between 2 images (we will use BruteForce Matcher Algorithm)
6. We will separate good and bad points using RANSAC Algorithm
7. Register 2 images (use homology)
"""

import cv2
import numpy as np

# img=cv2.imread("images/monkey.jpg")
# img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# orb=cv2.ORB_create(50)
# kp,des=orb.detectAndCompute(img1,None)
# img2=cv2.drawKeypoints(img1,kp,None,flags=None)
# cv2.imshow("ORB",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##################################################################

im1=cv2.imread("images/monkey_distorted.jpg") # image to be registered
im2=cv2.imread("images/monkey.jpg") # reference image

img1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)


# initiate ORB
orb=cv2.ORB_create(50)

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)


# Bruteforce matcher takes the descriptor of one feature from first set(first image)
# and it matches with all other features from set 2 (second image)
# then does some distance calculation

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(des1,des2,None)
matches=sorted(matches, key = lambda x:x.distance)

#img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None)

# cv2.imshow("Matches",img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

points1=np.zeros((len(matches),2),np.float32)
points2=np.zeros((len(matches),2),np.float32)

# RANSAC
for i,match in enumerate(matches):
    points1[i,:]=kp1[match.queryIdx].pt # storing coordinates of key points of image 1
    points2[i,:]=kp2[match.trainIdx].pt # storing coordinates of key points of image 2

# h= homography matrix(transformation matrix): img1*h=img2
h,mask=cv2.findHomography(points1,points2,cv2.RANSAC,5.0)

# Use homography

height,width,channels=im2.shape

im1Reg=cv2.warpPerspective(im1,h,(width,height))

img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None)
cv2.imshow("Key Points Matches",img3)
cv2.imshow("Registered Image",im1Reg)
cv2.waitKey(0)
cv2.destroyAllWindows()