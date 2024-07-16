# Feature Descriptors and Key Points

# key point is a region of interest
# feature detector detects these key points
# feature descriptor describes these key points. It could be how surrounding looks like

# Once we identify these key points, we can use those key points to transform the image back to
# reference image. So registration would be easily guided by these key points

# Harris Corner Detection

import cv2
import numpy as np

# img = cv2.imread("images/grains.jpg")
#
# # harris corner detection algo works on gray float32
#
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray=np.float32(gray)
# harris=cv2.cornerHarris(gray,2,3,0.04)
# img[harris>0.01*harris.max()]=[255,0,0]
# cv2.imshow('Harris',img) # detect corner/key points
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# I am not a big fan of harris so let's move on something else

###########################################################################################

# Shi-Tomasi Corner detector -- Good feature to track
# img = cv2.imread("images/grains.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# # corners is nothing but all the corners in the form of coordinates
# corners=np.int8(corners)
#
#
# for i in corners:
#     x,y=i.ravel()
#     print(x,y)
#     cv2.circle(img,(x,y),3,255,-1)
#
# cv2.imshow('Corners',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# These are key points detectors, not descriptors

#####################################################################################

# SIFT (scale invariant feature transform) and SURF
# it is both detector and descriptor
# It has information about the key points and also information about description of those key points

####################################################################################

# FAST (Features from accelerated segmented test) algorithm for corner detection

# it is a feature detector, not feature descriptor

# But with any feature detector, we also have to use feature descriptor

img=cv2.imread("images/grains.jpg",0)

# #initiate FAST object with default values
# detector=cv2.FastFeatureDetector_create(50) # detect 50 points
# kp=detector.detect(img,None) # kp means key points
# img2=cv2.drawKeypoints(img,kp,None,flags=0)
# cv2.imshow('Corners',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################################################################

# BRIEF (Binary Robust Independent Elementary Features)

# it is a feature descriptor. So we need a feature detector also.

###################################################################################

#My favorite is:

## ORB (Oriennted FAST and Rotated BRIEF)

# it is a combination of detector and descriptor

# SIFT was used extensively in past

img=cv2.imread("images/grains.jpg",0)

orb=cv2.ORB_create(50)
kp,des=orb.detectAndCompute(img,None)
img2=cv2.drawKeypoints(img,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Here, circle size reflects the strength of that key point and angle reflects the
# direction of the key point based on the  neighboring pixels
cv2.imshow("ORB",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()