# Plan: We make a one single python file to train the images and once we have
# created a machine learning model then the next step would be to segment the
# unknown images.

# Here we have images of sample micro CT scans or x-ray microscope scans of a sandstone.
# Here in image, Bright region has some heavy material and gray region has some quarks or
# SiO2 ans texture region is some sort of clay.

# We have 500 images and manual segmentation means going through every slice and validate it
# it is not an easy task or automated by histogram or watershed is also not promising.

#################################################################################################

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img=cv2.imread('images/Train_images/Sandstone_Versa0000.tif')
print(type(img))
print(img.shape)
print(img.dtype)

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(type(img))
print(img.shape)
print(img.dtype)

df=pd.DataFrame()

# The most important feature would be the pixel values. For example, if pixel value>400,
# then it might be bright pixel.

# Add original pixel values to the dataframe as feature #1
img2=img.reshape(-1)
df['Original Image']=img2

# Add other features

# Set-1. Gabor Features (my favorite)- It is like a Gaussian or canny edge detection filters

num=1
kernels=[]
for theta in range(2):
    theta=(theta/4)*np.pi
    for sigma in range(1,3):
        for lamda in np.arange(0,np.pi,np.pi/4):
            for gamma in (0.05,0.5): # 0.05 gives high aspect ratio kernel and 0.5 gives low aspect ratio kernel
                #print(theta,sigma,lamda,gamma)
                gabor_label='Gabor'+str(num)
                ksize=5
                kernel= cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
                kernels.append(kernel)
                fimg=cv2.filter2D(img,cv2.CV_8UC3,kernel) # 2D convolved image
                filtered_img=fimg.reshape(-1)
                df[gabor_label]=filtered_img
                num+=1

print(df.head())

#####################################################################################

# Set 2: Canny Edge Detector

edges=cv2.Canny(img,100,200)
edges1=edges.reshape(-1)
df['Canny Edge']=edges1

#####################################################################################

# set 3: other filters

from skimage.filters import roberts,sobel,scharr,prewitt

edge_roberts=roberts(img)
edge_roberts1=edge_roberts.reshape(-1)
df['Roberts']=edge_roberts1

edge_sobel=sobel(img)
edge_sobel1=edge_sobel.reshape(-1)
df['Sobel']=edge_sobel1

edge_scharr=scharr(img)
edge_scharr1=edge_scharr.reshape(-1)
df['Scharr']=edge_scharr1

edge_prewitt=prewitt(img)
edge_prewitt1=edge_prewitt.reshape(-1)
df['Prewitt']=edge_prewitt1

# Gaussian with sigma=3

from scipy import ndimage as nd
gaussian_img=nd.gaussian_filter(img,sigma=3)
gaussian_img1=gaussian_img.reshape(-1)
df['Gaussian s3']=gaussian_img1

# Gaussian with sigma=7

from scipy import ndimage as nd
gaussian_img2=nd.gaussian_filter(img,sigma=7)
gaussian_img3=gaussian_img2.reshape(-1)
df['Gaussian s7']=gaussian_img3

# Median with sigma=3

median_img=nd.median_filter(img,size=3)
median_img1=median_img.reshape(-1)
df['Median s3']=median_img1

# # Variance with size=3
#
# variance_img=nd.generic_filter(img,np.var,size=3)
# variance_img1=variance_img.reshape(-1)
# df['Variance s3']=variance_img1

print(df.head())
#######################################################################################

# Now we take the labeled image for ground truth

labeled_img=cv2.imread('images/Train_masks/Sandstone_Versa0000.tif')
labeled_img=cv2.cvtColor(labeled_img,cv2.COLOR_BGR2GRAY)
labeled_img1=labeled_img.reshape(-1)
df['Label']=labeled_img1

print(df.head())

# now we have done with data handling.

##########################################################################################

# Now, apply thr Random Forest Classifiers

Y= df['Label'].values

X= df.drop(labels=['Label'],axis=1)

# split data into train and test

from  sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

# Import ML algorithm and train the model

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=10, random_state=42)

from sklearn.svm import LinearSVC
model=LinearSVC(max_iter=1000)

model.fit(X_train, Y_train)


prediction_test = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(Y_test, prediction_test))

###############################################################################

# Feature Importance is not part of SVM

## which feature is important ?

#importances=list(model.feature_importances_)
#print(importances)

# feature_list=list(X.columns)
#
# feature_imp=pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
# print(feature_imp)

##################################################################################

# How to pickle the model and use it for the future use

import pickle

filename='sandstone_svm.pkl'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

result=loaded_model.predict(X)

segmented=result.reshape((img.shape))

import matplotlib.pyplot as plt

plt.imshow(segmented,cmap='jet')
plt.imsave('segmented_rock_svm.jpg',segmented,cmap='jet')
plt.show()

##################################################################################

# use saved model and segment all other 500 images


#####################################################################################

# Summary: Random Forest works much better than SVM in terms of running time and performance
# for most of the image segmentation even iterations in svm are increased but it takes much
# time. But for image classification, svm is a good choice not for pixel segmentation.