# BOVW is used for I mage Classification and not pixel classification

# Image classification means we have to classify a bunch of cat and dog images and it does
# not matter what is happening at pixel level when it comes to this application.
# It says what class does image belongs to.


# Example: In microscopy, whether a malarial cell is infected or a healthy cell.

# BOVW is based on the concept of Bag of Words(BOW)

# In BOVW, each word is a part of image. For example, if we have an image of human face.
# then human face has various key components(key points) and machine needs to extract these
# key points and describe them mathematically. Luckily we have something that can do that i.e.
# key points and descriptors i.e. SIFT, orb etc.

# Once we have key points like nose and descriptors then we need to cluster them into
# various types, for example, we have the eyes from 100s of images then we need to
# cluster them and cluster of noses, cluster of ears. In reality, we don't care
# which feature is what like this is nose or ear as long as this class belongs to this class
# and we do K-Means clustering for automatic clustering. Once we cluster them then we have to
# look at  the frequency of the vocabulary. So,

# Step 1: Find Key points
# Step 2: K-Means Clustering or any clustering
# Step 3: find frequency and create histograms
# Step 4: we need to vectorize and normalize
# Step 5: A machine learning algorithm should discriminate these vectors (positive and negative
# training images using svm or random forest etc)

#############################################################################################33

import cv2
import numpy as np
import os

train_path='images/train_image'
training_names=os.listdir(train_path)

image_paths=[]
image_classes=[]
class_id=0

def imglist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

for training_name in training_names:
    dir=os.path.join(train_path,training_name)
    class_path=imglist(dir)
    image_paths +=class_path
    image_classes +=[class_id]*len(class_path)
    class_id+=1

des_list = []

# BRISK is a good replacement to SIFT. ORB also works but didn't work well for this example.
# SIFT is not available in opencv anymore

brisk=cv2.BRISK_create(30)

for image_path in image_paths:
    im=cv2.imread(image_path)
    kpts,des=brisk.detectAndCompute(im,None)
    des_list.append((image_path,des))


# Stack all the descriptors vertically in a numpy array

descriptors=des_list[0][1]
for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))

descriptors_float=descriptors.astype(float)

from scipy.cluster.vq import kmeans, vq
k=200
voc,variance=kmeans(descriptors_float,k,1)

im_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1

nbr_occurrences=np.sum((im_features>0)*1,axis=0 )
idf=np.array(np.log((1.0*len(image_paths)+1)/(1.0*nbr_occurrences+1)),"float32")


from  sklearn.preprocessing import StandardScaler
stdSlr=StandardScaler().fit(im_features)
im_features=stdSlr.transform(im_features)

from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=50000)
clf.fit(im_features,np.array(image_classes))

import joblib
joblib.dump((clf,training_names,stdSlr,k,voc),"bovw.pkl",compress=3)

########################################################################################

# Validation

"""
All cell images resized to 128 x 128
Images used for test are completely different that the ones used for training.
136 images for testing, each parasitized and uninfected (136 x 2)
104 images for training, each parasitized and uninfected (104 x 2)
"""

import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score  # sreeni
import joblib

# Load the classifier, class names, scaler, number of clusters and vocabulary
# from stored pickle file (generated during training)
clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

# Get the path of the testing image(s) and store them in a list
# test_path = 'dataset/test' # Names are Aeroplane, Bicycle, Car
test_path = 'images/test_images'  # Folder Names are Parasitized and Uninfected
# instead of test if you use train then we get great accuracy

testing_names = os.listdir(test_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0


# To make it easy to list all file names in a directory let us define a function
#
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


# Fill the placeholder empty lists with image path, classes, and add class ID number

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Create feature extraction and keypoint detector objects
# SIFT is not available anymore in openCV
# Create List where all the descriptors will be stored
des_list = []

# BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = brisk.detectAndCompute(im, None)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

# Calculate the histogram of features
# vq Assigns codes from a code book to observations.
from scipy.cluster.vq import vq

test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Scale the features
# Standardize features by removing the mean and scaling to unit variance
# Scaler (stdSlr comes from the pickled file we imported)
test_features = stdSlr.transform(test_features)

#######Until here most of the above code is similar to Train except for kmeans clustering####

# Report true class names so they can be compared with predicted classes
true_class = [classes_names[i] for i in image_classes]
# Perform the predictions and report predicted class names.
predictions = [classes_names[i] for i in clf.predict(test_features)]

# Print the true class and Predictions
print("true_class =" + str(true_class))
print("prediction =" + str(predictions))


###############################################
# To make it easy to understand the accuracy let us print the confusion matrix

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print(cm)

showconfusionmatrix(cm)

################# sreeni ###########################
"""
#For classification of unknown files we can print the predictions
#Print the Predictions 
print ("Image =", image_paths)
print ("prediction ="  + str(predictions))

#np.transpose to save data into columns, otherwise saving as rows

np.savetxt ('mydata.csv', np.transpose([image_paths, predictions]),fmt='%s', delimiter=',', newline='\n')
"""



