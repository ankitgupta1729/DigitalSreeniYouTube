# K-Means is unsupervised segmentation

# Supervised segmentation means we can actually supply a labelled image where we can paint the
# pixels and label them and this is my ground truth and use this for training the algorithm
# and in future for other images, we can use the trained model to segment future images.

# In case of image segmentation or image processing, data is pixels.

# Gaussian Mixture Model is also a clustering technique like K-Means but it has a variation.

# It comes under unsupervised that is used for data segmentation or image segmentation or
# image processing

# K-Means has some limitation because it's a hard assignment of data points to the centroids
# but what if clusters overlap. How to deal with it. So apparently it looks like
# Gaussian Mixture Model works pretty well in these scenarios.

# Suppose, an image has 4 regions which have different gray levels then we are lucky because
# they are separable and we can make histogram easily and there are 4 different peaks which
# are separable and we can use watershed and other algorithms.

# GMM assumes gausians in image histograms means image histogram is made of different gausian
# distributions that are mixed at different ranges. So, all gaussians are mixed and each
# gaussian has centre and standard deviation (width) and the height/weight.

# It is a probabilistic model which is a combination of individual gaussians with unknown
# parameters. Here we have mentioned 3 parameters i.e. centre,width and height.

# In GMM, we have the probability for each datapoint. Suppose there are 2 overlapping
# Gaussians blue and green. Data point which is near to green has higher probability
# that it is generated from green(say 98%) than blue (say 2%).
# This is not a hard assignment, k-means is a hard assignment.
# But like K-Means, mean,variance and weights get updated. Here all data points are
# weighted by probability and like K-Means, it takes the average

# GMM uses expectation maximization. Expectation finds the probability and maximization
# means updating the parameters i.e. center(mean),width(variance) and height(weight) after
# each iteration.

# import numpy as np
# import cv2
#
# img=cv2.imread('images/plant_cells4.jpg')
# img2=img.reshape((-1,3))
#
# from sklearn.mixture import GaussianMixture as GMM
#
# gmm_model=GMM(n_components=2,covariance_type='tied').fit(img2)
# gmm_labels=gmm_model.predict(img2)
#
# original_shape=img.shape
# segmented=gmm_labels.reshape(original_shape[0],original_shape[1])
# cv2.imwrite('segmented_plant.jpg',segmented)
#
# # the saved image has 2 levels so it will be shown as black..
#
# # open imageJ tool--> open segmented plant image --> go to image--> adjust --> Brightness &Contrast
# # --> set --> min=0 and max=2


import numpy as np
import cv2

img=cv2.imread('images/BSE_Image.jpg')
img2=img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM

gmm_model=GMM(n_components=4,covariance_type='tied').fit(img2)
gmm_labels=gmm_model.predict(img2)

original_shape=img.shape
segmented=gmm_labels.reshape(original_shape[0],original_shape[1])
cv2.imwrite('segmented_rock.jpg',segmented)
