# AIC/BIC to determine number of clusters/segments

# Both helps to find the number of parameters. Both estimate the quality of a model using
# penalty term for number of parameters. Make a graph between number of parameter(x-axis) and
# penalty (y-axis) (not a continuous curve). Pick a elbow point.

import cv2
import numpy as np

img=cv2.imread('images/Alloy.jpg')
img2=img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM
n_components=np.arange(1,10)
gmm_models=[GMM(n,covariance_type='tied').fit(img2) for n in n_components]

import matplotlib.pyplot as plt

plt.plot(n_components,[m.bic(img2) for m in gmm_models],label='BIC')
plt.xlabel('Number of components')
plt.show()

# gmm_labels=gmm_model.predict(img2)
#
# original_shape=img.shape
# segmented=gmm_labels.reshape(original_shape[0],original_shape[1])
# cv2.imwrite('segmented_alloy.jpg',segmented)
