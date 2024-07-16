# It is used in life sciences and biological sciences

# assay means an experiment which repeats over and over

# Here we use traditional methods not any ML

# Here we don't use the histogram based segmentation because it is a gray image and
# pixel values are almost same

# Entropy is a measure of disorder, so for clean area entropy is low and for other area entropy is high

from skimage.filters.rank import entropy
from skimage.morphology import  disk
from skimage.filters import threshold_otsu
from skimage import io
import matplotlib.pyplot as plt

import  numpy as np

img = io.imread("images/scratch.jpg") # reading images
entropy_img = entropy(img,disk(10)) # applying entropy filter so that we can separate
# the two regions better
thresh=threshold_otsu(entropy_img) # using autothresholding method to find the sweet spot to
# separate these two regions
print(thresh) # a single value
binary=entropy_img <= thresh # true when condition is true o/w false
#plt.imshow(binary) # image segmentation
#plt.show()

# image processing for microscopy is not only about image acquisition or image
# adjustment or image segmentation. It is about to get some information from the image

print(np.sum(binary==1)) # number of pixels in white region

#######################################################################################

import glob

time=0
time_list=[]
area_list=[]

path="images/scratch_assay/*.*"

for file in glob.glob(path):
    img = io.imread(file)
    entropy_img = entropy(img, disk(10))
    thresh=threshold_otsu(entropy_img)
    binary=entropy_img <= thresh
    scratch_area=np.sum(binary==1)
    #print(time,scratch_area)
    time_list.append(time)
    area_list.append(scratch_area)
    time +=1

plt.plot(time_list,area_list,'bo')
plt.show()

from scipy.stats import linregress
print(linregress(time_list,area_list))




