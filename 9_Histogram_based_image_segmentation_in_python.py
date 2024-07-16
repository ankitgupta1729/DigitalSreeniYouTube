# Here we take gray level regions and segment different regions and that's got
# different gray levels

# if we have different regions let's say with different textures but the gray level is the
# same then there is no way to do it. In fact we have to use some sort of pre-processing
# i.e. some texture based identification may be entropy filter or some filter to separate
# the regions before we can do the histogram based segmentation.

# Sometimes we may have to use machine learning to segment the image but for most images,
# with preprocessing, we can actually use histograms based segmentation to separate them
#

from skimage import io,img_as_float,img_as_ubyte
from scipy import ndimage as nd
import matplotlib.pyplot as plt

img=img_as_float(io.imread("images/BSE_Google_noisy.jpg"))

# non-local means filter

import numpy as np
from skimage.restoration import denoise_nl_means,estimate_sigma
sigma_est=np.mean(estimate_sigma(img,channel_axis=None))
denoise =denoise_nl_means(img,h=1.15*sigma_est,channel_axis=None,fast_mode=True,patch_size=5,patch_distance=3)
denoise_ubyte=img_as_ubyte(denoise)
#plt.imshow(denoise_ubyte,cmap='gray')
#plt.hist(denoise_ubyte.flat,bins=100, range=(0,255))

seg1= (denoise_ubyte<=55) # it is a binary image where all pixels has value <=55
seg2= (denoise_ubyte>55) & (denoise_ubyte<=110)
seg3= (denoise_ubyte>110) & (denoise_ubyte<=210)
seg4= (denoise_ubyte>210)

all_segments=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))
all_segments[seg1]=(1,0,0)
all_segments[seg2]=(0,1,0)
all_segments[seg3]=(0,0,1)
all_segments[seg4]=(1,1,0)
#plt.imshow(all_segments)
#plt.show()


# cleaning image
np.ones((3,3))
from scipy import ndimage as nd
seg1_opened = nd.binary_opening(seg1,np.ones((3,3)))
seg1_closed = nd.binary_closing(seg1,np.ones((3,3)))
seg2_opened = nd.binary_opening(seg2,np.ones((3,3)))
seg2_closed = nd.binary_closing(seg2,np.ones((3,3)))
seg3_opened = nd.binary_opening(seg3,np.ones((3,3)))
seg3_closed = nd.binary_closing(seg3,np.ones((3,3)))
seg4_opened = nd.binary_opening(seg4,np.ones((3,3)))
seg4_closed = nd.binary_closing(seg4,np.ones((3,3)))

all_segments_cleaned=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))
all_segments_cleaned[seg1_closed]=(1,0,0)
all_segments_cleaned[seg2_closed]=(0,1,0)
all_segments_cleaned[seg3_closed]=(0,0,1)
all_segments_cleaned[seg4_closed]=(1,1,0)

plt.imsave("images/segmented_new.jpg",all_segments_cleaned)
