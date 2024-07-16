# If we have regions that are visibly kind of different but they are very noisy, so
# we can't use histogram based segmentation because peaks in the histograms are not separable

# In this situation we have many tools and one tool is random walker segmentation

from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

img = img_as_float(io.imread("images/Alloy_noisy.jpg"))

# plt.hist(img.flat, bins=100, range =(0,1))
# plt.show()

# here histogram is overlapped but from image is clear that regions are seperated. so we
# can't use histogram based segmentation

# there are many ways of denoising but we have to preserve texture, edges

sigma_est = np.mean(estimate_sigma(img, channel_axis=None))
patch_kw = dict(
    patch_size=5, patch_distance=6, channel_axis=None  # 5x5 patches  # 13x13 search area
)
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)

from skimage import exposure

eq_image = exposure.equalize_adapthist(denoise_img)

#plt.hist(eq_image.flat, bins=100, range =(0,1))
#plt.show()

#plt.imshow(eq_image, cmap='gray')
#plt.show()

# Random Walker Segmentation

markers= np.zeros(img.shape,dtype=np.uint)

markers[(eq_image<0.6) & (eq_image>0.3)]=1
markers[(eq_image>0.8) & (eq_image<0.99)]=2
#
# plt.imshow(markers)
# plt.show()

# defining random walker

from skimage.segmentation import  random_walker

labels = random_walker(eq_image,markers,beta=10,mode='bf')
# plt.imshow(labels)
# plt.show()

seg1=labels==1
seg2=labels==2

all_segments=np.zeros((eq_image.shape[0],eq_image.shape[1],3))

all_segments[seg1]=(1,0,0)
all_segments[seg2]=(0,1,0)

plt.imshow(all_segments)
plt.show()