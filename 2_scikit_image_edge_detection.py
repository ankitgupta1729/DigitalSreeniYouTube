import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from  skimage.filters import sobel
#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img=cv2.imread("images/nature.jpg",0) # 0 is for gray image and 1 is for color image
img=cv2.resize(img,(500,500))
cv2.imshow("pic",img)
print(img.shape)
img2=sobel(img)
cv2.imshow("edge",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection for the default image

image = camera()
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
