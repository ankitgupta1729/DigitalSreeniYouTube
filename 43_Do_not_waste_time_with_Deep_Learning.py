# there are many scenarios like denoising where traditional ML gives good results
# than deep learning

from PIL import Image,ImageOps

img = Image.open(r"images/BSE_Image.jpg").convert("L")
#img1= ImageOps.colorize(img,black='blue',white='red')
#img1.show()

# using opencv
import cv2
gray_img = cv2.imread("images/BSE_Image.jpg",0)
color_img= cv2.applyColorMap(gray_img,cv2.COLORMAP_JET)
#cv2.imshow("Color Image",color_img)
#cv2.waitKey(0)

# Denoising

# Anisotropic Diffusion

import matplotlib.pyplot as  plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion

img = io.imread("images/BSE_noisy.jpg",as_gray=True)
img_filtered= anisotropic_diffusion(img,niter=5,kappa=50,gamma=0.1,option=2)
plt.imshow(img_filtered,cmap='gray')
plt.show()












