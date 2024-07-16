# Histogram equalization means stretch the histogram to cover entire range

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("images/Alloy.jpg",0)
eq_img=cv2.equalizeHist(img)

plt.hist(img.flat,bins=100, range=(0,255))
plt.hist(eq_img.flat,bins=100, range=(0,255))
#plt.show()
# cv2.imshow("Equalized Image",eq_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# There are much noise in the eqaulized image because it stretches the histogram
# so histogram equalization considers the global contrast of the image not the
# local contrast

# So performing a global equalization may not work on your images.

# So there is something called adaptive histogram equalization. It does histogram
# equalization in small patches or small tiles. Also, it has algorithm called
# contrast limiting adaptive histogram equalization (CLAHA)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img=clahe.apply(img)
#cv2.imshow("Equalized Image",eq_img)
#cv2.imshow("CLAHE",cl_img)
plt.hist(cl_img.flat,bins=100, range=(0,255))
ret,thresh1=cv2.threshold(cl_img,190,150,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(cl_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Original",img)
cv2.imshow("Binary Threshold 1 ",thresh1)
cv2.imshow("OTSU ",thresh2)

#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# IF image has noise then first denoise it and then apply above process


