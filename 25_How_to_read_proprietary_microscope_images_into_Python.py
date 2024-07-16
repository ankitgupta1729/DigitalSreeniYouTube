# These are difficult to read because it involves another dimension i.e. time

import czifile

img= czifile.imread('images/Osteosarcoma_01.czi') # 6D image
img1= img[0,0,:,:,:,0]

print(img1.shape)

img2=img1[2,:,:]

import  matplotlib.pyplot as plt

plt.imshow(img2,cmap='hot')
plt.show()