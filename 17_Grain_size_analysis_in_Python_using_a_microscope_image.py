# we have an image which has a whole bunch of grains and we need to do some statistics
# about it i.e. what is grain size distribution looks like.

"""
Step 1: Read image and define pixel size (if needed convert results into microns)
Step 2: denoising, if required and threshold image to separate grains from boundaries.
Step 3: Clean up image, if needed (erode etc.) and create a mask for grains
Step 4: Label grains in the masked image
Step 5: Measure the properties of each grain (object)
Step 6: Output results into a csv file
"""

import cv2
import  numpy as np
import  matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io,color, measure

# step 1
img=cv2.imread("images/grains2.jpg",0)
pixels_to_um=0.5 # um means micrometer. 1 pixel= 0.5 um or 1 pixel = 500 nm

# step 2
#plt.hist(img.flat,bins=100,range=(0,255))
#plt.show()

ret,thresh=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('thresholded image',thresh)
# cv2.waitKey(0)

# step 3

# clean up the image

kernel=np.ones((3,3),np.uint8)
eroded=cv2.erode(thresh,kernel,iterations=1)
#cv2.imshow("eroded images",eroded)

dilated=cv2.dilate(eroded,kernel,iterations=1)
#cv2.imshow('dilated image',dilated)

mask = dilated==255

#io.imshow(mask) # cv2.imshow does not show binary image

# step 4 Labeling

s = [[1,1,1],[1,1,1],[1,1,1]] # 8-connectivity
label_mask,num_labels=ndimage.label(mask,structure=s)

img2=color.label2rgb(label_mask,bg_label=0)

#cv2.imshow("colored labels",img2)
#cv2.waitKey(0)

# step 5 measuring

clusters= measure.regionprops(label_mask,img)

print(clusters[0].perimeter)

for props in clusters:
    print('Label: {} Area: {}'.format(props.label,props.area))

# step 6 dump into csv file

propList=['Area','equivalent_diameter','orientation','MajorAxisLength',
          'MinorAxisLength','Perimeter','MinIntensity','MeanIntensity','MaxIntensity']

output_file=open('image_measurements.csv','w')
output_file.write(','+','.join(propList)+'\n')

for cluster_props in clusters:
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if prop=='Area':
            to_print=cluster_props[prop]*pixels_to_um**2
        elif prop=='orientation':
            to_print=cluster_props[prop]*57.2958 # convert degree from radians
        elif prop.find('Intensity')<0:
            to_print=cluster_props[prop]*pixels_to_um
        else:
            to_print=cluster_props[prop]
        output_file.write(','+str(to_print))
    output_file.write('\n')