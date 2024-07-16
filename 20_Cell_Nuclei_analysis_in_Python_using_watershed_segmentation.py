# Here we have a biological image from a biological sample that contains a whole bunch of
# cells

# download zen lite software and see some insights

# watershed algorithm is very powerful way to segment different grains or different cells
# or different regions that are difficult to segment using regular thresholding approach

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure

img1 = cv2.imread("images/Osteosarcoma_01.tif")
img = img1[:,:,0]

# cv2.imshow("blue img", img)
# cv2.waitKey(0)
pixels_to_um = 0.454  # um means micrometer. 1 pixel= 0.5 um or 1 pixel = 500 nm

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)

# performing opening operation

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

from skimage.segmentation import clear_border

opening = clear_border(opening)

# cv2.imshow("opening image",opening)
# cv2.waitKey(0)

sure_bg = cv2.dilate(opening, kernel, iterations=10)

# cv2.imshow("Sure Background",sure_bg)
# cv2.waitKey(0)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

# cv2.imshow("Distance Transform",dist_transform)
# cv2.waitKey(0)

ret2, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
#
# cv2.imshow('Sure Foreground',sure_fg) # this image shows that I am sure that
# # pixels correspond to the grains
# cv2.waitKey(0)

unknown = cv2.subtract(sure_bg, sure_fg) # unknown area

# cv2.imshow("unknown",unknown) # inverse image of sure foreground
# cv2.waitKey(0)

ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers + 10  # markers is an ndarray

markers[unknown == 255] = 0

# plt.imshow(markers,cmap='jet')
# plt.show()

markers = cv2.watershed(img1, markers)

img1[markers == -1] = [0, 255, 255]

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay Original Image',img1)
cv2.imshow('Colored grains',img2)
cv2.waitKey(0)

regions = measure.regionprops(markers, intensity_image=img)
propList = ['Area', 'equivalent_diameter', 'orientation', 'MajorAxisLength',
            'MinorAxisLength', 'Perimeter', 'MinIntensity', 'MeanIntensity', 'MaxIntensity']

output_file = open('image_measurements3.csv', 'w')
output_file.write('Cell #' + ','.join(propList) + '\n')

grain_number = 1
for region_props in regions:
    output_file.write(str(grain_number) + ',')
    for i, prop in enumerate(propList):
        if prop == 'Area':
            to_print = region_props[prop] * pixels_to_um ** 2
        elif prop == 'orientation':
            to_print = region_props[prop] * 57.2958  # convert degree from radians
        elif prop.find('Intensity') < 0:
            to_print = region_props[prop] * pixels_to_um
        else:
            to_print = region_props[prop]
        output_file.write(',' + str(to_print))
    output_file.write('\n')
    grain_number += 1

output_file.close()