# This library is dedicated for opencv types of applications. It can be images
# real-time videos etc

# It is used for many types of applications like facial recognition, object detection,
# motion tracking, OCR etc

import cv2

img= cv2.imread("images/RGBY.jpg",1)

print(img.shape)
print("TOP LEFT", img[0,0])
print("TOP RIGHT", img[0,414])
print("BOTTOM LEFT", img[585,0])
print("BOTTOM RIGHT", img[585,414])

# In opencv when we read color images then convention is BGR, not RGB.

# blue=img[:,:,0]
# green=img[:,:,1]
# red=img[:,:,2]

blue,green,red = cv2.split(img)

# cv2.imshow("blue pixels",blue)
# cv2.imshow("green pixels",green)
# cv2.imshow("red pixels",red)

# merged_img=cv2.merge((blue,green,red))
# cv2.imshow("Merged Image",merged_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Resize image

img=cv2.imread("images/monkey.jpg",1)

resized_img=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
cv2.imshow("original image",img)
cv2.imshow("resized image",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()