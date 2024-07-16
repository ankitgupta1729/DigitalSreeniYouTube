import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

img=cv2.imread("images/synthetic.JPG")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow("Original",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

df=pd.DataFrame()
img2=img.reshape(-1)
df['Original Pixels']=img2

num=1
for theta in range(2):
    theta=(theta/4)*np.pi
    for sigma in range(3,5):
        for lamda in np.arange(0,np.pi,np.pi/4):
            for gamma in (0.05,0.5):
                #print(theta,sigma,lamda,gamma)
                gabor_label='Gabor'+str(num)
                kernel= cv2.getGaborKernel((5,5),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
                fimg=cv2.filter2D(img,cv2.CV_8UC3,kernel) # 2D convolved image
                filtered_img=fimg.reshape(-1)
                df[gabor_label]=filtered_img
                num+=1

#print(df.head())

print(df.to_csv("Gabor.csv",index=False))