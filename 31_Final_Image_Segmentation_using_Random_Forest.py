import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  glob
import  pickle
def feature_extraction(img):
    df = pd.DataFrame()

    # The most important feature would be the pixel values. For example, if pixel value>400,
    # then it might be bright pixel.

    # Add original pixel values to the dataframe as feature #1
    img2 = img.reshape(-1)
    df['Original Image'] = img2

    # Add other features

    # Set-1. Gabor Features (my favorite)- It is like a Gaussian or canny edge detection filters

    num = 1
    kernels = []
    for theta in range(2):
        theta = (theta / 4) * np.pi
        for sigma in range(1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):  # 0.05 gives high aspect ratio kernel and 0.5 gives low aspect ratio kernel
                    # print(theta,sigma,lamda,gamma)
                    gabor_label = 'Gabor' + str(num)
                    ksize = 5
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)  # 2D convolved image
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1

    #print(df.head())

    #####################################################################################

    # Set 2: Canny Edge Detector

    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1

    #####################################################################################

    # set 3: other filters

    from skimage.filters import roberts, sobel, scharr, prewitt

    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    # Gaussian with sigma=3

    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    # Gaussian with sigma=7

    from scipy import ndimage as nd
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    # Median with sigma=3

    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    return df

filename='sandstone.pkl'
load_model = pickle.load(open(filename, 'rb'))

path= 'images/Train_images/*.tif'
for file in glob.glob(path):
    img1 = cv2.imread(file)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    X=feature_extraction(img)
    result=load_model.predict(X)
    segmented=result.reshape((img.shape))
    name=file.split("e_")
    plt.imshow(segmented,cmap='jet')
    plt.show()
    plt.imsave('unknown_file.jpg',segmented,cmap='jet')
