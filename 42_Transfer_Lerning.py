# An alternate way of increasing accuracy is: we tke the existing images and the
# data generation (data gen) for ex. scikit learn then flip or rotate the image and
# do the image transformation operation to simulate multiple images from one single
# image, this is data augmentation and this is proven to work but nothing works like
# lots of data i.e. annotated images but often times we don't have this luxury.

# transfer learning is a way of what actually have been done and transferring it into
# a different domain. For example VGG16, it is trained on multiple images of cat, dogs etc.

# VGG16 is trained on thousands of thousands images with multiple epochs where I believe accuracy
# is above 99% for around 10k classes. But you can say this is not trained on say
# neuron or cell segmentation and you are correct but here we use convolutional layers
# from which we can find edges or curviness etc. So, instead of image classification, we can take
# the feature exatraction part and take weights and then chop off and the decoder part
# of the image data.

#####################################################################################

from  keras.applications.vgg16 import VGG16

model=VGG16()
#print(model.summary())  here we are not chopping anything will do in next file

from tensorflow.keras.utils import load_img

image = load_img("images/monkey.jpg",target_size=(224,224))

from tensorflow.keras.utils import img_to_array

image=img_to_array(image)
print(image.shape)

# here we have used one image but generally we use multiple images

import numpy as np
image = np.expand_dims(image,axis=0)

from keras.applications.vgg16 import preprocess_input
image=preprocess_input(image)

pred=model.predict(image)

from keras.applications.mobilenet import  decode_predictions
pred_classes=decode_predictions(pred,top=5)
for i in pred_classes[0]:
    print(i)





## As transfer learning suggests use one technique like vgg16 and chop off some layers and
# transfer its learning to other applications

# here we use colorizing images using some part of vgg16

# In VGG16, first 19 layers are convolutional layers and they are for feature extraction
# here we use first 19 layers of vgg16 for feature extraction and then use our own custom decoder
# final output is A and B channels from LAB (L means lighting)
from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator

from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import keras
import os

from  keras.applications.vgg16 import VGG16

vggmodel=VGG16()
#print(vggmodel.summary())
newmodel=Sequential()

for i, layer in enumerate(vggmodel.layers):
    if i<19:          #Only up to 19th layer to include feature extraction only
      newmodel.add(layer)
print(newmodel.summary())
for layer in newmodel.layers:
  layer.trainable=False   #We don't want to train these layers again, so False.

path = 'images/colorization/'
# Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)

train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode=None)

# Convert from RGB to Lab
"""
by iterating on each image, we convert the RGB to Lab. 
Think of LAB image as a grey image in L channel and all color info stored in A and B channels. 
The input to the network will be the L channel, so we assign L channel to X vector. 
And assign A and B to Y.

"""

X = []
Y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:, :, 0])
        Y.append(lab[:, :, 1:] / 128)  # A and B values range from -127 to 128,
        # so we divide the values by 128 to restrict values to between -1 and 1.
    except:
        print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape + (1,))  # dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)

# now we have one channel of L in each layer but, VGG16 is expecting 3 dimension,
# so we repeated the L channel two times to get 3 dimensions of the same L channel

vggfeatures = []
for i, sample in enumerate(X):
    sample = gray2rgb(sample)
    sample = sample.reshape((1, 224, 224, 3))
    prediction = newmodel.predict(sample)
    prediction = prediction.reshape((7, 7, 512))
    vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

# Decoder
model = Sequential()

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()

model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
model.fit(vggfeatures, Y, verbose=1, epochs=10, batch_size=128)

model.save('colorize_autoencoder_VGG16.model')

############################################
# Predicting using saved model.
model = tf.keras.models.load_model('colorize_autoencoder_VGG16_10000.model',
                                   custom_objects=None,
                                   compile=True)
testpath = 'images/colorization2/test_images/'
files = os.listdir(testpath)
for idx, file in enumerate(files):
    test = img_to_array(load_img(testpath + file))
    test = resize(test, (224, 224), anti_aliasing=True)
    test *= 1.0 / 255
    lab = rgb2lab(test)
    l = lab[:, :, 0]
    L = gray2rgb(l)
    L = L.reshape((1, 224, 224, 3))
    # print(L.shape)
    vggpred = newmodel.predict(L)
    ab = model.predict(vggpred)
    # print(ab.shape)
    ab = ab * 128
    cur = np.zeros((224, 224, 3))
    cur[:, :, 0] = l
    cur[:, :, 1:] = ab
    imsave('images/colorization2/vgg_result/result' + str(idx) + ".jpg", lab2rgb(cur))

