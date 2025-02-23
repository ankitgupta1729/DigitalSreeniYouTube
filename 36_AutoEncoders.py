# AutoEncoders are neural networks in which architecture is designed in such a way that
# where the target output is input.

# In Encoding phase, input image is encoded into much smaller dimensions and they can be decoded
# into original input image that is called reconstructed image.
# Due to some reconstruction loss, output is not exactly same as input but it is approximately same

# Generally it has application in noise reduction wrt image processing
# Consider X as a noisy image and Y as a clean image

# Another application is Anamoly Detection. For example if we have a whole bunch of data
# and it has a lot of failures then we can predict failures. In case of anamoly, reconstruction
# error/loss is very high

# Another application is domain adaptation, for example, x=einstein image and y=monalisa image
# it is possible when we have enough data points

# Another example is Image colorization to color the input gray image by model training.

################################################################################################

from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

np.random.seed(42)

SIZE = 256
img_data = []

img = cv2.imread('images/monalisa.jpg', 1)  # Change 1 to 0 for grey images
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB to show images in true colors
img = cv2.resize(img, (SIZE, SIZE))
img_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

model.fit(img_array, img_array,
          epochs=5000,
          shuffle=True)

print("Neural network output")
pred = model.predict(img_array)

imshow(pred[0].reshape(SIZE, SIZE, 3), cmap="gray")

##############################################################################################


