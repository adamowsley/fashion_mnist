from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Download the fashion images
fashion_mnist = keras.datasets.fashion_mnist

# Extract the training and test sets from the load
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Image Shape: {0}".format(train_images.shape))

# We have 60000 training images sized 28 x 28
# One (1) denotes grayscale images
train_images = train_images.reshape((60000, 28, 28, 1))

# We have 10000 test images
test_images = test_images.reshape((10000, 28, 28, 1))

print("Image Reshape: {0}".format(train_images.shape))

# Normalize every pixel value in every image to a real number between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the sequential model
model = keras.Sequential()

# Specify first layer
# The convolution filter/kernel is a 3 x 3 matrix that detects features in an image. We have 64 
# filters in our first layer.
# Activation is the Recified Linear Unit (RELU), so f(x) = x for values > 0; f(x) = 0 for values < 0.
# Specify the input_shape once for only one image (not 60000 images)
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling/Subsampling/Downsampling reduces the dimensionality of our feature maps; ours is a
# 2 x 2 matrix. Note one of the benefits of convolution and pooling when viewing the display of the 
# model summary below; we are reducing the number of inputs. Think about how this affects large
# images and computation/train/predict times.
model.add(keras.layers.MaxPooling2D((2, 2)))

# Dropout is a method to help prevent overfitting. Nodes can be dropped which reduces the number of nodes
# that need to be trained.
model.add(keras.layers.Dropout(0.05))

# Specify second layer
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.1))

# Specify third layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.1))

# Feed the last output tensor of shape (3, 3, 64) into one or more dense layers to perform the classification.
# Dense layers take the flattened 1D input.
# The final dense layer in this example has 10 classification outputs along with the softmax 
# activation function. See definition of class_names in the code below.
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# Output of all Conv2D and MaxPooling2D layers is a 3D tensor of shape (height, width, and channel)
# Width and height typically shrink as we go deeper into the network, so we can computationally 
# afford to add more output channels in each Conv2D layer.
model.summary()

# Compile the model
# The optimizer is sadly not named after me. It stands for Adaptive Moment Estimation and is a 
# stochastic optimizer that is efficient and requires little memory. See the Keras documentation for 
# further information.
# sparse_categorical_crossentropy loss accepts values between 0 and 1
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train using the images and their corresponding labels
# epochs lets us select how many training runs we'll use
train_model = model.fit(train_images, train_labels, epochs=3)

# Evaluate the model using the test images and corresponding labels
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test loss: {0:3.2f}%".format(test_loss*100))
print("Test accuracy: {0:3.2f}%".format(test_acc*100))

# Save the trained model for use in another application using the HDF5 format
model.save("trained_model.h5")

print("Save trained model to trained_model.h5")